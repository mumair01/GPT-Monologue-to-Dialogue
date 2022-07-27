# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 10:26:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-27 15:07:08


############################
# This module is a re-implementation of the TurnGPT model as a comparison to the
# speaker identity embedding approach that we have taken.
# NOTE: This code is taken from the repository specified below and may or may
# not be modified.
# Acknowledgements:
#   Paper: https://arxiv.org/abs/2010.10874
#   Code: https://github.com/ErikEkstedt/TurnGPT
############################


# NOTE: This script is requires torch 1.12.

from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

import pytorch_lightning as pl
import torch
import torch.nn as nn

import einops

# NOTE: Change this to absolute import when using as a package
from gpt_dialogue.turngpt.tokenizer import SpokenDialogTokenizer


def load_transformer(
        pretrained_model_name_or_path="gpt2-large", pretrained=True, **model_kwargs):

    _IMPLEMENTED = [
        "gpt2-large"
    ]

    _UPDATE_ON_PRETRAIN = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]

    assert pretrained_model_name_or_path.lower() in _IMPLEMENTED, \
        f"{pretrained_model_name_or_path} not supported - must be one of {_IMPLEMENTED}"

    config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
    if pretrained:
        for k, v in model_kwargs.items():
            if k in _UPDATE_ON_PRETRAIN and v is not None:
                config.update({k : v})
        transformer = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config
        )
    else:
        for k, v in model_kwargs.items():
            if v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel(config=config)
    return transformer



class Utils:
    pass


class TurnGPT(pl.LightningModule):

    def __init__(self,
            pretrained_model_name_or_path="gpt2",
            pretrained=True,
            trp_projection_steps=-1,
            trp_projection_type="linear",
            omit_dialog_states=False,
            no_train_first_n=5,
            learning_rate=1e-4,
            weight_loss=False,
            weight_regular_token=0.5,
            weight_eos_token=1.0,
            **model_kwargs):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        self.pretrained = pretrained

        # train parameters
        self.no_train_first_n = no_train_first_n
        self.learning_rate = learning_rate
        self.weight_loss = weight_loss
        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.omit_dialog_states = omit_dialog_states

        # Load `transformers` model
        self.transformer = load_transformer(
            pretrained_model_name_or_path, pretrained=pretrained, **model_kwargs
        )

        # TRP projection head
        self.trp_projection_steps = trp_projection_steps
        if trp_projection_steps > 0:
            self.trp_projection_type = trp_projection_type
            hidden_size = self.transformer.config.hidden_size

            # MultiTask Head operating on n last hidden states
            if trp_projection_type.lower() == "attention":
                raise NotImplementedError()
            else:
                self.trp_projection_head = nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    ############ ADDITIONAL METHODS

    @property
    def run_name(self):
        name = "TurnGPT"
        if self.trp_projection_steps > 0:
            name += f"_proj_{self.trp_projection_steps}"
        return name

    def init_tokenizer(self):
        # The tokenizer should always be a part of the model
        self.tokenizer = SpokenDialogTokenizer(self.name_or_path)

        # Add extra embeddings for custom tokens
        # Optional: Initialize <ts> to be close to punctuation tokens.
        self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

    def initialize_special_embeddings(self, tokens=["!", "?", "."]):
        """
        Initialize `eos_token` as the average of `tokens`.

        By default (or looking at <speaker1/2>) the embeddings are initalized to m=0, std=0.02
        """
        ts = self.tokenizer.eos_token_id
        # pre = self.transformer.transformer.wte.weight[ts].clone()
        with torch.no_grad():
            ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            avg_emb = self.transformer.transformer.wte(ids).mean(0)
            self.transformer.transformer.wte.weight.data[ts] = avg_emb
        # post = self.transformer.transformer.wte.weight[ts]
        # print(pre == post)
        print(f"Initalized {self.tokenizer.eos_token} -> avg({tokens})")


    def get_labels(self, input_ids, mask, value=-100):
        """Don't shift the labels (happens internally)"""
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = value

        if self.no_train_first_n > 0:
            labels[:, : self.no_train_first_n] = value
        return labels


    @torch.no_grad()
    def get_loss_weight(self):
        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float)
            * self.weight_regular_token
        )
        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        return weight.to(self.device)

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        weight = None
        if self.weight_loss:
            weight = self.get_loss_weight()

        loss_fct = nn.CrossEntropyLoss(weight=weight, reduction="none")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens and calc loss
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if reduction != "none":
            loss = loss.mean()
        return loss

    def bce_loss(self, logits, labels):
        """
        Simple BCELoss for binary trp projection

        Must extend this if multiple labels are to be used...
        """
        loss_fct = nn.BCEWithLogitsLoss()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1]  # , :].contiguous()
        shift_labels = labels[..., 1:]  # .contiguous()

        # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100
        loss = loss_fct(
            torch.masked_select(shift_logits, indices_for_training),
            torch.masked_select(shift_labels, indices_for_training),
        )
        # shift_logits = torch.masked_select(shift_logits, indices_for_training)
        # shift_labels = torch.masked_select(shift_labels, indices_for_training)
        # loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_likelihood(self, logits, labels, pad_last=True, pad_first=False):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_probs = shift_logits.softmax(dim=-1)

        # likelihood = shift_probs[shift_labels]
        bn = torch.ones_like(shift_labels)
        bn[0] = 0

        seq = torch.arange(shift_labels.shape[-1])
        seq = torch.stack([seq] * shift_labels.shape[0])

        likelihood = shift_probs[bn.view(-1), seq.view(-1), shift_labels.view(-1)]
        likelihood = likelihood.view(shift_labels.shape)
        if pad_first:
            likelihood = torch.cat(
                [torch.zeros(likelihood.shape[0], 1), likelihood], dim=-1
            )
        elif pad_last:
            likelihood = torch.cat(
                [likelihood, torch.zeros(likelihood.shape[0], 1)], dim=-1
            )
        return likelihood

    ############ OVERLOADED METHODS

    def forward(self,
            input_ids=None,
            speaker_ids=None,
            labels=None,
            mc_labels=None,
            use_cache=None,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )

        transformer_outputs = self.transformer.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=speaker_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.transformer.model_parallel:
            torch.cuda.set_device(self.transformer.transformer.first_device)
            hidden_states = hidden_states.to(self.transformer.lm_head.weight.device)

        # Language Modeling
        lm_logits = self.transformer.lm_head(hidden_states)
        lm_loss = None
        if labels is not None:
            lm_loss = self.cross_entropy_loss(lm_logits, labels)

        # MultiTask Modeling
        mc_logits = None
        mc_loss = None
        if self.trp_projection_steps > 0:
            # NOTE:
            # Assumed to only guess a single class
            mc_logits = self.trp_projection_head(hidden_states).squeeze(-1)

            if mc_labels is not None:
                mc_loss = self.bce_loss(mc_logits, mc_labels)

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            print("#" * 70)
            print("LOAD CHECKPOINT TOKENIZER")
            self.tokenizer = checkpoint["tokenizer"]
            print("Loaded tokenizer")
            print(self.tokenizer)

            # Add extra embeddings for custom tokens
            self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
            print("Resized weights")
            print("#" * 70)

    def training_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )

        if self.trp_projection_steps > 0:
            self.log("loss_lm", out["loss"])
            self.log("loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            self.log("loss", out["loss"])
            total_loss = out["loss"]
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )

        if self.trp_projection_steps > 0:
            self.log("val_loss_lm", out["loss"])
            self.log("val_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            total_loss = out["loss"]

        self.log("val_loss", total_loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="gpt2")
        parser.add_argument(
            "--pretrained",
            type=bool,
            default=True,
            help="Load pretrained weights or not.",
        )
        # Model specific
        parser.add_argument("--embd_pdrob", type=float, default=None)
        parser.add_argument("--attn_pdrob", type=float, default=None)
        parser.add_argument("--resid_pdrob", type=float, default=None)
        parser.add_argument("--n_head", type=int, default=None)
        parser.add_argument("--n_layer", type=int, default=None)
        parser.add_argument("--n_embd", type=int, default=None)
        parser.add_argument("--activation_function", type=str, default=None)

        # TurnGPT specific
        parser.add_argument(
            "--omit_dialog_states",
            action="store_true",
            help="To omit dialog-states in transformer embedding",
        )
        parser.add_argument("--trp_projection_steps", default=-1, type=int)
        parser.add_argument(
            "--no_train_first_n",
            default=-1,
            type=int,
            help="Don't train on the n first tokens.",
        )
        parser.add_argument(
            "--trp_projection_type",
            default="linear",
            type=str,
            help="'Linear' or 'Attention'",
        )

        # Training
        parser.add_argument(
            "--dropout",
            default=None,
            type=float,
            help="Set to None which uses the values in the original config.json",
        )
        parser.add_argument("--learning_rate", default=6.25e-5, type=float)
        parser.add_argument("--weight_loss", action="store_true")
        parser.add_argument("--weight_eos_token", type=float, default=1.0)
        parser.add_argument("--weight_regular_token", type=float, default=0.5)
        return parser


if __name__ == "__main__":
    model = TurnGPT(
        pretrained_model_name_or_path="gpt2-large"
    )
    print(model)





