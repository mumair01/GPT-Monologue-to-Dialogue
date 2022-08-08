# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 10:26:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-08 10:29:28


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

import os
import sys
from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, Optional, Tuple

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from gpt_dialogue.turngpt.tokenizer import SpokenDialogueTokenizer


"""
Below are some general notes based on my understanding of the original TurnGPT:

    1. It uses a DoubleLMHead since the model is multi-task i.e., it is optimizing
        on the sum of two different loss functions and can perform multiple tasks
        during inference.
    2. The speaker tokens are included in the language modelling task and the
        TRP probability predictions are defined as the maximum assigned
        output probability over the speaker tokens --> This is why we use
        multi-task loss and a double head.
    3. In the original implementation, they are predicting the TRP token
        in the next N steps (or after the next N tokens). Do I really need to
        do that? My use case is simply to get the P(token | context) on a model
        that is sensitive to the speaker --> So this task is different from
        TurnGPT.
"""


# TODO: Investigate how this labeler is working.
class ProjectionLabeler(nn.Module):
    """
    This module generates labels for the trp projection task based on the input_ids.
    """

    def __init__(self, projection_steps, token_id):
        super().__init__()
        self.projection_steps = projection_steps
        self.token_id = token_id
        self.labeler = nn.ConvTranspose1d(
            in_channels=1, out_channels=1, kernel_size=projection_steps, bias=False
        )
        self.labeler.weight.data.fill_(1.0)
        self.labeler.requires_grad_(False)
        self.offset = projection_steps - 1

    def forward(self, input_ids):
        # prepare inputs
        eos = (input_ids == self.token_id) * 1.0
        eos = eos.unsqueeze(1)
        # construct labels
        proj_label = self.labeler(eos)
        # offset the transpose-output and remove channel
        proj_label = proj_label[..., self.offset :].squeeze(1)
        return proj_label

class TurnGPT(pl.LightningModule):
    """
    Lightning Data Module acting as a wrapper used to extend a huggingface
    Transformer.
    """

    # Base GPT-2 models supported by TurnGPT.
    _SUPPORTED_PRETRAINED_MODELS = [
        "gpt2",
        "gpt2-large",
        "distilgpt2"
    ]

    # Parameters of the GPT-2 Config that will be re-trained / updated
    # even if loading pretrained configurations.
    # Link: https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/gpt2#transformers.GPT2Config
    _CONFIG_PRETRAIN_UPDATE_PARAMS = [
        "embed_pdrop",
        "attn_pdrop",
        "resid_pdrop"
    ]

    def __init__(self,
            pretrained_model_name_or_path : str = "gpt2",
            load_pretrained_configs : bool = True,
            trp_projection_steps : int =-1,
            trp_projection_type : str = "linear",
            omit_speaker_ids : bool = False,
            no_train_first_n : int = 5,
            learning_rate=1e-4,
            **kwargs):
        """
        Args:
            pretrained_model_name_or_path (str):
                Name of the underlying pretrained model.
            load_pretrained_configs (bool):
                If True, uses pretrained values for model configuration. If
                False, uses configuration values passed in.
                Link: https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/gpt2#transformers.GPT2Config
            trp_projection_steps (int):
                Number of projection steps for the TRPs
            trp_projection_type (str): The type of the trp projection head
                - currently only 'linear' is supported.
            omit_speaker_ids (bool):
                If True, dialogue ids are not passed to the model during the
                forward pass i.e., it works as a regular GPT-2 model.
            no_train_first_n (int):
                Defines the first n input_ids every step that are going to be masked.
            learning_rate (float): Learning rate for the underlying optimizer.

        """
        super().__init__()
        # Save the params
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_pretrained_configs = load_pretrained_configs
        self.trp_projection_steps = trp_projection_steps
        self.trp_projection_type = trp_projection_type
        self.omit_speaker_ids = omit_speaker_ids
        self.no_train_first_n = no_train_first_n
        self.learning_rate = learning_rate

        # Verify that the model base is supported
        if not os.path.exists(pretrained_model_name_or_path) and \
                not pretrained_model_name_or_path in \
                    self._SUPPORTED_PRETRAINED_MODELS:
            print(f"ERROR: Provide valid path or select pre-trained model from "
                    f"{self._SUPPORTED_PRETRAINED_MODELS}")

        # ---Load the pretrained model
        config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
        if load_pretrained_configs:
            for k,v in kwargs.items():
                if k in self._CONFIG_PRETRAIN_UPDATE_PARAMS and v is not None:
                    config.update({k : v})
            self._transformer = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config)
        else:
            for k, v in kwargs.items():
                if v is not None:
                    config.update({k : v})
            self._transformer = GPT2LMHeadModel(config=config)

        # Initialize the tokenizer and resize model embeddings
        self._initialize_tokenizer()
        # Initialize the <ts> embedding.
        self._initialize_special_embeddings()
        # Initialize the TRP Projection head
        # NOTE: This is a unique contribution of TurnGPT
        self._initialize_trp_projection_head()
        # save all the hyper-params passed to the model - accessible using model.hparams.
        self.save_hyperparameters()

    ###################### LIGHTNING MODULE METHODS ##########################

    def forward(
        self,
        input_ids=None,
         # NOTE: Custom argument for speaker ids.
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
        **kwargs,
    ) -> Any:
        """
        Defines the unique computation for the model for every step - based
        on: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        In this case, this method is a re-write / wrapper for:
            https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/modeling_gpt2.py#L957
        , which is the forward method for GPT2DoubleHeadsModel.
        The unique computation here is the custom TRP projection.
        """
        return_dict = return_dict if return_dict is not None else self._transformer.config.use_return_dict

        transformer_outputs = self._transformer.transformer(
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
        if self._transformer.model_parallel:
            torch.cuda.set_device(self._transformer.first_device)
            hidden_states = hidden_states.to(self._transformer.lm_head.weight.device)

        # ---------- This part is the modification to this function.
        lm_logits = self._transformer.lm_head(hidden_states)
        mc_logits = self.trp_projection_head(hidden_states) \
            if self.trp_projection_steps > 1 else None


        # Calculate the lm task loss.
        lm_loss = None
        if labels is not None:
            # NOTE: Original model wrote a custom loss to weigh the eos token more.
            # However, I am less interested in that so using the default implementation.
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Calculate the TRP projection task loss
        mc_loss = None
        if mc_logits is not None and mc_labels is not None:
            # TODO: Implement the loss method here.
            mc_loss = self.bce_loss(mc_logits, mc_labels)

        # ---------

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
        """Initialize a single optimizer for both the transformer and projection"""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Method is called when saving a checkpoint. We save the unique tokenizer
        since it is part of the model.
        NOTE: It is unusual to call this method generally - but we make an
        exception since lightning will not save the tokenizer.
        """
        checkpoint['tokenizer'] = self._tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore any additional pickleable object saved during a checkpoint"""
        if "tokenizer" in checkpoint:
            self._tokenizer = checkpoint['tokenizer']
            self._transformer.resize_token_embeddings(
                new_num_tokens=len(self.tokenizer))

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss for one training step.
        This method does not support multi-GPU optimization.
        Here, we first get the labels for our task, do a forward pass to
        generate the combined loss, and return the values.
        """
        # Extract the task labels
        lm_labels = self._extract_labels(
            input_ids=batch['input_ids'],mask=batch['attention_mask'])

        trp_labels = None
        if self.trp_projection_steps > 0:
            trp_labels = self._extract_mc_labels(
                input_ids=batch['input_ids'],mask=batch['attention_mask'])

        if self.omit_speaker_ids:
            batch['speaker_ids'] = None

        # Do one forward pass to obtain the task losses
        out = self.forward(
            input_ids=batch['input_ids'],
            speaker_ids=batch['speaker_ids'],
            labels=lm_labels,
            mc_labels=trp_labels
        )
        # Calculate and return the total loss
        if self.trp_projection_steps > 0:
            self.log("loss_lm", out["loss"])
            self.log("loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            self.log("loss", out["loss"])
            total_loss = out["loss"]
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        """This step is for calculating and logging interesting metrics for validation"""
         # Extract the task labels
        lm_labels = self._extract_labels(
            input_ids=batch['input_ids'],mask=batch['attention_mask'])

        trp_labels = None
        if self.trp_projection_steps > 0:
            trp_labels = self._extract_mc_labels(
                input_ids=batch['input_ids'],mask=batch['attention_mask'])

        if self.omit_speaker_ids:
            batch['speaker_ids'] = None

        # Do one forward pass to obtain the task losses
        out = self.forward(
            input_ids=batch['input_ids'],
            speaker_ids=batch['speaker_ids'],
            labels=lm_labels,
            mc_labels=trp_labels
        )
        # Calculate and log te validation loss.
        if self.trp_projection_steps > 0:
            self.log("val_loss_lm", out["loss"])
            self.log("val_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            total_loss = out["loss"]
        self.log("val_loss", total_loss)

    ##################### ADDITIONAL PUBLIC METHODS ##########################

    # --- Tokenizer Utility methods
    # These methods are built on top of the tokenizer

    def tokenize(self, text ,*args, **kwargs):
        """
        Tokenize the given text 'after' applying padding.
        Moves the tokens to the available device.

        Args:
            text (str or list of strings or lists of lists)
        """
        tokens = self._tokenizer(text, *args, **kwargs)
        # NOTE: Device is a property of the lightning module.
        for k,v in tokens.items():
             tokens[k] = v.to(self.device)
        return tokens

    def decode(self, input_ids):
        return self._tokenizer.decode(input_ids)

    ######################### PRIVATE METHODS ################################

    def _initialize_tokenizer(self):
        """Create the tokenizer and resize the model embeddings"""
        self._tokenizer = SpokenDialogueTokenizer(self.pretrained_model_name_or_path)
        self._transformer.resize_token_embeddings(new_num_tokens=len(self._tokenizer))

    def _initialize_special_embeddings(self, tokens=["!", "?", "."]):
        """
        Sets the mean of the eos token as the average of the passed tokens.
        Ex. if tokens are closer to punctuation, then the word embedding for
        the special token is closer to that.
        Link: https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/modeling_gpt2.py#L679
        Link: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=nn%20embedding#torch.nn.Embedding
        """
        with torch.no_grad():
            # NOTE: self.transformer is the GPT2Model in the head that we are using.
            # Calculate the average work token embedding of the given tokens.
            ids = torch.tensor(self._tokenizer.convert_tokens_to_ids(tokens))

            avg_embedding = self._transformer.transformer.wte(ids).mean(0)
            # Assign the average as the embedding for the eos special token.
            self._transformer.transformer.wte.weight.data[
                self._tokenizer.eos_token_id] = avg_embedding
        print(f"Initialized special wte for {self._tokenizer.eos_token} to average of {tokens}")
        print(f"This will make the {self._tokenizer.eos_token} semantically closer to {tokens}")

    def _initialize_trp_projection_head(self):
        """
        Initialize the TRP projection head for the model.
        NOTE: Only a linear head is currently supported.
        """
        # TRP projection head
        # TODO: Determine what the t=project steps mean. Does -1 mean that the layer will not be initialized?
        if self.trp_projection_steps > 0:
            hidden_size = self._transformer.config.hidden_size
            # MultiTask Head operating on n last hidden states
            if self.trp_projection_type.lower() == "linear":
                self.trp_projection_head = nn.Linear(hidden_size, 1)
            else:
                raise NotImplementedError()

    def _bce_loss(self, logits, labels):
        """BCE loss to calculate a single class trp projection"""
        loss_fct = BCEWithLogitsLoss()
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
        return loss

    def _extract_labels(self, input_ids, mask, value=-100):
        """
        Obtain the labels for the language modeling task.
        NOTE: All labels set to -100 are ignored (masked) when computing the loss.
        """
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = value
        if self.no_train_first_n > 0:
            labels[:, : self.no_train_first_n] = value
        return labels

    def _extract_mc_labels(self, input_ids, mask, value=-100):
        # NOTE: This requires the projection labeler - which is a separate model
        # that can generate the TRP projection labels on the fly.
        labeler = ProjectionLabeler(
            projection_steps=self.trp_projection_steps,
            token_id=self.tokenizer.eos_token_id,
        ).to(self.device)
        proj_labels = labeler(input_ids)
        proj_labels[torch.logical_not(mask)] = value
        # TODO: Determine why we need to mask the first n tokens.
        if self.no_train_first_n > 0:
            proj_labels[:, : self.no_train_first_n] = value
        return proj_labels

