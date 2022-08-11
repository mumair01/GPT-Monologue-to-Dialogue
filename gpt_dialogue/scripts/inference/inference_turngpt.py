# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-08 14:49:25
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-11 12:09:55

#############################################################
'''
Inference script for TurnGPT specifically. This script basically
calls the same inference methods as the transformers inference script but
specifically for TurnGPT.
'''
#############################################################

import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd

import pytorch_lightning as pl

from gpt_dialogue.turngpt.model import (
    TurnGPTDoubleHeadsModel,
    TurnGPTLMHeadModel
)
from gpt_dialogue.scripts.inference.utils import (
    load_inference_dataset,
    generate_conditional_probs
)


import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --------------------------- GLOBALS --------------------------------------

HYDRA_CONFIG_RELATIVE_DIR = "../../../conf"
HYDRA_CONFIG_NAME = "inference_turngpt"

# ------------------------ INFERENCE MAIN  --------------------------------


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def turngpt_inference(cfg : DictConfig):
    logger.info(f"Loading TurnGPT for inference {cfg.inference.model.pretrained_model_name_or_path}")

     # Load the appropriate model
    if cfg.inference.model.model_head == "DoubleHeads":
        if os.path.isfile(cfg.inference.model.pretrained_model_name_or_path):
            model = TurnGPTDoubleHeadsModel.load_from_checkpoint(cfg.inference.model.pretrained_model_name_or_path)
        else:
            model = TurnGPTDoubleHeadsModel(**cfg.inference.model)
    elif cfg.inference.model.model_head == "LMHead":
        if os.path.isfile(cfg.inference.model.pretrained_model_name_or_path):
            model = TurnGPTLMHeadModel.load_from_checkpoint(cfg.inference.model.pretrained_model_name_or_path)
        else:
            model = TurnGPTLMHeadModel(**cfg.inference.model)
    else:
        raise NotImplementedError(
            f"TurnGPT with head {cfg.model.model_head} has not been implemented"
        )

    logging.info(f"Loading inference dataset: {cfg.dataset.test_path}")
    conversation_dfs = load_inference_dataset(
        csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.test_path),
        **cfg.inference.dataset
    )
    os.makedirs("sample_results",exist_ok=True)
    data = []
    df_columns = [
        'conversationName','conversationNumber', 'turnNumber','wordNumber','context',
        'word', 'probability']
    for i, conversation_df in enumerate(conversation_dfs):
        results = generate_conditional_probs(
            model=model,
            tokenizer=model._tokenizer,
            conversation_df=conversation_df,
            conv_no=i,
            **cfg.inference.inference
            )
        # Load the data as a single dataframe and save (important if the
        # program crashes).
        save_path = os.path.join(os.getcwd(),
                "{}_conditional_probs.csv".format(results[0][0]))
        pd.DataFrame(results,columns=df_columns).to_csv(save_path)
        logger.info(f"Saving results: {save_path}")
        data.extend(results)
    logger.info("Complete!")

if __name__ == "__main__":
    turngpt_inference()