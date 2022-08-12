# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-12 13:52:49


import sys
import os

from omegaconf import DictConfig, OmegaConf
import hydra


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.conditional_probs import (
    load_inference_dataset,
    process_conversation_dfs
)


########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"

########################### MAIN METHODS ####################################

@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Load the appropriate model
    if cfg.experiment.name == "inference_monologue_gpt":
        model = MonologueGPT()
    elif cfg.experiment.name == "inference_turngpt":
        model = TurnGPT()
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    model.load(**cfg.experiment.load)

    # Generate the conditional probs.
    conversation_dfs = load_inference_dataset(
        csv_path = os.path.join(cfg.env.paths.root,cfg.dataset.dataset_path),
        **cfg.experiment.dataset
    )

    process_conversation_dfs(
        save_dir = os.getcwd(),
        model=model,
        tokenizer=model.tokenizer,
        conversation_dfs=conversation_dfs,
        **cfg.experiment.inference
    )

if __name__ == "__main__":
    main()
