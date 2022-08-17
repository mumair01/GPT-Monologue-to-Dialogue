# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-12 13:17:15

import sys
import os

from omegaconf import DictConfig, OmegaConf
import hydra

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT


########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"

########################### MAIN METHODS ####################################


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Load the appropriate model
    if cfg.experiment.name == "finetune_monologue_gpt":
        model = MonologueGPT()
    elif cfg.experiment.name == "finetune_turngpt":
        model = TurnGPT()
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    model.load(**cfg.experiment.load)

    # Finetune
    model.finetune(
        train_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.train_csv_path),
        val_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.validation_csv_path),
        save_dir = os.getcwd(),
        **cfg.experiment.finetune
    )

if __name__ == "__main__":
    main()