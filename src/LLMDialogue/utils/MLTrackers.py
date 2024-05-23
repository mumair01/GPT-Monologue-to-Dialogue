# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-22 13:57:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 09:09:39


import sys
import os
import wandb

from typing import Dict, Any, List, Tuple, Optional, Callable


class MLTracker:

    def start_run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def finish_run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


class WANDBTracker(MLTracker):
    """
    Wrapper class that helps manage a wandb run manually.
    """

    def __init__(self):
        super().__init__()

    #####################
    # PUBLIC METHODS
    #####################

    def start_run(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        mode: Optional[str] = None,
        run_name: Optional[str] = None,
        logging_dir: Optional[str] = None,
        log_model: Optional[str] = "end",
        **wandb_init_kwargs,
    ):

        # Log in if needed.
        if api_key is not None:
            wandb.login(anonymous="never", key=api_key, verify=True)

        # If the log_model is specified, then set that as a param.
        if log_model is not None:
            os.environ["WANDB_LOG_MODEL"] = log_model
        # Initialize the run and save the run object.
        self.run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=run_name,
            settings=wandb.Settings(start_method="fork"),
            dir=logging_dir,
            **wandb_init_kwargs,
        )

    def finish_run(self):
        self.run.finish()

    def download_artifact(
        self,
        project: str,
        entity: str,
        artifact_name: str,
        alias: Optional[str] = "latest",
        download_path: Optional[str] = None,
    ) -> str:
        api = wandb.Api()
        id_ = f"{entity}/{project}/{artifact_name}:{alias}"
        artifact = api.artifact(id_)
        return artifact.download(root=download_path)

    # def download_saved_artifact_dir(
    #     self,
    #     run_name: str,
    #     tag: Optional[str] = "latest",
    # ) -> str:
    #     with self.run:
    #         name = f"{run_name}:{tag}"
    #         artifact = self.run.use_artifact(name)
    #         # Download the model directory that can be loaded
    #         artifact_dir = artifact.download()
    #     return artifact_dir

    #####################
    # PRIVATE METHODS
    #####################
