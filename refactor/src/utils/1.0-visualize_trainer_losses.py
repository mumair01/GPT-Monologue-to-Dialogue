# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-07 14:27:54
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-07 14:48:34

#############
# Quick script to visualize the training and evaluation losses when given
# the trainer_state.json file that is obtained when training using huggingface
# Trainer.
#############

from mimetypes import init
import os
import sys
import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def save_fig(save_dir,fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(save_dir, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def parse_trainer_state(trainer_state_path):
    with open(trainer_state_path, "r") as f:
        try:
            data = json.load(f)
            log_history = data["log_history"]
            # Separate the training and evaluation losses
            train_losses = {}
            eval_losses = {}
            for log in log_history:
                if 'loss' in log:
                    train_losses[log['epoch']] = log['loss']
                if 'eval_loss' in log:
                    eval_losses[log['epoch']] = log['eval_loss']
        except:
            raise Exception("Could not parse trainer_state: {}".format(
                trainer_state_path))
    return train_losses, eval_losses

def visualize(save_dir, fig_name, train_losses, eval_losses, init_epoch):
    plt.figure()
    plt.plot(list(train_losses.keys())[init_epoch:], list(train_losses.values())[init_epoch:],
        label="training loss")
    plt.plot(list(eval_losses.keys())[init_epoch:], list(eval_losses.values())[init_epoch:],
        label="evaluation_loss")
    plt.title("GPT-2 Finetuning Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_fig(save_dir,fig_name if fig_name != None else "gpt2-finetuning-losses")
    plt.show()

def main(args):
    assert os.path.isfile(args.trainer_state)
    train_losses, eval_losses = parse_trainer_state(args.trainer_state)
    visualize(args.save_dir, args.fig_name, train_losses, eval_losses,args.init_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer-state",type=str,required=True,
        help="Path to the trainer_state.json file (contained in a checkpoint)")
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Save file name (typically a .png extension")
    parser.add_argument(
        "--fig-name", type=str, help="Figure output name")
    parser.add_argument(
        "--init-epoch", type=int,default=0)
    args = parser.parse_args()
    main(args)