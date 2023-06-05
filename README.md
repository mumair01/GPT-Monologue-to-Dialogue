# GPT-Monologue-to-Dialogue


## Abstract

Transformer-based Large Language Models (LLMs), including ChatGPT, have recently increased in popularity. While LLMs can produce human-like writing, no study has investigated the extent to which these models can learn to predict spoken language in natural interaction. This is a non-trivial question, as spoken and written language differ in syntax and pragmatics, and interlocutors in natural dialogue follow a number of complex norms. Previous work suggests that LLMs can over-learn superficial statistical regularities but fail to learn the patterns underlying data. This implies that LLMs may not learn subtle norms in spoken dialogue, but may instead model superficial statistical regularities in speech. In this paper, we investigate whether LLMs can learn that the identity of the speaker in spoken dialogue influences what is likely to be said. 

To answer this question, we tested two variants of GPT: one with explicit representation of speaker identity and one without. We fine-tuned the models on transcripts of natural spoken dialogue. Then, we extracted the LLM-produced surprisal values for turns spoken by correct and incorrect speakers. All fine-tuned models used speaker identity to predict upcoming words, but may have inserted speaker transitions to make some stimuli more plausible. These findings suggest that while LLMs may learn to represent some common norms, they cannot (yet) replicate human behavior in natural spoken dialogue. 

## Contents

This document has the following sections:

- [Built With](#built-with)
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Acknowledgements](#acknowledgements)
- [Research Notice](#research-notice)

Please find below links to additional/complementary resources for this project:
- [Project OSF](https://osf.io/fxn8y/?view_only=c70f49ab84a149b6b999be606f619eb2)

## Built with

Here are some of the popular frameworks this project uses:

- [Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Weights and Biases](https://wandb.ai/site)
- [Hydra](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710)

## Overview

### Modelling



In this project, we implement two variants of GPT-2: one with explicit speaker identity information and one without. Since we aim to investigate whether LLMs can generalize to natural spoken dialogue, we fine-tuned our models using transcripts of naturalistic conversations from the In Conversation Corpus (ICC). The ICC is a high quality audio corpus recorded in the *Anonymous* lab at *Anonymous* University. Each conversation is approximately 25 minutes long and features a pair 116 of undergraduate students. Participants sat in two sound-proofed rooms separated by a glass window, communicated using a 117 microphone and headset, and were recorded 2 on separate channels for complete sound isolation. In half the conversations, the 118 participants were recruited separately and were strangers. In the other half, they were recruited together and knew each other.


#### GPT2

GPT-2 is a transformer-based LLM pre-trained on a very large corpus of English data in a self-supervised fashion. This means it was pre-trained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences. More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token i only uses the inputs from 1 to i but not the future tokens.This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pre-trained for however, which is generating texts from a prompt.

It is important to realize that GPT-2 has two types of embeddings, word and positional embeddings, which encode a word and its position relative to other words. However, it does not have a notion of individual speakers, which makes it a monologue model. Although this model is well-known for capturing language syntax and semantics, it does not guarantee that the model captures the structure or norms underlying spoken dialogue.

#### TurnGPT

[TurnGPT](https://arxiv.org/abs/2010.10874) is a transformer based language model that introduces a third type of embedding: a speaker identity embedding for every token in a word embedding, which allows it to become sensitive to speaker identity. It was designed
to predict Transition Relevance Places (TRPs) i.e. points in a turn where an interlocutor may start speaking, using pragmatic and syntactic completion points. In this project, we use a modified version of TurnGPT with a LMHead for causal language modeling.

### Experiment and Measures

We used the stimuli and experimental setup from *Anonymous* (2022) to determine whether our fine-tuned models produced similar results as human participants. Their study investigated whether listeners in dialogue use preceding turns in conversation to predict speaker-specific upcoming turns. The results showed that listeners more accurately predicted the ends of turns spoken by the “correct" speaker as compared to the “incorrect" speaker. This suggests that listeners use speaker identity representations to anticipate upcoming turns. We used these experimental stimuli to investigate whether LLMs can accurately emulate human dialogue behavior. In this section, we describe how we leverage methods and measures from *Anonymous* (2022) in the current work.

The stimuli used for prediction belonged to one of six conditions in a two (speaker) by three (congruence) design, depending on the second turn in the two-turn sequence. The second turn differed in the identity of the speaker (same vs different) and the plausibility of the turn by that speaker (congruent, incongruent, and violative). Congruent second turns were relatively plausible, i.e. spoken by the “correct"speaker. Incongruent second turns were not plausible given the preceding turn context. Specifically, they contained the same words as the congruent stimuli, except that they were spoken by the “wrong" speaker, which rendered them implausible.

We extracted the LLM-produced surprisal values for turns spoken by correct and incorrect speakers. For a given sequence, The equation below defines turn surprisal where the first turn has K words denoted $w_1^{1}, w_2^{1} ... w_K^{1}$ and the second turn has N words denoted $w_1^{2}, w_2^{2} ... w_N^{2}$, such that the superscript represents the turn number and the subscript represents the position of a word in that turn. The second turn surprisal is then the sum of the negative log probability for each word in the second turn given all previous words in the second turn and the entire first turn. As the second turn in our stimuli can contain at most two words, N $\in$ \{1, 2\}.

$\textit{Second Turn Surprisal} = \sum_{i=1}^{N} - log P(w_i^{2} | w_1^{2}, ... w_{i-1}^{2}, w_1^{1}, ... w_K^{1})$


## Getting Started

This repository contains code to fine-tune GPT-2 and TurnGPT, as well as produce second-turn surprisal values for stimuli from *Anonymous* (2022). These surprisal values are then used for further anaysis in R (See Project [OSF](https://osf.io/fxn8y/?view_only=c70f49ab84a149b6b999be606f619eb2)). 

To get started, the first step is to clone this repository:

```bash
git clone https://github.com/mumair01/GPT-Monologue-to-Dialogue.git
```

### Structure

In this section, we provide an overview of the structure of this project and the various components.

The cloned repository should have the following structure: 

```txt
gpt_monologue_dialogue
|-- bin/
    |-- *.sh
|-- conf/
    |-- dataset/
    |-- experiment/
    |-- config.yaml
|-- data_lib/
|-- gpt_dialogue/
    |-- gpt2/
    |-- pipelines/
    |-- turngpt/
    |-- *.py
|-- scripts/
|-- tests/
|-- .gitignore
|-- environment.yml
|-- LICENSE
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- setup.py

```

Here is a description of what each directory contains:
| Directory      | Description |
| ----------- | ----------- |
| bin      | Shell scripts tp configure the project environment.     |
| conf   | Hydra configurations for using custom datasets and experiments.        |
| data_lib   | Data preprocessing package.       |
| gpt_dialogue   | Main modeling package containing GPT2 and TurnGPT.          |
| scripts   | Main scripts for finetuning and inference.        |
| tests   | Pytest based testing folder. |


## Environment Setup 

### Python environment

First, set up the [conda](https://docs.conda.io/en/latest/) environment for this project, which installs all the required libraries:

```bash
conda env create -f environment.yml
```

If using conda is not a preferred option, install all dependencies for the currently selected python environment using:
```bash
pip install -r requirements.txt
```

### Project environment

To ensure that this project runs smoothly across devices, a number of environment variables for the project **must** be sourced before any scripts can be run. 

We provide a number of shell scripts in the bin/ directory to streamline this process:

| Script      | Description |
| ----------- | ----------- |
| set_env     | Sets the local environment.     |
| hpc_slurm/set_hpc_env   | Set the project environment for a high performance cluster.        |
| set_configs   | Sets variables that are commonly changed across environments.       |

To run this project locally, set the project environment using the following:

```bash
source set_env.sh
```

**NOTE**: The above file must be sources, even for running the test suite. 

To run this project on a high performance cluster (HPC), use the following. 

**Note**: The paths in this script may need to be changed based on the type of HPC. 
```bash
source bin/set_hpc_env.sh
```

The set_configs.sh script contains variables that are used across scripts in this project, but may commonly need to be changed. For example, the path to the model used for inference may need to be changed across experiments. The aim of this script is to provide a single point of change for the project. 

**IMPORTANT:** The project environment must be sourced every time there are changes made to set_configs.sh.


### Dataset 

We use two datasets as part of this experiment: the In Conversation Corpus (ICC) and stimuli from *Anonymous* (2022). The ICC is sub-divided into two additional datasets: one containing five conversations and one containing 28 conversations. 

Unfortunately, while we do not publicly release the ICC and stimuli due to IRB constrains, we do provide scripts to process the raw data. The bin/generate_datasets.sh script can be used to generate all processed datasets. We hope that we will be able to release our raw data in the future. 

**NOTE**: If you would like access to our data for academic research, please send an email to: muhammad.umair@tufts.edu


## Experiments 

To replicate our experiments, the first step is to fine-tune various configurations of GPT2 and TurnGPT. Next, these fine-tuned models can be used to generate conditional probability values for stimuli from *Anonymous* (2022). Finally, conditional probabilities can be converted into surprisal, which we analyze in our paper. 

Note that we log both finetuning and inference runs using wandb, for which a [wandb account](https://wandb.ai/site) is required. 

### Finetuning 

Transformer-based models show high performance when learning new tasks due to their capacity for transfer learning. Under this paradigm, models are first pre-trained using large datasets on data-rich tasks (e.g, next-word prediction) in an unsupervised fashion. During the pre-training process, the model learns general-purpose domain knowledge. Next, the model is fine-tuned on smaller, task-specific datasets. During the finetuning process, models learn task-specific knowledge on top of the existing general-knowledge gained during pre-training. This process of pre-training and fine-tuning enables a language model to achieve state of the art performance on  a number of language benchmarks. 


Once the fine-tuning and inference datasets have been created, use fine-tune.py scripts directory to fine-tune as follows:

```bash
python finetune.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG>
``` 

Since there are a number of model-dataset fine-tuning configurations we generate as part of our experiments, the bin/local directory contains a number of shell scripts that can automate this process.

### Inference


To generate surprisal from our fine-tuned models, we first want to generate the conditional probability of each word for all stimuli in the dataset from *Anonymous* (2022). This can be done using the scripts/inference.py script as follows:

```bash
python inference.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG>
```

### Configurations 

Both the fine-tuning and inference processes can be configured. We use hydra to ensure that minimum configuration changes are required for all scripts to be run on different systems. Please review all files in the conf/hydra directory (which have detailed comments before running any scripts). 

## Acknowledgements

**NOTE**: Acknowledgements have been removed for this version to ensure the integrity of the double-blind peer review process. 

The following code sources were used as a starting point for our implementation, and we would like to thank the authors for making their projects open-source. 

- [TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- [Transformers](https://huggingface.co/docs/transformers/index)

## Research Notice

The ideas presented in this repository are novel and part of academic research with a research paper forthcoming. Please do not use the novel ideas presented in a research setting without permission from the authors.