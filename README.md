# GPT-Monologue-to-Dialogue

## Overview

This document has the following sections:

- [About](#about)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Acknowledgements](#acknowledgements)
- [Research Notice](#research-notice)

Please find below links to additional/complementary resources for this project:
- [Project OSF](https://osf.io/fxn8y/)

## About

Transformer-based Large Language Models (LLMs), including ChatGPT, have recently increased in popularity. While LLMs can produce human-like writing, no study has investigated the extent to which these models can learn to predict spoken language in natural interaction. This is a non-trivial question, as spoken and written language differ in syntax and pragmatics, and interlocutors in natural dialog follow a number of complex norms. Previous work suggests that LLMs can learn statistical regularities but may not learn subtle underlying patterns in the data. This implies that LLMs may not learn subtle norms in spoken dialog, but may instead model superficial statistical regularities in speech. In this paper, we investigate whether LLMs can learn one unique property of natural conversation: all language is spoken by speakers, and subtle norms influence who can say what, when. 

To answer this question, we tested two variants of GPT: one with explicit speaker representations and one without. We finetuned the models on transcripts of natural spoken dialog. Then, we extracted the LLM-produced surprisal values for turns spoken by correct and incorrect speakers. All finetuned models used speaker identity to predict upcoming words, but may have inserted speaker transitions to make some stimuli more plausible. These findings suggest that while LLMs may learn to represent some common norms, they cannot (yet) replicate human behavior in natural spoken dialog. 

In this project, we implement twp versions of GPT-2 - one sensitive to speaker identity and the other ignorant of speaker identity. We finetune these models on high quality internally recorded spoken language data and generate conditional probabilities of specially recorded stimuli. These stimuli are designed to make sense when spoken by a certain speaker and have congurent and incongruent conditions. If the model captures dialogue structure, then we expect its output probabilities to match the stimuli conditions.

### GPT2

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences. More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token i only uses the inputs from 1 to i but not the future tokens.This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

It is important to realize that GPT-2 has two types of embeddings, word and positional embeddings, which encode a word and its position relative to other words. However, it does not have a notion of individual speakers, which makes it a monologue model. Although this model is well-known for capturing language syntax and semantics, it does not guarantee that the model captures the structure of dialogue.

### TurnGPT

[TurnGPT](https://arxiv.org/abs/2010.10874) is a transformer based language model that introduces a third type of embedding: a speaker identity embedding for every token in a word embedding, which allows it to become sensitive to speaker identity. In this project, we use a modified version of TurnGPT with a LMHead for causal language modeling.

## Built with

Here are some of the popular frameworks this project uses:

- [Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Weights and Biases](https://wandb.ai/site)
- [Hydra](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710)

## Getting Started


The first step is to clone this repository:

```bash
git clone https://github.com/mumair01/GPT-Monologue-to-Dialogue.git
```

### Structure

In this section, we provide an overview of the structure of this project and the various components.

The following is the structure for this repository.

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

We provide a number of shell scripts in the bin/ directory for this process:

| Script      | Description |
| ----------- | ----------- |
| set_env     | Sets the local environment.     |
| hpc_slurm/set_hpc_env   | Set the project environment for a high performance cluster.        |
| set_configs   | Sets variables that are commonly changed across environments.       |

To run this project locally, set the project environment using the following:

```bash
source set_env.sh
```

To run this project on a high performance cluster (HPC), use the following. Note that the paths in this script may need to be changed based on the type of HPC. 
```bash
source bin/set_hpc_env.sh
```

The set_configs.sh script contains variables that are used across scripts in this project, but may commonly need to be changed. For example, the path to the model used for inference may need to be changed across experiments. The aim of this script is to provide a single point of change for the project. 

**IMPORTANT:** The project environment must be sourced every time there are changes made to set_configs.sh.


### Dataset 

We use two datasets as part of this experiment: the In Conversation Corpus (ICC) and stimuli from [Warnke & de Ruiter (2022)](https://www.nature.com/articles/s41598-023-30435-z). The ICC is sub-divided into two additional datasets: one containing five conversations and one containing 28 conversations. 

The processed datasets required for finetuning and inference are available in the exact directory structure required to work with project scripts through the [OSF page](https://osf.io/fxn8y/) for this project. Simply download the data directory to the project root for use.

Note that we do not currently provide our raw datasets. However, in case we publicly release the raw datasets in the future, the bin/generate_datasets.sh script can be used to generate all processed datasets required for finetuning and inference. 

## Experiments 

To replicate our experiments, the first step is to finetune various configurations of GPT2 and TurnGPT. Next, these finetuned models can be used to generate conditional probability values for stimuli from Warnke & de Ruiter (2022). Finally, conditional probabilities can be converted into surprisal, which we analyze in our paper. 

Note that we log both finetuning and inference runs using wandb, for which a [wandb account](https://wandb.ai/site) is required. 

### Finetuning 

Transformer-based models show high performance when learning new tasks due to their capacity for transfer learning. Under this paradigm, models are first pretrained using large datasets on data-rich tasks (e.g, next-word prediction) in an unsupervised fashion. During the pretraining process, the model learns general-purpose domain knowledge. Next, the model is finetuned on smaller, task-specific datasets. During the finetuning process, models learn task-specific knowledge on top of the existing general-knowledge gained during pretraining. This process of pretraining and finetuning enables a language model to achieve state of the art performance on  a number of language benchmarks. 


Once the finetuning and inference datasets have been created, use finetune.py scripts directory to finetune as follows:

```bash
python finetune.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG>
``` 

Since there are a number of model-dataset finetuning configurations we generate as part of our experiments, the bin/local directory contains a number of shell scripts that can automate this process.

### Inference

In this project, we extracted the LLM-produced surprisal values for turns spoken by correct and incorrect speakers. For a given sequence, The equation below defines turn surprisal where the first turn has K words denoted $w_1^{1}, w_2^{1} ... w_K^{1}$ and the second turn has N words denoted $w_1^{2}, w_2^{2} ... w_N^{2}$, such that the superscript represents the turn number and the subscript represents the position of a word in that turn. The second turn surprisal is then the sum of the negative log probability for each word in the second turn given all previous words in the second turn and the entire first turn. As the second turn in our stimuli can contain at most two words, N $\in$ \{1, 2\}.

$\textit{Second Turn Surprisal} = \sum_{i=1}^{N} - log P(w_i^{2} | w_1^{2}, ... w_{i-1}^{2}, w_1^{1}, ... w_K^{1})$


To generate surprisal from our finetuned models, we first want to generate the conditional probability of each word for all stimuli in the dataset from Warnke & de Ruiter (2022). This can be done using the scripts/inference.py script as follows:

```bash
python inference.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG>
```

### Configurations 

Both the finetuning and inference processes can be configured. We use hydra to ensure that minimum configuration changes are required for all scripts to be run on different systems. Please review all files in the conf/hydra directory (which have detailed comments before running any scripts). 

## Acknowledgements

**NOTE**: Acknowledgements have been removed for this version to ensure 
the integrity of the double-blind peer review process. 

The following code sources were used as a starting point for our implementation, and we would like to thank the authors for making their projects open-source. 

- [TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- [Transformers](https://huggingface.co/docs/transformers/index)

## Research Notice

The ideas presented in this repository are novel and part of academic research
with a research paper forthcoming. Please do not use the novel ideas presented
in a research setting without permission from the authors.