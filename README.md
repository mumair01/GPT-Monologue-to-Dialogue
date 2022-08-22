# GPT-Monologue-to-Dialogue

## Content

- [About](#about)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Models](#models)
- [Acknowledgements](#acknowledgements)

## About

Over the past few years, Generate Pre-trained Transformer (GPT) has made the news for producing eerily human-like written language. One blog post was the top Hacker News blog post before the author announced that it was written by GPT-3
(https://adolos.substack.com/p/feeling-unproductive-maybe-you-should). These successes have lead many to voice fears that
GPT-3 will be used to mass-produce misinformation. A recent study found that readers were unable to distinguish between
human-written news articles and those written by GPT-21. In fact, readers found some GPT-2 articles more credible than
human-written articles.

All of this is to say GPT and other language models have made huge progress in modeling and simulating written language.
However, there has been relatively little progress in creating language models that can accurately simulate spoken language.
There would be many benefits to doing so. Conversational AI agents with more naturalistic speech will be easier to use, understand and trust. Language models could also help psycholinguists with stimulus norming, the process of ensuring that a stimulus
is predictable or surprising. Language models could even help psycholinguists create stimuli, a process that typically requires
extensive time and creativity. These are only some of the ways that language models capable of replicating human dialog would
be useful. However, there is no evidence that the state-of-the-art language models, currently trained on written language scraped
from websites like Reddit, can quickly learn to use language in dialog.

In this project, we implement twp versions of GPT-2 - one sensitive to speaker identity and the other ignorant of speaker identity. We finetune these models on high quality internally recorded spoken language data and generate conditional probabilities of specially recorded stimuli. These stimuli are designed to make sense when spoken by a certain speaker and have congurent and incongruent conditions. If the model captures dialogue structure, then we expect its output probabilities to match the stimuli conditions.

### Monologue GPT

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

### Structure

In this section, we provide an overview of the structure of this project and the various components.

The following is the structure for this repository.

```txt
gpt_monologue_dialogue
|-- bin/
    |-- *.sh
|-- conf/
    |-- dataset/
    |--env/
    |--experiment/
    |-- config.yaml
|-- data_lib/
    |-- data_lib/
    |-- *.py
|-- gpt_dialogue/
    |-- monologue_gpt/
    |-- turngpt/
    |-- *.py
|-- notebooks/
|-- scripts/
|-- tests/
|-- .gitignore
|-- environment.yml
|-- LICENSE
|-- pyproject.toml
|-- README.md
|-- setup.py

```

Here is a description of what each directory contains:
| Directory      | Description |
| ----------- | ----------- |
| bin      | Contains various shell scripts      |
| conf   | Hydra configurations for using custom datasets, environments, and experiments.        |
| data_lib   | Data preprocessing package       |
| gpt_dialogue   | Main modeling package containing Monologue and Turn GPT          |
| notebooks   | Proof of concept notebooks       |
| scripts   | Scripts for finetuning and inference        |
| tests   | Pytest testing folder        |

### Usage

The first step is to clone this repository using:

```bash
https://github.com/mumair01/GPT-Monologue-to-Dialogue.git
```

Next, use [conda](https://docs.conda.io/en/latest/) to install all required libraries using:

```bash
conda env create -f environment.yml
```

In our experiments, we use two datasets - the In Conversation Corpus (ICC) and Speaker Identity Stimuli - that we do not publicly share. Therefore, to replicate our experiments, use the data_lib scripts icc.py and speaker_identity_stims.py along with conf to set up the expected datasets.

Once the finetuning and inference datasets have been created, use finetune.py and inference.py in the scripts directory to finetune and generate conditional probabilities. Both these scripts can be run as [hydra apps](https://hydra.cc/docs/intro/) as follows:

```bash
python finetune.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG> +env <ENV_CONFIG>

python inference.py +experiment <EXPERIMENT_CONFIG> +dataset <DATASET_CONFIG> +env <ENV_CONFIG>
```

Both these scripts expect a conf folder to be present in the root directory - similar to the structure of this project.

The finetune.py script is responsible for loading the specified model, loading the specified dataset, and finetuning the model on that specific dataset. Similarly, the inference script uses finetuned models to generate the conditional probability of words i.e., P(word | context) using the specified model and dataset. Both scripts use wandb to log results - for which an [account](https://wandb.ai/site) is required.


## Acknowledgements

This project was conducted in the [Tufts Human Interaction Lab](https://sites.tufts.edu/hilab/). It is part of a research paper with the following members.

with the following team members:

- [Muhammad Umair](https://mumair01.github.io/) - Primary developer and second author.
- [Julia Mertens](https://www.linkedin.com/in/juliamertens/) - First author.
- [Lena Warnke](https://lenawarnke.com/About) - Third author.
- [Jan P. de Ruiter](https://engineering.tufts.edu/cs/people/faculty/jp-de-ruiter) - Principal Investigator.

Code sources:

- [TurnGPT](https://github.com/ErikEkstedt/TurnGPT)
- [Transformers](https://huggingface.co/docs/transformers/index)

