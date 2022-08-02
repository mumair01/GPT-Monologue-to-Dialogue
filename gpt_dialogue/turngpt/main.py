# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-28 18:11:34
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-29 10:33:05

from datasets import load_dataset

from gpt_dialogue.turngpt.model import TurnGPT
from gpt_dialogue.turngpt.tokenizer import SpokenDialogTokenizer


# Load tokenizer
tokenizer = SpokenDialogTokenizer("gpt2")
print(tokenizer.decode(tokenizer.encode('someText')))

model = TurnGPT(pretrained_model_name_or_path="gpt2-large")
model.init_tokenizer()
model.initialize_special_embeddings()

context = [
    "Hello there I basically had the worst day of my life",
    "Oh no, what happened?",
    "Do you want the long or the short story?",
]



print(model(tokenizer.encode('someText')))


# batch = model.tokenize_strings(context)
# out = model(**batch, use_cache=True)
# print(out)


# model = TurnGPT(pretrained_model_name_or_path="gpt2-large")

# TOKENIZER_CHECKPOINT = "gpt2"
# TRAIN_PATH = ""
# # Load tokenizer
# tokenizer = SpokenDialogTokenizer(TOKENIZER_CHECKPOINT)

# # Load my dataset
# dataset = load_dataset("csv", data_files={
#         'train' : '/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/datasets/processed/ICC/julia_dissertation/train.csv'
#     })
# # Remove the speaker labels that were already there.
# def clean_speaker_labels(data):
#     if len(data['Utterance'].split()) > 1:
#         data['Utterance'] = " ".join(data['Utterance'].split()[1:-1])
#     return data


# dataset = dataset.map(
#     clean_speaker_labels,
#     batched=False,
#     remove_columns=["Unnamed: 0","convName","convID"]
# )

# def tokenize_fn(tokenizer):
#     return lambda data: tokenizer(data["Utterance"], truncation=True)

# # TODO: Not sure why we remove the column names here - got this from: https://huggingface.co/course/chapter5/3?fw=pt#the-%3Ccode%3Emap()%3C/code%3E-method%E2%80%99s-superpowers
# tokenized_datasets = dataset.map(tokenize_fn(tokenizer), batched=True, batch_size=32,remove_columns=dataset["train"].column_names)

# print(tokenized_datasets)

# print(tokenized_datasets['train'][0])

# model(tokenizer('test'))


# print(dataset['train'])

# The tokenizer expects a list of strings to allow it to treat each separate
# list as an utterance by the next speaker.
# NOTE: The assumption here is that each item in the list is an utterance
# by another speaker and speaker 1 starts.
#train_dataset = [data['Utterance'] for data in list(dataset['train'])]
# # Tokenize this dataset
# tokenized_train_dataset = tokenizer(train_dataset)
# print(tokenized_train_dataset)

# model(tokenized_train_dataset)