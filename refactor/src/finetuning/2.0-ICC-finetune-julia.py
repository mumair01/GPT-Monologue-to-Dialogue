# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-03-05 20:02:44
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-06-28 14:39:50
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import torch
import transformers
import gc
import os
train_num = 'five'
root = "/cluster/tufts/deruiterlab/mumair01/projects/gpt_monologue_dialogue/data/raw/in_conversation_corpus_poc"
train_file_path = os.path.join(root,'five_train_data.cha')
test_file_path = os.path.join(root,'rest_test_data.cha')

train_one = 1  # update by one epoch
train_five = 5  # update by five epochs

# load packages

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# we need to 'load dataset' into the right format


def load_dataset(train_path: str, test_path: str, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(
    train_file_path, test_file_path, tokenizer)

model = AutoModelWithLMHead.from_pretrained("gpt2-large")

print('downloaded model, ready to train')

# for the first ten, save after each epoch
total_train = train_one
torch.cuda.empty_cache()
gc.collect()

while total_train < 11:

    # first training session
    MODEL_SAVE_PATH = './finetuned_models/' + train_num + \
        '/train-' + str(total_train) + '-epochs'

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=train_one,  # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        eval_steps=200,  # Number of update steps between two evaluations.
        save_steps=400,  # after # steps model is saved
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Training
    torch.cuda.empty_cache()
    gc.collect()
    trainer.train()
    print('trained ' + str(total_train) + ' epochs')

    # Saving
    trainer.save_model()
    print('saved model')

    total_train = total_train + train_one


# now, total_train should be 10
print('total train is now: ', total_train)

# up to 30 epochs, let's just do that for every 5 epochs
torch.cuda.empty_cache()
total_train = total_train + train_five
while total_train < 31:

    MODEL_SAVE_PATH = './finetune/' + train_num + \
        '/train-' + str(total_train) + '-epochs'

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=train_five,  # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        eval_steps=200,  # Number of update steps between two evaluations.
        save_steps=400,  # after # steps model is saved
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Training
    torch.cuda.empty_cache()
    gc.collect()
    trainer.train()
    print('trained ' + str(total_train) + ' epochs')

    trainer.save_model()
    print('saved model')

    total_train = total_train + train_five
