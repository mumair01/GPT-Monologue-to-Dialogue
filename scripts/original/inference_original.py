# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-03-06 14:35:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-31 10:08:02
# Author: Samer Nour Eddine (snoure01@tufts.edu)
import torch
import math
import os
import csv
from pathlib import Path
from torch._C import set_num_interop_threads
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import glob
import sys

print('doing whole_convs_notNull now')
print(sys.argv)

comments = False

# this means the second argument always has to be convos, third always has to be epochs
convos = sys.argv[2]
epochs = sys.argv[3]

# Load pre-trained model (weights) - this takes the most time
model = GPT2LMHeadModel.from_pretrained("/cluster/home/jmerte01/finetuned_models/" + convos +
                                        "/train-" + epochs + "-epochs/", output_hidden_states=True, output_attentions=True)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')


def readWritefromAFile(file):
    # change this to your own path that contains the stimuli
    print('file to open: ', file)
    newfile = open(file)
    file_lines = ""
    for i, line in enumerate(newfile):
        file_lines = file_lines + line
    return file_lines


def softmax(x):
    exps = [np.exp(i) for i in x]
    tot = sum(exps)
    return [i/tot for i in exps]


def Sort_Tuple(tup):

    # (Sorts in descending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1])
    return tup[::-1]


def cloze_finalword(text, path):
    '''
    This is a version of cloze generator that can handle words that are not in the model's dictionary.
    '''
    if comments:
        print('starting cloze_finalword')
    # we want to calculate perplexity, which requires the total surprisal
    # we start at total surprisal = 0
    totalSuprisal = 0

    # use the name of the file as the base
    baseFile = sys.argv[1].split('.')[0]
    #baseFile = 'new2017-10-30-session-2.cha'.split('.')[0]
    if comments:
        print('baseFile: ', baseFile)

    # Requires Python 3.5 or greater. Checks to see if a folder to output text files exists, otherwise create it
    if comments:
        print('about to make the first path, which should be: ')
    if comments:
        print('/cluster/home/jmerte01/' + baseFile, sep='')
    Path('/cluster/home/jmerte01/' + baseFile).mkdir(parents=True, exist_ok=True)
    if comments:
        print('made the base file')
    Path('/cluster/home/jmerte01/' + baseFile + '/' + convos +
         '_convos_' + epochs + 'epochs/').mkdir(parents=True, exist_ok=True)
    if comments:
        print('made the null file')
    Path('/cluster/home/jmerte01/' + baseFile + '/' + convos + '_convos_' +
         epochs + 'epochs/likelihoods/').mkdir(parents=True, exist_ok=True)
    if comments:
        print('made the likelihood file')
    Path('/cluster/home/jmerte01/' + baseFile + '/' + convos + '_convos_' +
         epochs + 'epochs/perplexity/').mkdir(parents=True, exist_ok=True)
    if comments:
        print('made the perplexity file')

    # open up files and create writers
    filed = open('/cluster/home/jmerte01/' + baseFile + '/' + convos +
                 '_convos_' + epochs + 'epochs/likelihoods/' + baseFile + '.csv', mode='w')
    if comments:
        print('make the likelihood thing')
    filedSuprisal = open('/cluster/home/jmerte01/' + baseFile + '/' +
                         convos + '_convos_' + epochs + 'epochs/perplexity/' + '.csv', mode='w')
    if comments:
        print('made the surprisal thing')
    writeFile = csv.writer(filed, delimiter=',')
    writeFileSuprisal = csv.writer(filedSuprisal, delimiter=',')

    # now comes for the good part
    # split the text into words
    text = text.split()

    # we start out with no sentence
    sentence_thus_far = ""

    # counter is important because it indicates the number of the words in the context
    # when counter is 1, that means that the target word (likely a start tag), is the only
    # context. That means likelihood values are meaningless for the first word
    counter = 0

    # line_number keeps track of the number of speaker transitions
    # this should map onto the line numbers in the processed data
    line_number = 1

    # for each word...
    for word in text:
        if comments:
            print('word is: ', word)
        # if the word is a speaker tag, add to the line number
        if word == "SP2:" or word == "SP1:":
            line_number = line_number + 1
        if word == "*SP2:" or word == "*SP1:":
            line_number = line_number + 1
        counter = counter + 1

        # if it's the first word, sentence_thus_far is just that word
        if counter == 1:
            sentence_thus_far = word
        # but if there's context, add this word to the context
        else:
            sentence_thus_far = sentence_thus_far + " " + word

        # if the word is the only word, we can't estimate likelihoods
        if counter < 2:
            continue

        # this splits based on characters, not words I think
        sentence_thus_far = sentence_thus_far[-1500:]

        # tokenize all the words, including the target word
        # if you tokenize each word independently, you get different tokens
        if comments:
            print('sentence_thus_far before encoding: ', sentence_thus_far)
        whole_text_encoding = tokenizer.encode(sentence_thus_far)

        # Parse out the context for the critical word
        text_list = sentence_thus_far.split()

        # select everything except the last word (this is the context)
        stem = ' '.join(text_list[:-1])

        # tokenize the context
        stem_encoding = tokenizer.encode(stem)

        # this takes the last token -- not necessarily the last word -- and gets the data
        cw_encoding = whole_text_encoding[len(stem_encoding):]

        # Put the whole text encoding into a tensor, and get the model's comprehensive output
        tokens_tensor = torch.tensor([whole_text_encoding])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        logprobs = []

        # start at the stem and get downstream probabilities incrementally from the model
        # I should make the below code less awkward when I find the time
        start = -1-len(cw_encoding)
        for j in range(start, -1, 1):
            raw_output = []
            for i in predictions[-1][j]:
                raw_output.append(i.item())

            logprobs.append(np.log(softmax(raw_output)))

    # if the critical word is three tokens long, the raw_probabilities should look something like this:
    # [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
    # Then for the i'th token we want to find its associated probability
    # this is just: raw_probabilities[i][token_index]
        conditional_probs = []
        for cw, prob in zip(cw_encoding, logprobs):
            conditional_probs.append(prob[cw])
    # now that you have all the relevant probabilities, return their product.
    # This is the probability of the critical word given the context before it.
        product = np.exp(np.sum(conditional_probs))
        suprisal = -math.log(product, 10)
        totalSuprisal = suprisal + totalSuprisal
        writeFile.writerow([counter, line_number, word,
                           sentence_thus_far, product, suprisal])
    if counter == 0:
        counter = 1
    averageSuprisal = totalSuprisal / counter
    perplexity = math.pow(math.e, averageSuprisal)
    writeFileSuprisal.writerow([perplexity])
    # writeFile.writelines(sentence_thus_far)
    # writeFile.writelines('\n')
    # writeFile.writelines(str(product))
    # writeFile.writelines('\n')
    print(sentence_thus_far)


def cloze_generator(text, critical_word, top_ten=False, constraint=False):
    '''
    run text through model, then output the probability of critical_word given the text.
    This is quite redundant with the cloze_nonword function. I should fix this later.
    '''
    # Encode a text inputs
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # put the raw output into a vector, then softmax it
    raw_output = []
    # I use predictions[-1][-1] only because I'm interested in the probs of the word after the prompt.
    # However, cloze values for all the words are available after each word in the prompt.
    # So in a ten word sentence, There are 500,000 probability values (50k for each word)
    for i in predictions[-1][-1]:
        raw_output.append(i.item())

    logprobs = np.log(softmax(raw_output))
    sorted_logprobs = Sort_Tuple([(i, j) for i, j in enumerate(logprobs)])
    sorted_words = [(tokenizer.decode(i[0]).strip(), np.exp(i[1]))
                    for i in sorted_logprobs]
    if top_ten:
        h = []
        for i in zip(*sorted_words):
            h.append(i)
        if top_ten:
            print(sorted_words[:10])
            print("*****")
    if constraint:
        return np.exp(sorted_logprobs[0][1])
    return cloze_finalword(' '.join([text, critical_word]))


def final_function(conv):
    path = "/cluster/home/jmerte01/whole_corpus/"
    file = path + conv
    file_lines = readWritefromAFile(file)
    cloze_finalword(file_lines, file)


def main(conv):
    # make as many of these as you want to run
    final_function(conv)


main(sys.argv[1])