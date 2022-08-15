# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 15:30:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 08:57:11

import sys
import os
from typing import List, Any, Dict

import torch
import torch.nn as nn

from tqdm import tqdm

from copy import deepcopy
from dataclasses import dataclass

@dataclass
class Param:
    value : Any = None
    required : bool = True


class ConditionalProbabilityPipeline:
    """
    Pipeline that loads a causal language Model with a __call__ method and
    tokenizer and uses it to generate conditional probabilities.
    """

    _PREPROCESS_PARAMS = {
    }

    _FORWARD_PARAMS = {
        "model" : Param(required=True),
        "N" : Param(required=True),
        "context_buffer_size" : Param(required=True)
    }

    _POSTPROCESS_PARAMS = {}


    def __init__(self, **kwargs):
        """
        Args:
            model : (required)
            N (int) : The number of last words in every turn whose probability
                has to be generated. Set to - 1 to generate the probabilities
                of all the words in a turn.
                Ex: If N = 1, generates probability of the last word only.
            context_buffer_size (int): The total number of words that should
                be kept as context when predicting the probability of a word.
                Note that this should be less than the total number of tokens
                that can be processed by the model.
        """
        self._sanitize_parameters(**kwargs)

    def _sanitize_parameters(self, **kwargs):
        """Clean up the input kwargs and assign to the appropriate map."""
        for param_dict in (self._PREPROCESS_PARAMS,
                self._FORWARD_PARAMS,self._POSTPROCESS_PARAMS):
            for k,v in param_dict.items():
                # Required args must be present.
                if v.required and not k in kwargs:
                    raise Exception(
                        f"ERROR: Value not specified for required parameter {k}"
                    )
                if k in kwargs:
                    v.value = kwargs[k]


    def __call__(self, utterances : List[str]):
        """Processes a list of strings containing utterances"""
        return self.postprocess(self._forward(self.preprocess(utterances)))

    def preprocess(self, utterances : List[str]):
        """Prepare inputs given to __call__ for _forward"""
        return {"utterances" : utterances}

    def _forward(self, preprocess_output : Dict) :
        """Main execution logic"""

        utterances = preprocess_output["utterances"]

        # Get the required param values
        N = self._FORWARD_PARAMS["N"].value
        model = self._FORWARD_PARAMS["model"].value
        context_buffer_size = self._FORWARD_PARAMS["context_buffer_size"].value

        pbar = tqdm(total=len(utterances))
        text = ""
        results_list = []
        for turn_no, turn in enumerate(utterances):
            turn = turn.strip()
            text += " " + turn
            text = text.strip()
            turn_length = len(turn.split(' '))
            n_probs = turn_length if N == -1 or N > turn_length else N
            results = self._get_final_n_words_probs(
                model = model,
                tokenizer = model.tokenizer,
                text=text,
                N = n_probs,
                context_buffer_size=context_buffer_size
            )
            for result_no, result in enumerate(results):
                word_no = turn_length - n_probs + result_no
                result.update({
                    "turn_no" : turn_no,
                    "word_no" : word_no,
                })
                results_list.append(deepcopy(result))
            pbar.update()
        return results_list

    def postprocess(self, forward_outputs):
        """Prepare _forward outputs to be returned. """
        return forward_outputs

    def _get_final_n_words_probs(
        self,
        model,
        tokenizer,
        text : str,
        N : int,
        context_buffer_size : int
    ):
        """
        Get the conditional probability of the N final words in the given text.
        This builds the text from the bottom-up i.e., from word len(text) - N
        to word at len(text).
        """
        words = text.strip().split()
        if N == -1:
            N = len(words)
        assert not (N > len(words) or N<= 0)
        words[:len(words) - N]
        sentence_so_far = " ".join(words[:len(words) - N])
        results = []
        for word in words[len(words) - N:]:
            sentence_so_far += " " + word.strip()
            # Reset the buffer if required
            num_words_so_far = len(sentence_so_far.split(' '))
            if num_words_so_far > context_buffer_size:
                sentence_so_far = " ".join(
                    sentence_so_far.split(' ')[num_words_so_far - context_buffer_size - 1:])
            last_word_prob = self._get_last_word_prob(
                model, tokenizer, sentence_so_far)
            context = " ".join(sentence_so_far.split(' ')[:-1])
            results.append(deepcopy({
                "context" : context,
                "word" : word,
                "last_word_prob" : last_word_prob
            }))
        return results

    def _get_last_word_prob(self, model, tokenizer, text : str):
        """Get the probability of the last word only in the given text."""
        sentence_so_far = text
        context = ' '.join(text.split()[:-1])
        # Encode
        context_encoding = tokenizer.encode(
            context, return_tensors="pt")
        whole_text_encoding = tokenizer.encode(
            sentence_so_far, return_tensors="pt")
        cw_encoding = whole_text_encoding[:, context_encoding.shape[1]:]
        # move to the appropriate device before inference
        # TODO: This was giving an issue wit TurnGPT.
        # whole_text_encoding = whole_text_encoding.to(TORCH_DEVICE)
        output = model(whole_text_encoding)
        # Obtain the logits for the last hidden state and the logits
        # that provide values for the tokens in the critical word.
        # i.e., if cw token starts at position i in the sentence, then the logits
        # are from i-1 to len(tokens) - 1.
        cw_extracted_logits = output.logits[-1, context_encoding.shape[1]-1:-1, :]
        # Obtain the probabilities from the logits
        softmax = torch.nn.Softmax(dim=-1)
        cw_extracted_probs_from_logits = softmax(cw_extracted_logits)
        # NOTE: Converting to log scale and taking exponential sum of the log
        # probabilities at the end will ensure that there is not floating point
        # overflow issue for very small probability values.
        cw_extracted_log_probs_from_logits = torch.log(
            cw_extracted_probs_from_logits)
        # Extract the probabilities of the specific tokens
        cw_tokens_probs = []
        for cw_subtoken, probs in zip(cw_encoding[0], cw_extracted_log_probs_from_logits):
            cw_tokens_probs.append(probs[cw_subtoken])
        return float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))




# # ----------- NOTE: Below if the original code - keeping here for reference.


# import sys
# import os

# import yaml
# from tqdm import tqdm
# from typing import Union, List

# import pandas as pd
# import numpy as np
# import torch
# # Scikit-Learn ≥0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"


# CUDA_ENV = torch.cuda.is_available()
# TORCH_DEVICE = torch.device('cuda') if CUDA_ENV else torch.device('cpu')







# # ------------------------ DATASET HELPERS  -----------------------------

# # NOTE: Assuming that the dataset is in the correct format.
# def load_inference_dataset(csv_path, start_conv_no, end_conv_no):
#     df = pd.read_csv(csv_path,index_col=0)
#     conversation_dfs = [df.loc[df['convID'] == i] for i in range(
#         np.max(df['convID'].unique()) + 1)]
#     if end_conv_no > len(conversation_dfs) or end_conv_no == -1:
#         end_conv_no = len(conversation_dfs)
#     assert len(conversation_dfs) >= end_conv_no
#     assert start_conv_no < end_conv_no
#     conversation_dfs = conversation_dfs[start_conv_no:end_conv_no]
#     return conversation_dfs

# # ------------------------ INFERENCE HELPERS  -----------------------------


# def get_last_word_prob(model, tokenizer, text):
#     sentence_so_far = text
#     context = ' '.join(text.split()[:-1])
#     # Encode
#     context_encoding = tokenizer.encode(
#         context, return_tensors="pt")
#     whole_text_encoding = tokenizer.encode(
#         sentence_so_far, return_tensors="pt")
#     cw_encoding = whole_text_encoding[:, context_encoding.shape[1]:]
#     # move to the appropriate device before inference
#     # TODO: This was giving an issue wit TurnGPT.
#     # whole_text_encoding = whole_text_encoding.to(TORCH_DEVICE)
#     output = model(whole_text_encoding)
#     # Obtain the logits for the last hidden state and the logits
#     # that provide values for the tokens in the critical word.
#     # i.e., if cw token starts at position i in the sentence, then the logits
#     # are from i-1 to len(tokens) - 1.
#     cw_extracted_logits = output.logits[-1, context_encoding.shape[1]-1:-1, :]
#     # Obtain the probabilities from the logits
#     softmax = torch.nn.Softmax(dim=-1)
#     cw_extracted_probs_from_logits = softmax(cw_extracted_logits)
#     # NOTE: Converting to log scale and taking exponential sum of the log
#     # probabilities at the end will ensure that there is not floating point
#     # overflow issue for very small probability values.
#     cw_extracted_log_probs_from_logits = torch.log(
#         cw_extracted_probs_from_logits)
#     # Extract the probabilities of the specific tokens
#     cw_tokens_probs = []
#     for cw_subtoken, probs in zip(cw_encoding[0], cw_extracted_log_probs_from_logits):
#         cw_tokens_probs.append(probs[cw_subtoken])
#     return float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))

# def get_final_n_word_probs(model, tokenizer, text,
#         N, context_buffer_size):
#     words = text.strip().split(' ')
#     if N == -1:
#         N = len(words)
#     assert not (N > len(words) or N<= 0)
#     words[:len(words) - N]
#     sentence_so_far = " ".join(words[:len(words) - N])
#     results = []
#     for word in words[len(words) - N:]:
#         sentence_so_far += " " + word.strip()
#         # Reset the buffer if required
#         num_words_so_far = len(sentence_so_far.split(' '))
#         if num_words_so_far > context_buffer_size:
#             sentence_so_far = " ".join(
#                 sentence_so_far.split(' ')[num_words_so_far - context_buffer_size - 1:])
#         last_word_prob = get_last_word_prob(
#             model, tokenizer, sentence_so_far)
#         context = " ".join(sentence_so_far.split(' ')[:-1])
#         results.append((context,word,last_word_prob))
#     return np.asarray(results)


# def generate_conditional_probs(model, tokenizer, conversation_df,
#         N, context_buffer_size, conv_no):
#     results_list = []
#     text = ""
#     pbar = tqdm(desc="Processing conversation {}".format(conv_no),
#                     total=len(conversation_df))
#     for turn_no, turn in enumerate(conversation_df.itertuples()):
#         text += " " +  turn.Utterance.strip()
#         text = text.strip()
#         turn_length = len(turn.Utterance.split(' '))
#         n_probs = turn_length if N == -1 or N > turn_length else N
#         results = get_final_n_word_probs(
#             model, tokenizer,text,n_probs, context_buffer_size)
#         for result_no, result in enumerate(results):
#             word_no = turn_length - n_probs + result_no
#             context, word, prob = result
#             results_list.append((
#                 turn.convName,turn.convID, turn_no, word_no, context,word,prob))
#         pbar.update()
#     return results_list

# def process_conversation_dfs(model, tokenizer, conversation_dfs,
#     N, context_buffer_size, save_dir):
#     data = []
#     df_columns = [
#             'conversationName','conversationNumber', 'turnNumber','wordNumber','context',
#             'word', 'probability']
#     for i, conversation_df in enumerate(conversation_dfs):
#         results = generate_conditional_probs(
#             model=model,
#             tokenizer=tokenizer,
#             conversation_df=conversation_df,
#             N=N,
#             context_buffer_size=context_buffer_size,
#             conv_no=i)
#         # Load the data as a single dataframe and save (important if the
#         # program crashes).
#         save_path = os.path.join(save_dir,
#                 "{}_conditional_probs.csv".format(results[0][0]))
#         pd.DataFrame(results,columns=df_columns).to_csv(save_path)
#         data.extend(results)
#     # Save the results as a dataframe
#     results_df = pd.DataFrame(data, columns=df_columns)
#     results_df.to_csv(
#         os.path.join(save_dir,"conditional_probs_combined.csv"))
