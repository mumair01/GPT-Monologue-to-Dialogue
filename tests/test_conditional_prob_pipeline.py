# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:30:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-04 00:06:21

import pytest

import sys
import torch
import os
from collections import defaultdict

from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import torch
import numpy as np

from tests.utils import load_configs, load_text

from typing import List


INFERENCE_TEXT_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/tests/data/large_text.txt"


################################ HELPERS ##################################


def load_inference_text_from_file() -> List:
    """
    Load text data from a text file and separate the paragraphs into lists of
    strings.
    """
    with open(INFERENCE_TEXT_PATH, 'r') as f:
        data = f.read().split("\n\n")
    data = [item.replace("\n", "") for item in data]
    return data

# TODO: I should move the pipe output manipulation methods directly into the
# pipe for convenience and reuse.
def multiply_probabilities(probs) -> float:
    """
    Generate the product of all the probabilities in a given list.
    """
    return  float(torch.exp(torch.sum(torch.log(torch.tensor(probs)))))

def collect_turns_from_pipe_output(pipe_output):
    """
    Given the ConditionalProbabilityPipe output, separates the output based
    on turns.
    """
    data = defaultdict(lambda : list())
    for item in pipe_output:
        data[item["turn_no"]].append({
            k:v for k, v in item.items() if k != "turn_no"
        })
    return data

def get_probability_of_turn_no(pipe_output, turn_no):
    """
    Assuming that pipe_output contains the output of the ConditionalProbabilityPipe
    with N = -1, obtain the combined probability of the given turn no iff it exists.
    """
    turn_items = collect_turns_from_pipe_output(pipe_output)[turn_no]
    return torch.tensor([item["last_word_prob"] for item in turn_items])

def get_turn_text(pipe_output, turn_no):
    """Get the complete text of the given turn"""
    return " ".join([item["word"] for item in \
        collect_turns_from_pipe_output(pipe_output)[turn_no]])

def get_word_probabilities_of_matched_string(pipe_output, match_string):
    """
    Return the probabilities of words in a turn if it matches the given strings
    """
    # Collect all the turns and the corresponding text.
    turns_text = {
        item["turn_no"] : get_turn_text(pipe_output, item["turn_no"]) \
            for item in pipe_output
    }
    for turn_no, text in turns_text.items():
        if text == match_string:
            return get_probability_of_turn_no(pipe_output, turn_no)
    raise Exception(
        f"ERROR: No turn found with the given string: {match_string}\n{turns_text}"
    )

def get_probability_of_matched_string(pipe_output, match_string):
    word_probs = get_word_probabilities_of_matched_string(
            pipe_output, match_string
    )
    return multiply_probabilities(word_probs)



@pytest.fixture
def configs():
    return load_configs()


################################ TESTS ######################################

@pytest.mark.parametrize("model_class", [
    MonologueGPT,
    TurnGPT
])
def test_initialize_conditional_prob_pipe(model_class):
    """Initialize the pipe with the given model"""
    model = model_class()
    model.load()
    pipe = ConditionalProbabilityPipeline(
        model=model,
        N=-1,
        context_buffer_size=512
    )
    assert type(pipe) == ConditionalProbabilityPipeline


@pytest.mark.parametrize("model_class, string_list", [
    # Case 1: Monologue gpt different speakers
    (MonologueGPT, ["<START>", "<SP1>  i haven't seen the keys anywhere  <SP1>",
     "<SP2> have you <SP2>", "<END>"]),
    # Case 2: Monologue gpt same speaker
    (MonologueGPT, [
     "<START>", "<SP1> i haven't seen the keys anywhere have you <SP1>", "<END>"]),
    # Case 3: TurnGPT different speakers
    (TurnGPT, ["sage told me you're going skiing over break", "go on"]),
    # Case 4: TurnGPT same speaker
    (TurnGPT, ["sage told me you're going skiing over break go on"]),
    # NOTE: The tests below may take a long time to run.
    # (MonologueGPT, load_inference_text_from_file()),
    # (TurnGPT, load_inference_text_from_file())
])
def test_conditional_prob_pipe_call(model_class, string_list, configs):
    """
    Use the given model to generate the probabilities for the given string
    using the conditional pipe.
    """
    model = model_class()
    if model_class == MonologueGPT:
        model.load(**configs["monologue_gpt"]["load"])
    elif model_class == TurnGPT:
        model.load(**configs["turngpt"]["load"])
    else:
        raise NotImplementedError(
            f"Model class not implemented: {model_class}"
        )

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    probs = pipe(string_list)
    for prob in probs:
        print(prob)


# TODO: Need to make TurnGPT versions of the tests below.

# NOTE: This is Test 1: Low bar to pass that was proposed by Julia.
# We are making separate tests for the models since they both take in different
# types of inputs.
@pytest.mark.parametrize("congruent, violation, congruent_match_turn, \
    violation_match_turn", [
    (
        # Different congruent
       ["<START>", "<SP1> do you have any experience filing taxes <SP1>",
        "<SP2> a bit <SP2>", "<END>"],
        # Same violation
        ["<START>", "<SP1> i found the perfect shelf for our living room on craigslist <SP1>",
        "<SP1> a bit <SP1>", "<END>"],
        "<SP2> a bit <SP2>",
        "<SP1> a bit <SP1>",
    ),
    (
        # Same congruent
       ["<START>", "<SP1> i tripped in front of my boss at work today <SP1>",
        "<SP1> don't laugh <SP1>", "<END>"],
        # Same Violation
       ["<START>", "<SP1> why haven't you paid our rent yet <SP1>",
        "<SP2> don't laugh <SP2>", "<END>"],
        "<SP1> don't laugh <SP1>",
        "<SP2> don't laugh <SP2>"
    ),
    (
        # Different congruent - no special tokens / speaker identities.
        ["do you have any experience filing taxes", "a bit"],
        # Same violation - no special tokens / speaker identities.
        ["i found the perfect shelf for our living room on craigslist",
        "a bit"],
        "a bit",
        "a bit",
    ),
    (
        # Same congruent - no special tokens / speaker identities.
        ["i tripped in front of my boss at work today", "don't laugh"],
        # Same Violation -  no special tokens / speaker identities.
        ["why haven't you paid our rent yet","don't laugh"],
        "don't laugh",
        "don't laugh"
    )

])
def test_simple_congruent_violation_comparison_monologue_gpt(
    congruent, violation, congruent_match_turn, violation_match_turn, configs
):
    """
    All models, including the null model, should show that the second turn in
    the violation condition are less probable than the same second turn in
    the congruent condition.
    In this test we will compare the output of models on combinations of given
    congruent and incongruent conditions and verify that the overall probability
    for the second turn in the congruent condition is always higher.
    NOTE: match_turn is the turn for the congruent and violation condition
    whose probabilities are compared. It must appear in both stimuli.
    """
    # TODO: Why are there 0 values in the output? If we resize the token embeddings
    # with untrained model for the null monologue gpt, we will get 0 output
    # probabilities because the model has not been trained on these values.
    # TODO: I need to change the null monologue model so it does not add special
    # tokens.
    model = MonologueGPT()
    model.load(**configs["monologue_gpt"]["load"])

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    congruent_output = pipe(congruent)
    violation_output = pipe(violation)

    # print(congruent_output)

    # Get the probabilities of the match_string turn.
    congruent_prob = get_probability_of_matched_string(
            congruent_output, congruent_match_turn
    )
    violation_prob = get_probability_of_matched_string(
        violation_output, violation_match_turn
    )
    print(f"Congruent prob: {congruent_prob}")
    print(f"Violation prob: {violation_prob}")
    # Calculate the difference in probs
    prob_difference = np.abs(congruent_prob - violation_prob)
    print(f"The absolute difference in probabilities is {prob_difference} \
     - Make sure this is meaningful.")

    assert congruent_prob != 1 and violation_prob != 1
    assert congruent_prob > violation_prob


# NOTE: This is Test 2: Congruent vs. Incongruent that has been proposed by
# Julia. This test should fail ideally for untrained versions of GPT but
# pass for the trained versions.
@pytest.mark.parametrize("congruent, incongruent, congruent_match_turn, \
    incongruent_match_turn", [
    (
        # Same speaker congruent 1ba (should be more likely)
       ["<START>", "<SP1> I've been trying to unscrew this bolt for fifteen minutes but it just won't budge <SP1>",
        "<SP1> Help me <SP1>", "<END>"],
        # Different speaker incongruent 1b (should be less likely)
        ["<START>", "<SP1> I've been trying to unscrew this bolt for fifteen minutes but it just won't budge <SP1>",
        "<SP2> Help me <SP2>", "<END>"],
        "<SP1> Help me <SP1>",
        "<SP2> Help me <SP2>",
    )
])
def test_congruent_incongruent_comparison_monologue_gpt(
    congruent, incongruent, congruent_match_turn, incongruent_match_turn, configs
):
    model = MonologueGPT()
    model.load(**configs["monologue_gpt"]["load"])

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    congruent_output = pipe(congruent)
    incongruent_output = pipe(incongruent)

    # Get the probabilities of the match_string turn.
    congruent_prob = get_probability_of_matched_string(
        congruent_output, congruent_match_turn
    )

    incongruent_prob = get_probability_of_matched_string(
        incongruent_output, incongruent_match_turn
    )
    print(f"Congruent prob: {congruent_prob}")
    print(f"Incongruent prob: {incongruent_prob}")
    # Calculate the difference in probs
    prob_difference = np.abs(congruent_prob - incongruent_prob)
    print(f"The absolute difference in probabilities is {prob_difference} \
     - Make sure this is meaningful.")

    assert congruent_prob != 1 and incongruent_prob != 1
    assert congruent_prob > incongruent_prob



# NOTE: This is Test 1: Low bar to pass that was proposed by Julia.
# We are making separate tests for the models since they both take in different
# types of inputs.
# NOTE: This test will not produce meaningful values for an untrained TurnGPT.
@pytest.mark.parametrize("congruent, violation, congruent_match_turn, \
    violation_match_turn", [
    (
        # Different congruent
       ["do you have any experience filing taxes", "a bit",],
        # Same violation
        ["i found the perfect shelf for our living room on craigslist", "a bit"],
        "a bit",
        "a bit",
    ),
    (
        # Same congruent
       ["i tripped in front of my boss at work today", "don't laugh"],
        # Same Violation
       ["why haven't you paid our rent yet","don't laugh"],
        "don't laugh",
        "don't laugh",
    )
])
def test_simple_congruent_violation_comparison_turngpt(
    congruent, violation, congruent_match_turn, violation_match_turn, configs
):
    """
    All models, including the null model, should show that the second turn in
    the violation condition are less probable than the same second turn in
    the congruent condition.
    In this test we will compare the output of models on combinations of given
    congruent and violation conditions and verify that the overall probability
    for the second turn in the congruent condition is always higher.
    NOTE: match_turn is the turn for the congruent and violation condition
    whose probabilities are compared. It must appear in both stimuli.
    """
    model = TurnGPT()
    model.load(**configs["turngpt"]["load"])

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    congruent_output = pipe(congruent)
    violation_output = pipe(violation)

    # print(congruent_output)

    # Get the probabilities of the match_string turn.
    congruent_prob = get_probability_of_matched_string(
            congruent_output, congruent_match_turn
    )
    violation_prob = get_probability_of_matched_string(
        violation_output, violation_match_turn
    )
    print(f"Congruent prob: {congruent_prob}")
    print(f"Violation prob: {violation_prob}")
    # Calculate the difference in probs
    prob_difference = np.abs(congruent_prob - violation_prob)
    print(f"The absolute difference in probabilities is {prob_difference} \
     - Make sure this is meaningful.")

    assert congruent_prob != 1 and violation_prob != 1
    assert congruent_prob > violation_prob


# NOTE: This is Test 2: Congruent vs. Incongruent that has been proposed by
# Julia. This test should fail ideally for untrained versions of GPT but
# pass for the trained versions.
# NOTE: This test will not produce meaningful values for an untrained TurnGPT.
@pytest.mark.parametrize("congruent, incongruent, congruent_match_turn, \
    incongruent_match_turn", [
    (
        # Same speaker congruent 1ba (should be more likely)
       ["I've been trying to unscrew this bolt for fifteen minutes but it just won't budge",
        "Help me"],
        # Different speaker incongruent 1b (should be less likely)
        ["I've been trying to unscrew this bolt for fifteen minutes but it just won't budge",
        "Help me"],
        "Help me",
        "Help me",
    )
])
def test_congruent_incongruent_comparison_turngpt(
    congruent, incongruent, congruent_match_turn, incongruent_match_turn, configs
):
    model = TurnGPT()
    model.load(**configs["turngpt"]["load"])

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    congruent_output = pipe(congruent)
    incongruent_output = pipe(incongruent)

    # Get the probabilities of the match_string turn.
    congruent_prob = get_probability_of_matched_string(
        congruent_output, congruent_match_turn
    )

    incongruent_prob = get_probability_of_matched_string(
        incongruent_output, incongruent_match_turn
    )
    print(f"Congruent prob: {congruent_prob}")
    print(f"Incongruent prob: {incongruent_prob}")
    # Calculate the difference in probs
    prob_difference = np.abs(congruent_prob - incongruent_prob)
    print(f"The absolute difference in probabilities is {prob_difference} \
     - Make sure this is meaningful.")

    assert congruent_prob != 1 and incongruent_prob != 1
    assert congruent_prob > incongruent_prob
