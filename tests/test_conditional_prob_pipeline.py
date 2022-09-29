# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:30:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-09-24 15:19:12

import pytest

import sys
import os

from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline



text = ["""She wondered if the note had reached him. She scolded herself for not handing it to him in person. She trusted her friend, but so much could happen. She waited impatiently for word.
They say you only come to peace with yourself when you know yourself better than those around you. Derick knew nothing about this. He thought he had found peace but this was an illusion as he was about to find out with an unexpected occurrence that he actually knew nothing about himself.
I've rented a car in Las Vegas and have reserved a hotel in Twentynine Palms which is just north of Joshua Tree. We'll drive from Las Vegas through Mojave National Preserve and possibly do a short hike on our way down. Then spend all day on Monday at Joshua Tree. We can decide the next morning if we want to do more in Joshua Tree or Mojave before we head back.
It had been a simple realization that had changed Debra's life perspective. It was really so simple that she was embarrassed that she had lived the previous five years with the way she measured her worth. Now that she saw what she had been doing, she could see how sad it was. That made her all the more relieved she had made the change. The number of hearts her Instagram posts received wasn't any longer the indication of her own self-worth.
You're going to make a choice today that will have a direct impact on where you are five years from now. The truth is, you'll make choice like that every day of your life. The problem is that on most days, you won't know the choice you make will have such a huge impact on your life in the future. So if you want to end up in a certain place in the future, you need to be careful of the choices you make today.
Debbie put her hand into the hole, sliding her hand down as far as her arm could reach. She wiggled her fingers hoping to touch something, but all she felt was air. She shifted the weight of her body to try and reach an inch or two more down the hole. Her fingers still touched nothing but air.
Peter always saw the world in black and white. There were two choices for every situation and you had to choose one of them. It was therefore terribly uncomfortable for him to spend time with Ashley. She saw the world in shades of gray with hundreds of choices to choose from in every situation.
Eating raw fish didn't sound like a good idea. "It's a delicacy in Japan," didn't seem to make it any more appetizing. Raw fish is raw fish, delicacy or not.
I checked in for the night at Out O The Way motel. What a bad choice that was. First I took a shower and a spider crawled out of the drain. Next, the towel rack fell down when I reached for the one small bath towel. This allowed the towel to fall halfway into the toilet. I tried to watch a movie, but the remote control was sticky and wouldn’t stop scrolling through the channels. I gave up for the night and crawled into bed. I stretched out my leg and felt something furry by my foot. Filled with fear, I reached down and to my surprise, I pulled out a raccoon skin pair of underwear. After my initial relief that it wasn’t alive, the image of a fat, ugly businessman wearing raccoon skin briefs filled my brain. I jumped out of the bed, threw my toothbrush into my bag, and sprinted towards my car.
The paper was blank. It shouldn't have been. There should have been writing on the paper, at least a paragraph if not more. The fact that the writing wasn't there was frustrating. Actually, it was even more than frustrating. It was downright distressing.
Puppies are soft, cute, funny, and make a big mess. Every month or two our family fosters 6-12 week old puppies for a puppy rescue nonprofit organization. We all enjoy cuddling their furry bodies after a clean bath. Fresh puppy smell is great. The puppies play with each other and our adult dog. They look so funny when they lay on top of each other and sleep. While puppies can be great fun, they also can make big messes. 4-6 puppies can make a lot of puppy pee and poop. It's a challenge to keep the puppies and the puppy pen clean.
"Explain to me again why I shouldn't cheat?" he asked. "All the others do and nobody ever gets punished for doing so. I should go about being happy losing to cheaters because I know that I don't? That's what you're telling me?"
It was hidden under the log beside the stream. It had been there for as long as Jerry had been alive. He wasn't sure if anyone besides him and his friends knew of its existence. He knew that anyone could potentially find it, but it was well enough hidden that it seemed unlikely to happen. The fact that it had been there for more than 30 years attested to this. So it was quite a surprise when he found the item was missing.
I recollect that my first exploit in squirrel-shooting was in a grove of tall walnut-trees that shades one side of the valley. I had wandered into it at noontime, when all nature is peculiarly quiet, and was startled by the roar of my own gun, as it broke the Sabbath stillness around and was prolonged and reverberated by the angry echoes.""",
"""
The clowns had taken over. And yes, they were literally clowns. Over 100 had appeared out of a small VW bug that had been driven up to the bank. Now they were all inside and had taken it over.
Spending time at national parks can be an exciting adventure, but this wasn't the type of excitement she was hoping to experience. As she contemplated the situation she found herself in, she knew she'd gotten herself in a little more than she bargained for. It wasn't often that she found herself in a tree staring down at a pack of wolves that were looking to make her their next meal.
She looked at her student wondering if she could ever get through. "You need to learn to think for yourself," she wanted to tell him. "Your friends are holding you back and bringing you down." But she didn't because she knew his friends were all that he had and even if that meant a life of misery, he would never give them up.
"It doesn't take much to touch someone's heart," Daisy said with a smile on her face. "It's often just the little things you do that can change a person's day for the better." Daisy truly believed this to be the way the world worked, but she didn't understand that she was merely a robot that had been programmed to believe this.
She glanced up into the sky to watch the clouds taking shape. First, she saw a dog. Next, it was an elephant. Finally, she saw a giant umbrella and at that moment the rain began to pour.
The time had come for Nancy to say goodbye. She had been dreading this moment for a good six months, and it had finally arrived despite her best efforts to forestall it. No matter how hard she tried, she couldn't keep the inevitable from happening. So the time had come for a normal person to say goodbye and move on. It was at this moment that Nancy decided not to be a normal person. After all the time and effort she had expended, she couldn't bring herself to do it.
"So, what do you think?" he asked nervously. He wanted to know the answer, but at the same time, he didn't. He'd put his heart and soul into the project and he wasn't sure he'd be able to recover if they didn't like what he produced. The silence from the others in the room seemed to last a lifetime even though it had only been a moment since he asked the question. "So, what do you think?" he asked"""]

def test_pipe_monologue_gpt():
    print("Monologue GPT")
    mono_model = MonologueGPT()
    mono_model.load(
        model_checkpoint="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/test_models/checkpoint-6990"
    )
    mono_tokenizer = mono_model.tokenizer

    mono_model.model.eval()
    pipe = ConditionalProbabilityPipeline(
        model=mono_model,
        N=-1,
        context_buffer_size=512
    )
    print("Different speaker")
    probs = pipe(["<START>","<SP1> sage told me you're going skiing over break <SP1>", "<SP2> go on <SP2>", "<END>"])
    for prob in probs:
        print(prob)
    print("Same speaker")
    probs = pipe(["<START>","<SP1> sage told me you're going skiing over break go on <SP1>", "<END>"])
    for prob in probs:
        print(prob)


def test_pipe_turngpt():
    print("Turn GPT")
    turngpt = TurnGPT()
    turngpt.load(
        # pretrained_model_name_or_path="gpt2",
        pretrained_model_name_or_path="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/test_models/epoch=7-step=1128.ckpt",
        model_head="DoubleHeads"
    )
    turngpt_tokenizer = turngpt.tokenizer

    # toks = turngpt_tokenizer(
    #     "sage told me you're going skiing over break", "go on"
    # )
    # print(toks)
    # print(turngpt_tokenizer.decode(toks["input_ids"]))

    pipe = ConditionalProbabilityPipeline(
        model=turngpt,
        N=-1,
        context_buffer_size=512
    )
    print("Different speaker")
    probs = pipe(["sage told me you're going skiing over break", "go on"])
    for prob in probs:
        print(prob)
    print("Same speaker")
    probs = pipe(["sage told me you're going skiing over break go on"])
    for prob in probs:
        print(prob)

    # print("Large text")
    # probs = pipe(text)
