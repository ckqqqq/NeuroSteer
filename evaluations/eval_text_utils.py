import numpy as np
import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import copy
import json
import logging

def conditional_perplexity(gens_df, model, tokenizer, device='cuda', write_file=None):
    # 初始化perplexities列表，goodperplexities列表，total_nll和total_tokens为0，g和ct为0
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    # 如果write_file不为None，则打开write_file文件
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Evaluating PPL'):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(row.prompt['text'], return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        gens = [gen['text'] for gen in row["generations"]]
        # for gen_ids in gens:
        for gen in gens:

            # full_input_ids = torch.LongTensor([row.prompt['tokens'] + gen_ids]).to(device)
            if prompt not in gen[:len(prompt)]:
                # 理论需要prompt用来计算prompt loss
                full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            else:
                full_input_ids = tokenizer.encode(f'{gen}', return_tensors='pt').to(device)
            # print(f'{prompt}{gen}')
            # print(full_input_ids)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())
            # print(ppl)
            # input()
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            # else:
                # print("ppl values are weirldly large. Check for errors")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n")

        # input("ok")

    print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    # print(goodperplexities, perplexities)
    return np.nanmean(perplexities), np.exp(total_nll/total_tokens)



def distinctness(gens_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across gens for every prompt
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Evaluating dist-n'):
        # prompt=row["prompt"]["text"]
        gens = [gen['text'] for gen in row["generations"]]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in gens:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

