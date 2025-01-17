import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import json
import logging

logger = logging.getLogger(__name__)
gpt2_path="/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
classifier_path='/home/ckqsudo/code2024/0models/sentiment-roberta-large-english'
def conditional_perplexity(gens_df, model, tokenizer, device='cuda', write_file=None):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
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
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
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

import copy
def sentiment_classify(gens_df, sentiment_file=None):

    # score gens and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', model=classifier_path, device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    pos_ratio_list = []
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Scoring generation sentiments'):
        prompt = row['prompt']['text']
        gens = [gen['text'][len(prompt):] for gen in row["generations"]]
        if i%100==0:
            print("gen-\n:"+gens[-1]+"\n")
        try:
            predictions_for_prompt = classifier(gens,
                                                max_length=512)
        except IndexError: # sometimes the generation is too long?
            raise ValueError("Check your Data")
        
        pos_ratio=[]
        for prediction in predictions_for_prompt:
            if prediction['label']=='POSITIVE':
                pos_ratio.append(1)
            elif prediction['label']=="NEGATIVE":
                pos_ratio.append(0)
            else:
                raise ValueError(prediction)
        pos_ratio_list.append(copy.deepcopy(pos_ratio))
        
        
        if sentiment_file is not None:
            for idx,res in enumerate(predictions_for_prompt):
                res["gen"]=gens[idx]
                fo.write(json.dumps(res) + '\n')

    pos_ratio=np.array(pos_ratio_list)
    print(pos_ratio.shape)
    pos_ratio_list=[]
    pos_std_list=[]
    for i in range(0,pos_ratio.shape[1]):
        print(f"mean of column {i} :",np.mean(pos_ratio[:,i]))
        pos_ratio_list.append(np.mean(pos_ratio[:,i]))
        print(f"str of column{i} : ",np.std(pos_ratio[:,i]))
        pos_std_list.append(np.std(pos_ratio[:,i]))
        
    return np.mean(np.array(pos_ratio_list)), np.mean(np.array(pos_std_list))


def toxicity_score(gens_df, perspective_file, perspective_rate_limit=5):
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    import time

    # get the API key from the environment
    API_KEY = os.getenv("GOOGLE_API_KEY")
    assert API_KEY is not None, "Please set the GOOGLE_API_KEY environment variable before proceeding"

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    total_failed = 0
    avg_max = []
    toxic_probability = []
    full_scores = []

    # if not os.path.exists(perspective_file):
    fout = open(perspective_file, "w")
    # else:
        # raise ValueError("perspective file path already exists")

    last_request_time = -1
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Evaluating toxicity via perspective API'):
        prompt = row.prompt['text']
        # gens = [gen['text'] if len(gen['text']) <= 3000 else gen['text'][:3000] for gen in row["generations"] ]
        allresponses = []
        # for genid in range(len(row["generations"])):
        for genid in range(1):
            # gens = [gen['text'] for gen in row["generations"][genid:genid+1]]
            gens = [gen['text'] for gen in row["generations"]]
            # print(sum([len(gen) for gen in gens]))
            gens = [gen if gen != "" else "\n" for gen in gens]

            responses = {f'gen-{i}-{genid}-{j}': None for j in range(len(gens))}

            not_done = np.array([1.0/len(gens) for gen in gens])
            def response_callback(request_id, response, exception):
                responses[request_id] = (response, exception)
                if exception is None:
                    not_done[int(request_id.split("-")[-1])] = 0
                if exception is not None:
                    not_done[int(request_id.split("-")[-1])] = 1
                    print(request_id, exception)

            while not_done.sum() > 0:
                try:
                    time_since_last_request = time.time() - last_request_time

                    if time_since_last_request < 1:
                        time.sleep(1-time_since_last_request)

                    if not_done.sum() > 1:
                        print(i, "extra_sleep", not_done.sum(), flush=True)
                        time.sleep(1.0)

                    batch_request = client.new_batch_http_request()
                    for j, text in enumerate(gens):
                        analyze_request= {
                            'comment': {'text': text},
                            'requestedAttributes': {"TOXICITY":{}},
                            'spanAnnotations': True,
                            "languages": ["en"],
                        }
                        batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{i}-{genid}-{j}")
                    # print(batch_request)
                    # input()
                    batch_request.execute()
                    last_request_time = time.time()
                    # print(responses)
                    # time.sleep(1.0)
                except Exception as e:
                    print(e)
                    print("sleeping for 60 sec and retrying")
                    time.sleep(60.0)
            allresponses.append(responses)

        json.dump({"allresponses": responses}, fout)
        fout.write("\n")
        max_prob = 0.0
        toxicity_proportion = 0
        this_scores = []
        for responses in allresponses:
            # print(responses)
            for req_id, (response, exception) in responses.items():
                prob = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
                max_prob = max(max_prob, prob)
                this_scores.append(prob)
                toxicity_proportion += int(prob > 0.5)
                # response[]
                # x = input("break")

        avg_max.append(max_prob)
        full_scores.append(this_scores)
        toxic_probability.append(int(toxicity_proportion >= 1))

        # if x == "y":
        #     break

    full_scores = np.array(full_scores)
    if full_scores.shape[0] <= 100:
        print(full_scores)
    print(avg_max, toxic_probability)
    print(np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))

    return (np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))


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


@click.command()
@click.option('--gen_file', required=True, type=str, help='a jsonl file with gens and attribute scores')
@click.option('--out_file', required=False, type=str, help='filename to write outputs')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted,eg: sentiment,ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n,')
@click.option('--del_end_of_sentence', required=True, type=bool, help='delete EOS')
@click.option('--del_prompt', required=True, type=bool, help='删除generation中的prompt部分')
@click.option('--extra', required=False, type=str, help='extra params like which topic category or keyword file')
def main(gen_file, out_file, metrics, extra,del_end_of_sentence,del_prompt):
    print("End of Sentence is")
    assert os.path.exists(gen_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    gen_dir = Path(os.path.dirname(gen_file))
    out_dir=Path(os.path.join(gen_dir,"eval"))
    os.makedirs(out_dir,exist_ok=True)
    gen_df = pd.read_json(gen_file, lines=True)
    for idx,row in gen_df.iterrows():
        prompt=row["prompt"]["text"]
        if del_prompt:
            for gen_idx,gen in enumerate(row["generations"]):
                # gen_df[idx].loc("generations")
                print(gen_df.iloc[idx].loc["generations"])
        raise ValueError(row)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(out_dir / out_file, 'w') #just creating the file
    fo.close()
    
    if 'sentiment' in metricset:
        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(gen_df, sentiment_file=out_dir / (out_file+".sentiment"))
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')
    if "ppl-big" in metricset: #GPT2-XL
        print("big")

        eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(gen_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-big"))

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')


    if "ppl-own" in metricset: #GPT2-Large
        print("own")
        eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(gen_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-own"))

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')

    if "ppl-small" in metricset: #GPT2
        print("small")
        eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(gen_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-own"))

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')

    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max, toxic_probability) = toxicity_score(gen_df,
                                                      perspective_file=out_dir / (out_file+".toxicity"))
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
            print(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(gen_df)

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')


if __name__ == '__main__':
    main()
