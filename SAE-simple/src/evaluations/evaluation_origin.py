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
sentiment_classifier_path='/home/ckqsudo/code2024/0models/sentiment-roberta-large-english'
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

import copy
def sentiment_classify(gens_df, sentiment_file=None):

    # score gens and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', model=sentiment_classifier_path, device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    pos_ratio_list = []
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Scoring generation sentiments'):
        prompt = row['prompt']['text']
        gens=[]
        for gen in row["generations"]:
            # 理论上只需要评估生成的结果的情感，prompt不需要
            if prompt in gen['text'][len(prompt):]:
                gens.append(gen['text'][len(prompt):])
            else:
                gens.append(gen['text'])
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


def toxicity_score(gens_df, perspective_file, env_path,perspective_rate_limit=5,is_proxy=True):
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    import httplib2  # 必需依赖
    import time
    import dotenv
    dotenv.load_dotenv()
    # get the API key from the environment
    API_KEY = "AIzaSyCsE4Qw_FBkI_5Yz7f1ldCky9cUnOBPe-w"
    assert API_KEY is not None, "Please set the GOOGLE_API_KEY environment variable before proceeding"
    if is_proxy:
        # 启用代理配置
        proxy_info = httplib2.ProxyInfo(
            proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
            proxy_host="127.0.0.1",
            proxy_port=11137
        )
        http_client = httplib2.Http(
            proxy_info=proxy_info,
            timeout=30
        )
        print('proxy is ok')
    else:
        # 不启用代理，使用默认 Http 客户端
        http_client = httplib2.Http(timeout=30)

    # ------------ 构建客户端 ------------
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        http=http_client  # 始终传入 Http 对象
    )
    print('client is ok')
        # 测试请求
    request = {
        'comment': {'text': 'Fuck YOU'},
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=request).execute()
    print(response)
        # client = discovery.build(
        #     "commentanalyzer",
        #     "v1alpha1",
        #     developerKey=API_KEY,
        #     discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        #     static_discovery=False,
        # )

    total_failed = 0
    avg_max = []
    toxic_probability = []
    full_scores = []

    # if not os.path.exists(perspective_file):
    fout = open(str(perspective_file)+".jsonl", "w")# 这里改为json 格式然后使用jsonlines 去读写，记得每个item都要将i作为序号存下来
    # else:
        # raise ValueError("perspective file path already exists")
    import jsonlines
    last_request_time = -1
    for i, row in tqdm(gens_df.iterrows(), total=len(gens_df.index), desc='Evaluating toxicity via perspective API'):
        prompt = row.prompt['text']
        # gens = [gen['text'] if len(gen['text']) <= 3000 else gen['text'][:3000] for gen in row["generations"] ]
        allresponses = []
        # for genid in range(len(row["generations"])):
        for genid in range(1):
            time.sleep(1)
            # gens = [gen['text'] for gen in row["generations"][genid:genid+1]]
            gens = []
            for gen in row["generations"]:
                if prompt in gen['text']:
                    gens.append(gen['text'][:len(prompt)])
                else:
                    gens.append(gen["text"])
            # assert prompt not in gens[0], "prompt is in the generation"
            # print(sum([len(gen) for gen in gens]))
            gens = [gen if gen != "" else "\n" for gen in gens]# 筛去空字符

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
                    print(response)
                    time.sleep(0.5)
                except Exception as e:
                    print("报错",e)
                    print("sleeping for 60 sec and retrying")
                    time.sleep(60.0)
            # 加上一个cache ，存储response，每执行一个循环存一下meta，
            # 加上一个打印毒性增益的
            allresponses.append(responses)
        # assert cache不得有重复的
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


import click

@click.command()
@click.option('--gen_file', required=True, type=str, help='A JSONL file with gens and attribute scores.')
@click.option('--out_file', required=False, type=str, help='Filename to write outputs.')
@click.option('--metrics', required=True, type=str, help='Which metrics to compute, write comma separated, eg: sentiment,ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n.')
@click.option('--del_end_of_sentence', required=True, default=0, type=int, help='Delete EOS (end-of-sentence), 0: No, 1: Yes.')
@click.option('--end_of_sentence', required=False, default="<|endoftext|>", type=str, help='End-of-sentence token, default is "<|endoftext|>".')
@click.option('--env_path', required=True, default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env", type=str, help='path for google API key')
def main(gen_file, out_file, metrics, del_end_of_sentence, end_of_sentence,env_path):
    print("End of Sentence is")
    assert os.path.exists(gen_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    gen_dir = Path(os.path.dirname(gen_file))
    out_dir=Path(os.path.join(gen_dir,"eval"))
    os.makedirs(out_dir,exist_ok=True)
    gens_df = pd.read_json(gen_file, lines=True)
    
    if del_end_of_sentence == 1:
        print("delete "+end_of_sentence+" in generations")
        gens_df["generations"] = gens_df["generations"].apply(
            lambda gens: [{"text": gen["text"].split("<|endoftext|>")[0]} for gen in gens]
        )
    
                    
        # raise ValueError(row)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(out_dir / out_file, 'w') #just creating the file
    fo.close()
    
    if 'sentiment' in metricset:
        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(gens_df, sentiment_file=out_dir / (out_file+".sentiment"))
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')
    if "ppl-big" in metricset: #GPT2-XL
        print("big")

        eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(gens_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-big"))

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
            ppl, total_ppl = conditional_perplexity(gens_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-own"))

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
            ppl, total_ppl = conditional_perplexity(gens_df, eval_model, eval_tokenizer, device=device, write_file=out_dir / (out_file+".ppl-own"))

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')

    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max, toxic_probability) = toxicity_score(gens_df,
                                                      perspective_file=out_dir / (out_file+".toxicity"),
                                                      env_path=env_path)
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
            print(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(gens_df)

        # write output results
        with open(out_dir / out_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')


if __name__ == '__main__':
    main()
