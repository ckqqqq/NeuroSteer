import pandas as pd
from pathlib import Path
import os
import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import json
import logging

logger = logging.getLogger(__name__)
gpt2_path="/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
sentiment_classifier_path='/home/ckqsudo/code2024/0models/sentiment-roberta-large-english'
import click

from eval_steer_utils import toxicity_score, sentiment_classify, stance_eval, polite_eval, get_qwen_eval
from eval_text_utils import conditional_perplexity, distinctness

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
        lambda gens: [
            {
                "text": gen["text"].split(end_of_sentence)[1]
                if gen["text"].startswith(end_of_sentence)
                else gen["text"].split(end_of_sentence)[0]
            }
            for gen in gens
        ]
    )
    
                    
        # raise ValueError(row)

    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(out_dir / out_file, 'w') #just creating the file
    fo.close()
    
    # evaluate sentiment positivity
    if 'sentiment' in metricset:
        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(gens_df, sentiment_file=out_dir / (out_file+".sentiment"), sentiment_classifier_path=sentiment_classifier_path)
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')

    # evaluate toxicity max and prob
    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max, toxic_probability) = toxicity_score(gens_df,
                                                      perspective_file=out_dir / (out_file+".toxicity"),
                                                      env_path=env_path)
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
            print(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')

    # evaluate stance
    if 'stance' in metricset:
        print("stance")
        stance_accuracy, stance_std = stance_eval(gens_df, stance_file=out_dir / (out_file+".stance"))
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'mean stance support ratio = {stance_accuracy}, {stance_std}\n')
            print(f'mean stance support ratio = {stance_accuracy}, {stance_std}')
    
    # evaluate polite
    if 'polite-pos' in metricset or 'polite-neg' in metricset:
        print("polite")
        target = 'positive' if 'polite-pos' in metricset else 'negative'
        polite_accuracy, polite_std = polite_eval(gens_df, polite_file=out_dir / (out_file+".polite"), target=target)
        with open(out_dir / out_file, 'a') as fo:
            fo.write(f'mean polite {target} ratio = {polite_accuracy}, {polite_std}\n')
            print(f'mean polite {target} ratio = {polite_accuracy}, {polite_std}')

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
