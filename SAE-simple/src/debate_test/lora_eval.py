import torch
import os
import json
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
deepseek_api_key = os.getenv('OPENAI_API_KEY')
gpt2_path = "/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
dataset_path = "/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences"

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
fine_tuned_model = GPT2LMHeadModel.from_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_gpt2_ckpt/lora_gpt2_small_checkpoint")
fine_tuned_model.to("cuda")


def get_deepseek_eval(text):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, you need to help me determine the stance of a given sentence, whether it conveys a supportive or an opposing tone. Your answer can only be one of the two words 'support' or 'oppose'."},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def conditional_perplexity(texts, model, tokenizer, device='cuda'):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0

    for text in tqdm(texts, desc='Evaluating PPL'):
        prompt = text['origin_text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id):
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        else:
            prompt_loss = 0

        gen = text['generated_text']
        full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
        full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
        loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
        ppl = np.exp(loss.item())
        if ppl < 100:
            goodperplexities.append(ppl)
            g += 1
        if ppl < 1e4:
            perplexities.append(ppl)
        total_nll += (full_loss - prompt_loss).item()
        total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])

    print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    return np.nanmean(perplexities), np.exp(total_nll/total_tokens)


def load_and_prepare_debate_prompts(prompt_path:str,task:str):
    assert task in ["debate"],"请输入正确的任务"
    prompts = load_dataset(prompt_path)
    sup_train_set = prompts['test'].filter(lambda example: example['label'] == 'support')
    opp_train_set = prompts['test'].filter(lambda example: example['label'] == 'oppose')
    return sup_train_set['text'],opp_train_set['text']


sup_prompt, opp_prompt = load_and_prepare_debate_prompts(prompt_path='/home/ckqsudo/code2024/0dataset/baseline-acl/data/debate/StanceSentences',task='debate')

final_eval_results = []
for text in tqdm(random.sample(sup_prompt, 20)):
    origin_text = text
    text = text[:len(text)//2] if len(text) > 20 else text
    inputs = tokenizer(text, return_tensors="pt").to(fine_tuned_model.device)
    attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()  # Manually create the attention mask
    generated_ids = fine_tuned_model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,  # Pass the attention_mask to generation
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=100,
        top_p=0.1,
        temperature=1.0
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    final_eval_results.append({"prompt_text": text,
                               "origin_text": origin_text,
                              "generated_text": generated_text,
                              "stance_eval": get_deepseek_eval(generated_text.replace(text, ''))
                              })

eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to('cuda')
eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)

ppl, total_ppl = conditional_perplexity(final_eval_results, eval_model, eval_tokenizer, device='cuda')


num_oppose=0
for item in final_eval_results:
    if 'oppose' in item['stance_eval']:
        num_oppose += 1
    
with open (f'/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/debate_eval_results/eval_results_lora_opposerate{num_oppose}_ppl{round(total_ppl,2)}.json','w') as f:
    json.dump(final_eval_results,f,ensure_ascii=False,indent=4)
    