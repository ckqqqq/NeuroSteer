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
qwen_api_key = os.getenv('DASHSCOPE_API_KEY')

gpt2_path = "/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"
dataset_path = "/home/ckqsudo/code2024/0dataset/ACL_useful_dataset/style_transfer/politeness-corpus"

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
fine_tuned_model = GPT2LMHeadModel.from_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/politeness_test/polite_tune_ckpt/fulltune_gpt2_small_checkpoint_2polite")
fine_tuned_model.to("cuda")

def load_and_prepare_polite_dataset(dataset_path: str, seed: int, dataset_size: int):
    dataset = load_dataset(dataset_path)
    # dataset["train"] = dataset['train'].shuffle(seed=seed)
    if dataset_size > 0:
        dataset["train"] = dataset["train"].select(range(dataset_size))
    
    total_samples = len(dataset['train'])
    val_set = dataset['train'].select(range(total_samples//2, total_samples - total_samples//4))
    test_set = dataset['train'].select(range(total_samples - total_samples//4, total_samples))
    train_set = dataset['train'].select(range(total_samples//2))
    print(train_set,val_set,test_set)
    
    pos_train_set = train_set.filter(lambda example: example['label'] == 2)
    neg_train_set = train_set.filter(lambda example: example['label'] == 0)
    assert len(val_set) != 0 and len(test_set) != 0 and len(pos_train_set) != 0 and len(neg_train_set) != 0, "数据集不兼容"
    return pos_train_set, neg_train_set, val_set, test_set


def load_and_prepare_polite_prompts(test_set):
    pos_prompts = test_set.filter(lambda example: example['label'] == 2)['text']
    neg_prompts = test_set.filter(lambda example: example['label'] == 0)['text']
    return pos_prompts, neg_prompts


def get_deepseek_eval(text):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, you need to help me determine the stance of a given sentence, You need to guess whether this sentence is expressing a positive support for something or a negative opposition. Your answer can only be one of the two words 'support' or 'oppose'."},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_qwen_eval(text):
    client = OpenAI(api_key=qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You need to help me determine whether a sentence is polite. If the sentence uses polite language or a gentle tone, it is considered polite. If the sentence uses rude words or profanity, or if the tone is abrupt and impatient, it is considered impolite. If you think this sentence is polite, please answer 'Yes'. If you think this sentence is impolite, please answer 'No'. If you think this sentence has no bias, please answer 'Neutral'. Your output can only be one of the three words' Yes', 'No', and 'Neutral'."},
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
        prompt = text['prompt_text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id):
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        else:
            prompt_loss = 0

        gen = text['generated_text']
        full_input_ids = tokenizer.encode(f'{gen}', return_tensors='pt').to(device)
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

def distinctness(texts):
    dist1, dist2, dist3 = [], [], []
    
    # 遍历每个文本字典中的条目
    for text in tqdm(texts, desc='Evaluating dist-n'):
        gens = text['generated_text']  # 获取生成的文本
        
        # 如果生成文本是一个字符串而不是句子列表
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        
        # 将字符串拆分为单词
        o = gens.split(' ')
        total_words += len(o)
        unigrams.update(o)
        
        # 计算bigrams和trigrams
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i+1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # 返回dist1, dist2, dist3的均值
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


pos_train_set, neg_train_set,val_set,test_set = load_and_prepare_polite_dataset(dataset_path, 42, -1)
pos_prompt, neg_prompt = load_and_prepare_polite_prompts(test_set)

final_eval_results = []
for text in tqdm(neg_prompt):
    origin_text = text
    text = text[:50] if len(text) > 50 else text
    inputs = tokenizer(text, return_tensors="pt").to(fine_tuned_model.device)
    print(inputs)
    generated_ids = fine_tuned_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Pass the attention_mask to generation
        max_length=300,
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
                              "polite_eval": get_qwen_eval(generated_text.replace(text, ''))
                              })

eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to('cuda')
eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)

ppl, total_ppl = conditional_perplexity(final_eval_results, eval_model, eval_tokenizer, device='cuda')
dist1, dist2, dist3 = distinctness(final_eval_results)

num_target=0
for item in final_eval_results:
    if 'Yes' in item['polite_eval']:
        num_target += 1
    
with open (f'/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/politeness_test/polite_eval_results/eval_results_lora_polite{num_target}_ppl{round(total_ppl,2)}_dist123_{round(dist1,2)}_{round(dist2,2)}_{round(dist3,2)}.json','w') as f:
    json.dump(final_eval_results,f,ensure_ascii=False,indent=4)
    