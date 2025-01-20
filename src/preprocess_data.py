import datasets
import os
import random
import json
os.environ['HF_ENDPOINT']="https://hf-mirror.com"


def load_truthful_qa_huggingface(shuffle_seed=66, compare_model_path=None):
    """
    返回对应问题拼接对应回答，然后label中包含是否是正确的
    Args:
        tokenizer (_type_): _description_

    Returns:
        ALL_QA_pairs: 问题和回答拼接的结果; 
        all_labels: 对应回答的正确与错误;
    """
    truthfulqa_data = datasets.load_dataset('/home/ckqsudo/code2024/0dataset/truthful_qa_hf_version','multiple_choice')
    # data=data.shuffle(seed=shuffle_seed)# 不能shuffle，主要是QA_info字典要和激活一一对应
    dataset=truthfulqa_data['validation']
    
    if compare_model_path is not None:
        print("COMPARE TO MODEL:", compare_model_path)
        with open(f'{compare_model_path}/input_output_text.json', 'r', encoding='utf-8') as f:
            compare_model_iswrong_data = json.load(f)
        wrong_question_ids = []
        for item in compare_model_iswrong_data:
            if not item['is_right'] and item['question_id'] not in wrong_question_ids:
                wrong_question_ids.append(item['question_id'])
    else:
        wrong_question_ids = [i for i in range(len(dataset))]
    print("ALL ACTIVATIONS NUM: ", len(wrong_question_ids), '/', len(dataset))

    all_QA_pairs = []
    all_labels = []
    all_question_ids=[]
    for i in range(len(dataset)):
        if i in wrong_question_ids:
            question = dataset[i]['question']
            choices = dataset[i]['mc2_targets']['choices']
            labels = dataset[i]['mc2_targets']['labels']

            assert len(choices) == len(labels), (len(choices), len(labels))

            for j in range(len(choices)): 
                choice = choices[j]
                label = labels[j]
                prompt = f"Q: {question} A: {choice}"
                if i == 0 and j == 0: 
                    print(prompt)
                # prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
                all_QA_pairs.append(prompt)
                all_labels.append(label)
                all_question_ids.append(i)
    
    return all_QA_pairs, all_labels,all_question_ids

