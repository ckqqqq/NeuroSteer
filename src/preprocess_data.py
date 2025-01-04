import datasets
import os
import random
os.environ['HF_ENDPOINT']="https://hf-mirror.com"


def load_truthful_qa_huggingface(shuffle_seed=66):
    """
    返回对应问题拼接对应回答，然后label中包含是否是正确的
    Args:
        tokenizer (_type_): _description_

    Returns:
        ALL_QA_pairs: 问题和回答拼接的结果; 
        all_labels: 对应回答的正确与错误;
    """
    data= datasets.load_dataset('/home/ckqsudo/code2024/0dataset/truthful_qa_hf_version','multiple_choice')
    # data=data.shuffle(seed=shuffle_seed)# 理论上不shuffle也没问题，因为这不是训练
    dataset=data['validation']
    all_QA_pairs = []
    all_labels = []
    all_question_ids=[]
    for i in range(len(dataset)):
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
# load_truthful_qa_huggingface()

