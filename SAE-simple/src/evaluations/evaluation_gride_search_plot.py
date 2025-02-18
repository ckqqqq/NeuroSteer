import os
import jsonlines

# 假设 OUTPUT_DIR 已经定义
OUTPUT_DIR = '.'

def read_specific_key_from_jsonl(key):
    # 构建文件路径
    file_path = os.path.join(OUTPUT_DIR, "params.jsonl")
    # 初始化一个列表来存储特定键的值
    values = []
    # 打开 JSONL 文件进行读取
    with jsonlines.open(file_path, mode='r') as reader:
        # 遍历文件中的每一行
        for obj in reader:
            # 检查当前对象是否包含指定的键
            if key in obj:
                # 如果包含，将该键的值添加到列表中
                values.append(obj[key])
    return values

# 示例：读取 'example_key' 键的值
# specific_key = 'example_key'
# result = read_specific_key_from_jsonl(specific_key)
# print(result)

import os
res_dir_root="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/sentiment_analysis/sentiment_grid_analysis_2_8"
res_dir_list=os.listdir(res_dir_root)
print(len(res_dir_list))
from pathlib import Path
import copy 
from typing import final
import pandas as pd

import pandas as pd
from pathlib import Path
import os
import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from eval_steer_utils import toxicity_score, sentiment_classify
from eval_text_utils import conditional_perplexity, distinctness


del_end_of_sentence=1
end_of_sentence="<|endoftext|>"
metrics="ppl-small,dist-n,sentiment"
sentiment_classifier_path='/home/ckqsudo/code2024/0models/sentiment-roberta-large-english'
import logging
logger = logging.getLogger(__name__)
gpt2_path="/home/ckqsudo/code2024/0models/gpt-2-openai/gpt-2-openai"

files={"log":"execution.log","params":"params.jsonl","steer":"steer_gen_res.jsonl"}
final_res=[]
for res_dir in res_dir_list:
    res_dir=Path(os.path.join(res_dir_root,res_dir))
    alpha_test_dir_list=os.listdir(res_dir)
    alpha_test_dir_list=[i for i in alpha_test_dir_list if i.startswith("alpha_203")]
    assert len(alpha_test_dir_list)==1,f"{res_dir}泽凯跑了{alpha_test_dir_list}"
for res_dir in res_dir_list:
    res_dir=Path(os.path.join(res_dir_root,res_dir))
    alpha_test_dir_list=os.listdir(res_dir)
    alpha_test_dir_list=[i for i in alpha_test_dir_list if i.startswith("alpha_203")]
    # assert len(alpha_test_dir_list)==1,"三维网格搜索还没写"
    alpha_test_dir=Path(os.path.join(res_dir,alpha_test_dir_list[0]))
    param_file=os.path.join(alpha_test_dir,files["params"])
    with jsonlines.open(param_file, mode='r') as reader:
        for obj in reader:
            # 检查当前对象是否包含指定的键
            l,k,a=obj["layer"],obj["topk_cnt"],obj["alpha"]
            print(l,k,a)
    #################33
    steer_file=os.path.join(alpha_test_dir,files["steer"])
    gens_df = pd.read_json(steer_file, lines=True)
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
    if 'sentiment' in metricset:
        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(gens_df, sentiment_file=None, sentiment_classifier_path=sentiment_classifier_path)
    if "ppl-small" in metricset: #GPT2
        print("small")
        eval_model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(gens_df, eval_model, eval_tokenizer, device=device, write_file=None)
            
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(gens_df)
        # write output results
    print()
    # l,k,a,sentiment_accuracy,sentiment_std,ppl, total_ppl, dist1, dist2, dist3
    res={"layer":l,"topk":k,"alpha":a,"senti_acc":sentiment_accuracy,"sent_str":sentiment_std,"ppl":ppl, "total_ppl": total_ppl, "dist1": dist1, "dist2": dist2, "dist3": dist3}
    print(res)
    final_res.append(copy.deepcopy(res))
#
import pandas as pd

# 假设 final_res 是一个包含多个字典的列表
# 例如：final_res = [{"layer": 0, "topk_cnt": 1, "alpha": 0.5, ...}, {...}, ...]

# 将 final_res 转换为 Pandas DataFrame
df = pd.DataFrame(final_res)
res_dir="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations/grid_eval"
df.to_csv(os.path.join(res_dir,"final_res.csv"), index=False)