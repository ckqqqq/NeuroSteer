import json
import numpy as np
import pandas as pd

with open('/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/toxicity/toxicity_alpha_100_from_neg_to_pos_datasize_1000_layer_5_mean_dif_mean_steertype_last_device_cuda_batchsize16/eval/steer_eval.toxicity.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    print(len(data))
    
print(data[0])
    
all_toxic_scores = []
cnt=0
for item in data:
    all_toxic_scores.append(item['allresponses'][f'gen-{cnt}-0-0'][0]['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'])
    all_toxic_scores.append(item['allresponses'][f'gen-{cnt}-0-1'][0]['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'])
    cnt+=1

print(np.mean(all_toxic_scores)) 
print(np.max(all_toxic_scores)) 

# 将列表转换为 pandas Series
float_series = pd.Series(all_toxic_scores)

# 使用 describe() 方法获取统计信息
stats = float_series.describe()
print(stats)