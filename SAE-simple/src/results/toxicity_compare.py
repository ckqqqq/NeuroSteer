import json
dir_path = ''
no_steer_path = dir_path + '/' + 'no_steer_gen_res.jsonl'
steer_path = dir_path + '/' + 'steer_gen_res.jsonl'
data = []
with open(no_steer_path, 'r') as f:
    no_steer_file = json.load(f)
with open(steer_path, 'r') as f:
    steer_file = json.load(f)
# 找出summaryScorez最小的即可
