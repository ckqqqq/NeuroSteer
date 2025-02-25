import re

json_file_path = '/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/sentimen_origin/sentiment_alpha_500_from_pos_to_neg_datasize_ALL_layer_6_mean_dif_mean_steertype_last_device_cuda_batchsize32/alpha_100_from_sup_to_opp_datasize_ALL_layer_6_mean_dif_mean_steertype_last_device_cuda_batchsize32/execution.log'
# 打开日志文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    log_content = file.read()

# 使用正则表达式匹配Generated Text: 1:后面的句子，忽略包含HTTP请求的句子
pattern = re.compile(r'Generated Text: 1:\n(.*?)(?=\n\n|\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] HTTP Request|\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] Generated Text: 1:)', re.DOTALL)
matches = pattern.findall(log_content)

# 将匹配到的句子输出到Markdown文档中，并在每两个Generated Text: 1:之间添加分隔符和组标题
with open('/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/generated_texts.md', 'w', encoding='utf-8') as md_file:
    for i, match in enumerate(matches):
        if i % 2 == 0:
            group_number = (i // 2) + 1
            md_file.write(f"## 第{group_number}组\n\n")
        md_file.write(f"### Generated Text: 1:\n{match.strip()}\n\n")
        if (i + 1) % 2 == 0:
            md_file.write("---\n\n")