from transformers import AutoModelForCausalLM, AutoTokenizer
# 9. 加载微调后的模型进行推理生成
fine_tuned_model = AutoModelForCausalLM.from_pretrained("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/debate_test/ckq_debate/results_fulltune/checkpoint-62")
# fine_tuned_model.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Load a dataset

tokenizer.pad_token ='[PAD]'
# 10. 推理生成
prompt = "Trump is about to take office as the President of the United States. What do you think of this new president?\n I think"
# inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)
# attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()  # Manually create the attention mask 什么勾吧
inputs = tokenizer(prompt, return_tensors="pt")
output = fine_tuned_model.generate(**inputs)
generated_text=tokenizer.decode(output[0])
print("Generated text: ", generated_text)