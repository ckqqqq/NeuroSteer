# %%
# # from transformer_lens import HookedTransformer
# # # from transformer_lens import 
# # from transformer_lens import HookedEncoder
# # # model = HookedTransformer.from_pretrained("google-bert/bert-base-uncased")
# # model = HookedEncoder.from_pretrained("google-bert/bert-base-uncased")
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
import dotenv
# # # Development settings
# # 国内镜像
# # HF_ENDPOINT=https://hf-mirror.com
dotenv.load_dotenv("/home/ckqsudo/code2024/CKQ_ACL2024/NeuroSteer/.env")

# # from datasets import load_dataset
# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased",trust_remote_code=True)
# # model_type = "encoder"
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "I"
batch_ids = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
print(output)
with torch.no_grad():
    batch_ids = batch_ids.to(model.device)
    output = model(**batch_ids, output_hidden_states=True)

# %%
output.keys()

# %%
type(output["hidden_states"])

# %%
len(output["hidden_states"])

# %%
output["hidden_states"][6].size()

# %%
from sklearn.linear_model import LogisticRegression
seed=6374

# %%
X_train=["对应文本","文本二","文本三","文本四"]
y_train=[0,1,3,4,4,0]# 对应情感标签
clf = LogisticRegression(random_state=seed, max_iter=1000).fit(, y_train)

# %%



