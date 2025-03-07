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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)