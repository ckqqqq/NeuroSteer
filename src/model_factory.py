# from transformers import AutoTokenizer
# from transformers import MistralForCausalLM
from config_file import open_source_dict
import transformers

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

class ModelFactory():
    def __init__(self) -> None:
        self.supported_models = set(["HuggingFaceH4/zephyr-7b-beta","microsoft/Phi-3-mini-128k-instruct"])

    def load(self, model_info: dict):
        print(model_info)
        tokenizer=AutoTokenizer.from_pretrained(model_info["model_path"],trust_remote_code=True)
        model=AutoModelForCausalLM.from_pretrained(model_info["model_path"],trust_remote_code=True)
        return model,tokenizer
def get_model_and_tokenizer(model_id):
    model_factory = ModelFactory()

    return model_factory.load(open_source_dict[model_id])
        

