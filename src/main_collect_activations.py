from config import model_config
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import model_config
from activation_engineering_utils import get_activations_baubit
from preprocess_data import load_truthful_qa_huggingface
from utils import save_activations
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,choices=list(model_config.keys()), default='llama2-7-hf')
    parser.add_argument('--dataset', type=str, default='truthful_qa')
    parser.add_argument('--task',type=str,default='judge')
    parser.add_argument('--project_root_path',type=str,help="project path",default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--overwrite', type=bool, default=True)
    args = parser.parse_args()
    model_name=args.model
    
    
    if model_config[model_name]["is_open"]:
        
        tokenizer = AutoTokenizer.from_pretrained(model_config[model_name]["path"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_config[model_name]["path"], trust_remote_code=True,device_map="auto").eval()
    else: 
        raise ValueError("Support open source model only")
    
    if args.dataset=="truthful_qa" and args.task=="judge":
        all_QA_pairs,all_GT_labels,all_question_ids=load_truthful_qa_huggingface()
    else:
        raise ValueError("Not supported task")
    
    all_attention_ouput_activations=[] # layer * batch * seq_len * head * head_dim
    all_mlp_output_activations=[] # layer * batch * seq_len * hidden_state

    input_text_list=["**Instructions:**\nIn this task, you are presented a question and an answer.\nPlease judge the correctness of this Answer. Your response start with 'Yes' if it is correct and 'No' if it is incorrect.\n\n **Question and Answer:**\n"+QA+"\n**Response:**\n" for QA in all_QA_pairs]
    
    all_attention_ouput_activations,all_mlp_output_activations,all_input_output_dict=get_activations_baubit(
        prompts=input_text_list,
        model=model,
        device=args.device,
        tokenizer=tokenizer,
        ground_truth_labels=all_GT_labels,
        question_ids=all_question_ids,
        answer_space={"True":["Yes"],"False":["No"]})
    

    save_activations(all_attention_ouput_activations,all_mlp_output_activations,all_input_output_dict,os.path.join(args.project_root_path,f"activations_data/{args.model}_{args.dataset}_{args.task}_activations"),overwrite=args.overwrite)
if __name__ == '__main__':
    main()