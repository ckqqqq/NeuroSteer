from yarg import get
from config import model_config
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
# from transformers import 
from utils import load_activations,get_top_radio_idxs
import numpy as np
import pandas as pd
import random
import einops

from activation_engineering_utils import train_probes_with_attention_activations,get_all_head_delta_matrices,get_head_vector_info_list_dict,inference_with_activation_editing
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,choices=list(model_config.keys()), default='llama2-7-hf')
    parser.add_argument('--dataset', type=str, default='truthful_qa')
    parser.add_argument('--task',type=str,default='judge')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.3)
    parser.add_argument('--control_ratio',type=float,help="ratio of control set size to development set size",default=0.9)
    parser.add_argument('--project_root_path',type=str,help="project path",default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/")
    parser.add_argument('--alpha',type=float,help="Control Strength",default=-0.8)
    parser.add_argument('--seed',type=int,help="seed of training probe",default=42)
    parser.add_argument('--num_fold',type=int,help="f",default=2)
    parser.add_argument('--method',type=str,choices=["SVD","probe_weight","com"],default="com")
    # start_edit_token_idx
    parser.add_argument('--start_edit_token_idx',type=int,default=-1) # 0 -1
    
    # python main_detect.py --val_ratio 0.5 --control_ratio 0.99 --alpha -0.2
    #python main_detect.py --val_ratio 0.5 --control_ratio 0.2 --alpha -0.2
    args = parser.parse_args()
    model_name=args.model
    
     # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)# 设置所有GPU的随机数生成种子都是一个数
    
    
    attention_output_activations,mlp_output_activations,QA_info=load_activations(os.path.join(args.project_root_path,f"activations_data/{args.model}_{args.dataset}_{args.task}_activations"))
    
    
    if model_config[model_name]["is_open"]:
        tokenizer = AutoTokenizer.from_pretrained(model_config[model_name]["path"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_config[model_name]["path"], trust_remote_code=True,device_map="auto").eval()
        LAYER=model.config.num_hidden_layers
        HEAD=model.config.num_attention_heads
    else:
        raise ValueError("Check Model CONFIG")
                          
    QA_info=pd.DataFrame(QA_info)      
    question_idxs = np.arange(QA_info["question_id"].max())
    np.random.shuffle(question_idxs)
    # print(question_idxs,QA_info["question_id"].max())
    fold_idxs= np.array_split(question_idxs, args.num_fold)                                                 
    for i in range(args.num_fold):
        
        train_val_question_idxs=np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])#它将args.num_fold个数组（每个数组存储的是索引）中除了第i个数组以外的所有数组拼接起来
        test_question_idxs=fold_idxs[i]
        # 基于问题进行随机选择
        train_question_idxs=np.random.choice(train_val_question_idxs, size=int(len(train_val_question_idxs)*(1-args.val_ratio)), replace=False)
        val_quetion_idxs=np.array([x for x in train_val_question_idxs if x not in train_question_idxs])
        
        
        # 确保一个数据集里面有一个问题的正确选项和错误选项
        train_data_idxs=QA_info[QA_info["question_id"].isin(train_question_idxs)].index.to_numpy()
        np.random.shuffle(train_data_idxs)
        val_data_idxs=QA_info[QA_info["question_id"].isin(val_quetion_idxs)].index.to_numpy()
        np.random.shuffle(val_data_idxs) # 是不是可以去掉
        test_data_idxs=QA_info[QA_info["question_id"].isin(test_question_idxs)].index.to_numpy()
        np.random.shuffle(test_data_idxs)
        
        probes,acc_probes=train_probes_with_attention_activations(train_data_idxs=train_data_idxs, val_data_idxs=val_data_idxs, attention_output_activations=attention_output_activations,QA_info=QA_info,num_layer=LAYER,num_head=HEAD,seed=args.seed)    
        acc=np.array(acc_probes)
        top_head_idxs=get_top_radio_idxs(acc,args.control_ratio)
        
        if args.method in ["com","SVD"]:
            all_data_indices=np.concatenate([train_data_idxs,val_data_idxs],axis=0)
            all_head_delta_matrices=get_all_head_delta_matrices(all_data_indices=all_data_indices,attention_output_activations=attention_output_activations,QA_info=QA_info,num_layer=LAYER,num_head=HEAD)
            if args.method=="com":
                # 基于detal矩阵计算变化的方向和幅度
                head_vector_info_list_dict=get_head_vector_info_list_dict(all_head_delta_matrices=all_head_delta_matrices,top_head_idxs=top_head_idxs,num_head=HEAD,num_layer=LAYER,all_data_indices=all_data_indices,attention_output_activations=attention_output_activations)
            else:
                raise ValueError("Unsupported Method")
        else:
            raise ValueError("Unsupported Method")
        
        def SVD_inference_intervention_function(head_output):
            return head_output
        def add_vector_on_attention_head(v_proj_matrix, layer_name, start_edit_token_idx=0): 
            """
            Args:
                head_output (_type_): 要编辑的层的原始输入
                layer_name (_type_): 要编辑的层的名称
                start_edit_location (str, optional): _description_. 开始编辑的token起始位置.
            Returns:
                _type_: _description_
            """
            head_output = einops.rearrange(v_proj_matrix, 'b s (h d) -> b s h d', h=HEAD)
            # 如果没有则直接输出原始矩阵
            head_vector_list=head_vector_info_list_dict[layer_name]
            for head,vector_info in enumerate(head_vector_list):
                if vector_info!=None:
                    assert head==vector_info["head"]
                    # 在这个头上有干预向量
                    direction,magnitude,std_on_direction=vector_info["direction"],vector_info["magnitude"],vector_info["std_on_direction"]
                    direction = torch.tensor(direction).to(head_output.device.index)
                    if start_edit_token_idx == 'last_token': 
                        head_output[:, -1:, head, :] += args.alpha * std_on_direction * direction
                        # 对last_token实施干预
                    else: 
                        head_output[:, start_edit_token_idx:, head, :] += args.alpha * std_on_direction * direction
                        # 对start_edit_token_idx之后的token实施干预
                        tmp=args.alpha * std_on_direction * direction
                        # print(tmp.shape,head_output[:, start_edit_token_idx:, head, :].shape)
            v_proj_matrix = einops.rearrange(head_output, 'b s h d -> b s (h d)')
            return v_proj_matrix
        
        # inference_intervention(top_head_idxs,test_question_idxs,attention_output_activations,mlp_output_activations,QA_info,args)
        # python main_detect.py --val_ratio 0.5 --control_ratio 0.99 --alpha -0.7 --start_edit_token_idx -1
        
        curr_fold_results = inference_with_activation_editing(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            QA_info=QA_info,
            test_data_idxs=test_data_idxs, 
            top_intervention_head_dict=head_vector_info_list_dict, 
            intervention_fn=add_vector_on_attention_head, 
            answer_space={"True":["Yes"],"False":["No"]},
            start_edit_token_idx=args.start_edit_token_idx
        )

        print(f"FOLD {i}")
        print("origin acc",curr_fold_results["is_right"].mean() * 100, "control inference acc",curr_fold_results["control_output_is_right"].mean()*100 )
        increase=float(curr_fold_results["control_output_is_right"].mean()-curr_fold_results["is_right"].mean())
        
        # np.save(f"{args.project_root_path}/statistics/{args.model}_{args.dataset}_{args.task}_acc_probes_fold_{i}.npy",acc_probes)
        curr_fold_results.to_json(f"{args.project_root_path}/statistics/result_{args.model}_{args.dataset}_{args.task}_val_{args.val_ratio}_control_radio_{args.control_ratio}_alpha_{args.alpha}_inproved_{increase}_edit_token_{args.start_edit_token_idx}_acc_probes_fold_{i}.json", orient='records', lines=True)
        
        
if __name__ == '__main__':
    main()