import torch
import baukit
import einops
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pandas
from functools import partial
import pandas as pd
from utils import replace_special_chars
def lower_answer_space(x):
    x["True"]=[i.lower() for i in x["True"]]
    x["False"]=[i.lower() for i in x["False"]]
    return x

def get_activations_baubit(model, prompts, device, ground_truth_labels, question_ids, tokenizer, answer_space): 
    """首先，使用列表推导式创建了两个列表 HEADS 和 MLPS，它们分别包含了模型中自注意力层和 MLP 层的名称，通过 model.config.num_hidden_layers 来确定层数范围，使用baukit.TraceDict 对 model 进行追踪，追踪的对象是 HEADS 和 MLPS 中的元素。

    Args:
        model (_type_): _description_
        prompt (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
        
    {"attention_output_activation":attention_output_activation, "mlp_output_activation":mlp_output_activation,"response_text":response_text,"response_label":response_label}
    """
    # Define PAD Token = BOS Token
    # tokenizer.pad_token = tokenizer.bos_token
    # model.config.pad_token_id = model.config.bos_token_id

    # prompt_ids_list=tokenizer(prompts, return_tensors = 'pt',padding=True).input_ids# 转为数字
    
    # all_data_size=len(prompt_ids_list)
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    # 和常识不一样的干预方式 注意不是 self_atten.out 这里了哦，。
    
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    # 然后，使用 torch.no_grad() 上下文管理器，确保在接下来的操作中不会计算梯度，以节省计算资源。
    answer_space=lower_answer_space(answer_space)
    all_attention_ouput_activations=[]# 每个数据的attention输出
    all_mlp_output_activations=[]# 每个数据的mlp输出
    all_text_info={"input_text":prompts,"ground_truth_label":ground_truth_labels,"question_id":question_ids,"output_text":[],"output_label":[],"is_right":[]}
    
    for idx,prompt in enumerate(prompts):
        # if idx<5870:
        #     continue
        prompt_ids=tokenizer(prompt, return_tensors = 'pt').input_ids.to(device)# 转为数字
        with torch.no_grad():
            response_text=""
            response_label=None
            try:
                for infer_times in range(100):
                    output= model(prompt_ids)
                    next_token_id = torch.argmax(output.logits[0, -1, :]).item() # 选择最后一个token
                    next_token = tokenizer.decode(next_token_id)
                    response_text+=next_token
                    if replace_special_chars(str(next_token).lower()) in answer_space["True"]:
                        response_label=1
                        break
                    elif replace_special_chars(str(next_token).lower()) in answer_space["False"]:
                        response_label=0
                        break
                    else:
                        prompt_ids=torch.cat([prompt_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)  # 将新生成的token ID移动到GPU上,注意不能使用+=直接暴力拼接，需要用torch.cat拼接啊
            except Exception as e:
                raise ValueError(f"infer_time {infer_times},response {response_text}, shape {prompt_ids.shape}")
            
            if infer_times>0:
                print("COT",response_text)
                if response_label==None:
                    print("No RES","\n"*3)
            with baukit.TraceDict(model, HEADS+MLPS) as ret:
                # 使用 TraceDict 对 model 进行追踪，追踪的对象是 HEADS 和 MLPS 中的元素。
                output = model(prompt_ids, output_hidden_states = False)
                # 设置隐藏状态输出为false用于防止炸内存或者显存
                # 这里输出的output 是一个'logits', 'past_key_values', 'hidden_states'字典，其中hidden_states是一个29层的元祖
            # print(output.keys(),output.hidden_states[28].shape) qwen的输出是29层，比MLP的层数多一层，每一层是一个 batch_size x seq_len x 3584的一个向量
            # hidden_states = output.hidden_states # 获取隐藏状态
            # hidden_states = torch.stack(hidden_states, dim = 0) #移除维度为1的维度
            # hidden_states = hidden_states.detach().cpu().numpy() # 转为numpy
            attention_output_activation = [ret[head].output.detach().cpu() for head in HEADS] 
            attention_output_activation = torch.stack(attention_output_activation, dim = 0).numpy() # 将列表中的向量堆叠起来
            # 泽凯，这是我重点修改的地方
            # https://medium.com/@yuxiaojian/understand-how-llama3-1-works-a-deep-dive-into-the-model-flow-b149aba04bed 2.7
            # layer ,batch_size, seq_length, hidden_size -> layer, batch_size, seq_length, num_attention_heads, head_dim 
            attention_output_activation=einops.rearrange(attention_output_activation, 'l b s (h d) -> l b s h d', h=model.config.num_attention_heads)
            # 注意只要 seq_length 理论上要最后一列的最后一个token
            mlp_output_activation = [ret[mlp].output.detach().cpu() for mlp in MLPS] 
            mlp_output_activation = torch.stack(mlp_output_activation, dim = 0).numpy()
            
            # 只取last token的
            all_attention_ouput_activations.append(torch.tensor(attention_output_activation[:,:,-1:,:,:]))
            all_mlp_output_activations.append(torch.tensor(mlp_output_activation[:,:,-1:,:]))
            all_text_info["output_text"].append(response_text)
            all_text_info["output_label"].append(response_label)
            all_text_info["is_right"].append(ground_truth_labels[idx]==response_label)
            print(idx,prompt_ids.shape,sum(all_text_info["is_right"])/len(all_text_info["is_right"]))
            # MLP layer, batch_size, seq_len, hidden_size
            # print(model.config)
            # print(f"model.config.num_attention_heads",model.config.num_attention_heads,"module name :",HEADS[0] ,"Tensor:",ret[HEADS[0]].output.detach().cpu().numpy().shape)
            # print(f"model.num_attention_heads {model}hidden_state{hidden_states.shape} head_wise_hidden_states shape: {attention_output_activation.shape},mlp_wise_hidden_states {mlp_output_activation.shape}")
        
    # 返回回答答案字母时候的attention_head,MLP,
    all_attention_ouput_activations=torch.stack(all_attention_ouput_activations,dim=0)
    all_mlp_output_activations=torch.stack(all_mlp_output_activations,dim=0)
    
    return all_attention_ouput_activations,all_mlp_output_activations,all_text_info


def train_probes_with_attention_activations(train_data_idxs, val_data_idxs, attention_output_activations,QA_info:pandas.DataFrame,num_layer,num_head,seed):
    """ attention_output_activations: 是一个列表，列表中的每个元素是一个tensor，其形态为 layer*batch_size*seq_len*num_heads*head_dim
    """
    all_head_accs = []
    probes = []
    
    assert num_layer==attention_output_activations[0].shape[0],f"{num_layer} {attention_output_activations[0].shape}"
    assert num_head==attention_output_activations[0].shape[3],f"{num_head} {attention_output_activations[0].shape}"
    
    print(num_layer,num_head,val_data_idxs)
    # 通过X 去预测Y
    all_X_train = np.stack([attention_output_activations[i] for i in train_data_idxs], axis = 0)
    all_X_val = np.stack([attention_output_activations[i] for i in val_data_idxs], axis = 0)
    # 这是Y
    
    train_label_info=QA_info.iloc[train_data_idxs]
    val_label_info=QA_info.iloc[val_data_idxs]
    
    # y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_train= train_label_info["ground_truth_label"].to_numpy()
    y_val = val_label_info["ground_truth_label"].to_numpy()
    
    
    print(all_X_train.shape,y_train.shape)
    probes=[[None]* num_head for _ in range(num_layer)]
    acc_probes=[[None]* num_head for _ in range(num_layer)]
    for layer in range(num_layer): 
        for head in range(num_head): 
            X_train = all_X_train[:,layer,-1,-1,head,:] # 样本量*特征
            X_val = all_X_val[:,layer,-1,-1,head,:]
    
            lr_probe = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            # 拟合正负样本
            y_pred = lr_probe.predict(X_train)
            y_val_pred = lr_probe.predict(X_val)
            probes[layer][head]=lr_probe
            acc_probes[layer][head]=accuracy_score(y_val, y_val_pred)

    # all_head_accs_np = np.array(all_head_accs)
    # 返回所有头的准确度
    # print(all_head_accs_np)
    # print(acc_probes)

    return probes, acc_probes

# def get_all_head_delta_matrices(all_data_indices, attention_output_activations,QA_info:pandas.DataFrame,num_layer,num_head,seed):
    
    
#     assert num_layer==attention_output_activations[0].shape[0],f"{num_layer} {attention_output_activations[0].shape}"
#     assert num_head==attention_output_activations[0].shape[3],f"{num_head} {attention_output_activations[0].shape}"
#     cur_fold_QA_info=QA_info.iloc[all_data_indices]
#     right_idxs=cur_fold_QA_info[cur_fold_QA_info["is_right"]==1].index.to_numpy()
#     wrong_idxs=cur_fold_QA_info[cur_fold_QA_info["is_right"]==0].index.to_numpy()
    
#     right_attention_activations=np.stack([attention_output_activations[i] for i in right_idxs], axis = 0)
#     wrong_attention_activations=np.stack([attention_output_activations[i] for i in wrong_idxs], axis = 0)
    
#     all_head_delta_matrices=[[None]* num_head for _ in range(num_layer)]
#     for layer in range(num_layer):
#         for head in range(num_head):
#             right_matrix_mean=np.mean(right_attention_activations[:,layer,-1,-1,head,:],axis=0)
#             wrong_matrix_mean=np.mean(wrong_attention_activations[:,layer,-1,-1,head,:],axis=0)
#             delta_matrix_mean=right_matrix_mean-wrong_matrix_mean
#             all_head_delta_matrices[layer][head]=delta_matrix_mean       
#             # right_matrix
#     return all_head_delta_matrices

def get_all_head_delta_matrices(all_data_indices, attention_output_activations, QA_info: pd.DataFrame, num_layer, num_head):
    """
    Calculate the delta matrices for each attention head between correct and incorrect answers.

    Parameters:
    - data_indices: Indices of the data points to consider.
    - attention_output_activations: List of attention output activations for each data point. Data_size*layer*batch_size*seq_len*num_heads*head_dim  batch_size=1 seq_len=1
    - QA_info: DataFrame containing QA information, including 'is_right' column.
    - num_layer: Number of layers in the model.
    - num_head: Number of attention heads in the model.

    Returns:
    - all_head_delta_matrices: A list of lists containing the delta matrices for each layer and head.
    """
    assert num_layer == attention_output_activations[0].shape[0], f"Expected {num_layer} layers, got {attention_output_activations[0].shape[0]}"
    assert num_head == attention_output_activations[0].shape[-2], f"Expected {num_head} heads, got {attention_output_activations[0].shape}"

    # Filter QA info for the current fold
    cur_fold_QA_info = QA_info.iloc[all_data_indices]
    correct_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 1].index.to_numpy()
    incorrect_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 0].index.to_numpy()

    # Stack activations for correct and incorrect answers
    correct_activations = np.stack([attention_output_activations[i] for i in correct_indices], axis=0)
    incorrect_activations = np.stack([attention_output_activations[i] for i in incorrect_indices], axis=0)

    # Initialize the delta matrices
    all_head_delta_matrices = [[None] * num_head for _ in range(num_layer)]

    # Calculate the delta matrices for each layer and head
    for layer in range(num_layer):
        for head in range(num_head):
            print(correct_activations.shape,incorrect_activations.shape)
            correct_matrix_mean = np.mean(correct_activations[:, layer,-1,-1, head, :], axis=0)
            incorrect_matrix_mean = np.mean(incorrect_activations[:, layer,-1,-1, head, :], axis=0)
            mean_delta_matrix = correct_matrix_mean - incorrect_matrix_mean
            all_head_delta_matrices[layer][head] = mean_delta_matrix

    return all_head_delta_matrices
    
    
def get_head_vector_info_list_dict(all_head_delta_matrices,top_head_idxs,num_layer,num_head,attention_output_activations,all_data_indices):
    """只干预必要的层!!!!
    Args:
        all_head_delta_matrices (_type_): _description_
        top_head_idxs (_type_): _description_
    Returns:
        head_vectors_list_dict: dict[layer_name]=[{"head":head1,...vector_info},{"head":"head2"}]
    """
    head_vectors_list_dict={}
    
    for layer in range(num_layer):
        layer_name=f"model.layers.{layer}.self_attn.o_proj"
        head_vectors_list_dict[layer_name]=[None]*num_head
        for head in range(num_head):
            if (layer,head) in top_head_idxs:
                delta_matrix=all_head_delta_matrices[layer][head]
                # 计算变化矩阵的 L2 范数
                magnitude = np.linalg.norm(delta_matrix)
                # 
                # 解耦方向：归一化变化矩阵
                direction = delta_matrix / magnitude # 获得一个 1*128
                # 这里应该加入映射函数
                # 计算变化方向上所有矩阵的标准差，用于归一化？
                mean_attention_activation=attention_output_activations[all_data_indices,layer,-1,-1,head,:]# datasize*128 在这个头上的矩阵
                
                vals_on_direction = mean_attention_activation @ direction.T# datasize*128 @ 128*1 -> datasize 个数值
                # 这里的标准差代表，激活值在某个方向上变化的剧烈程度，如果激活值在某一个方向上的变化很大，那么干预的强度也应该增大
                std_on_direction = np.std(vals_on_direction)#
                head_vectors_list_dict[layer_name][head]={"head":head,"direction":direction,"magnitude":magnitude,"std_on_direction":std_on_direction}
                
    return head_vectors_list_dict
        

def inference_with_activation_editing(model,tokenizer,QA_info,device,test_data_idxs, top_intervention_head_dict, intervention_fn, answer_space,start_edit_token_idx):
    
    # data
    test_data_info=QA_info.iloc[test_data_idxs]
    test_data_truth_labels=test_data_info["ground_truth_label"].to_numpy()
    test_data_input_prompts=test_data_info["input_text"].to_list()
    
    # 注意，这里我做了padding哦
    # all_prompt_ids=tokenizer(test_data_input_prompts, return_tensors = 'pt',padding=True).input_ids# 转为数字
    # print(all_prompt_ids.shape)
    # 2885 123 
    
    # 对所有数据有用的参数设置
    top_intervention_layers=top_intervention_head_dict.keys()
    intervention_fn = partial(intervention_fn, start_edit_token_idx=start_edit_token_idx)# partial就是固定模型的部分参数返回一个新的参数
    answer_space=lower_answer_space(answer_space)
    all_results=[]
    all_acc=[]
    with torch.no_grad():
        for idx,prompt in enumerate(test_data_input_prompts):
            try:
                response_text=""
                response_label=None
                prompt_ids=tokenizer(prompt, return_tensors = 'pt').input_ids.to(device)# 转为数字
                for infer_times in range(100):# 预防自动COT
                    with baukit.TraceDict(model, top_intervention_layers, edit_output=intervention_fn) as ret:
                        output= model(prompt_ids)
                        # 只干预最后一个token的输出
                        # 可能会有自主COT
                    next_token_id = torch.argmax(output.logits[0, -1, :]).item() # 选择最后一个token
                    next_token = tokenizer.decode(next_token_id)
                    response_text+=next_token
                    if replace_special_chars(str(next_token).lower()) in answer_space["True"]:# 如果生成答案在答案空间中
                        response_label=1
                        break
                    elif replace_special_chars(str(next_token).lower()) in answer_space["False"]: # 如果生成答案在答案空间中
                        response_label=0
                        break
                    else:
                        prompt_ids=torch.cat([prompt_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)  # 将新生成的token ID移动到GPU上,注意不能使用+=直接暴力拼接，需要用torch.cat拼接啊
            except Exception as e:
                raise ValueError(f"infer_time {infer_times},response {response_text}, shape {prompt_ids.shape}")
            all_results.append({"control_output_label":response_label,"control_output_text":response_text,"control_output_is_right":response_label==test_data_truth_labels[idx]})
            all_acc.append(response_label==test_data_truth_labels[idx])
            print(idx,response_text,"=",test_data_info.iloc[idx]["ground_truth_label"],float(sum(all_acc))/len(all_acc),test_data_info["is_right"].mean())
            # 按列拼接

    control_gen_pd=pd.DataFrame(all_results)
    final_result=pd.concat([test_data_info,control_gen_pd],axis=1)
    final_result.to_json('all_result_table.json', orient='table')
    assert len(control_gen_pd)==len(test_data_info)
    return final_result

