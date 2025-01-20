from pyexpat import model
import torch
import baukit
import einops
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas
from functools import partial
import pandas as pd
from tqdm import tqdm
from utils import filter_special_chars,lower_chars
from decode_utils import LLM_decode_next_token, LLM_batch_decode_next_token

def get_activations_baubit(model, prompts, device, ground_truth_labels, question_ids, tokenizer, answer_space, decoding_strategy="greedy", sampling_value=None, max_infer_times=100): 
    """首先，使用列表推导式创建了两个列表 HEADS 和 MLPS，它们分别包含了模型中自注意力层和 MLP 层的名称，通过 model.config.num_hidden_layers 来确定层数范围，使用baukit.TraceDict 对 model 进行追踪，追踪的对象是 HEADS 和 MLPS 中的元素。

    Args:
        model (_type_): _description_
        prompt (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
        
    {"attention_output_activation":attention_output_activation, "mlp_output_activation":mlp_output_activation,"response_text":response_text,"response_label":response_label}
    """
    # 如果对其进行批次并行推理需要预先做padding  在这里加
    
    #定义要获取激活的注意力头和mlp
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    # 然后，使用 torch.no_grad() 上下文管理器，确保在接下来的操作中不会计算梯度，以节省计算资源。
    answer_space=lower_chars(answer_space)
    all_attention_ouput_activations=[]# 每个数据的attention输出
    all_mlp_output_activations=[]# 每个数据的mlp输出
    all_text_info={"input_text":prompts,"ground_truth_label":ground_truth_labels,"question_id":question_ids,"output_text":[],"output_label":[],"is_right":[]}
    
    for idx,prompt in enumerate(prompts):
        prompt_ids=tokenizer(prompt, return_tensors = 'pt').input_ids.to(device)# 转为prompt_id
        with torch.no_grad():
            response_text=""
            response_label=None
            try:
                for infer_times in range(max_infer_times):
                    output= model(prompt_ids)
                    logits = output.logits[0, -1, :]# batch_size x vocab_size
                    # 使用指定的解码策略选择下一个 token
                    next_token_id = LLM_decode_next_token(
                        logits, 
                        strategy=decoding_strategy, 
                        sampling_value=sampling_value
                    )
                    next_token = tokenizer.decode(next_token_id)
                    response_text+=next_token
                    if filter_special_chars(str(next_token).lower()) in answer_space["True"]:
                        response_label=1
                        break
                    elif filter_special_chars(str(next_token).lower()) in answer_space["False"]:
                        response_label=0
                        break
                    else:
                        prompt_ids=torch.cat([prompt_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)  # 将新生成的token ID移动到GPU上,注意不能使用+=直接暴力拼接，需要用torch.cat拼接啊
            except Exception as e:
                raise ValueError(f"infer_time {infer_times},response {response_text}, shape {prompt_ids.shape}")
            
            if infer_times > 0:
                print(f"推理中进行了COT {response_text}")
                if response_label is None:
                    print("response_text 的结果为none 证明推理没有采样到结果\n" * 3)


            with baukit.TraceDict(model, HEADS+MLPS) as ret:# 使用 TraceDict 对 model 进行追踪，追踪的对象是 HEADS 和 MLPS 中的层名字典。
                output = model(prompt_ids, output_hidden_states = False)
                """设置隐藏状态输出为false用于防止炸内存或者显存
                # output: 是一个'logits', 'past_key_values', 'hidden_states'（option）"""
            
            # hidden_states = output.hidden_states # 获取隐藏状态 debug qwen的输出是29层，比MLP的层数多一层，每一层是一个 batch_size x seq_len x 3584的一个向量
            attention_output_activation = [ret[head].output.detach().cpu() for head in HEADS] 
            attention_output_activation = torch.stack(attention_output_activation, dim = 0).numpy() # 将列表中的向量堆叠起来
            attention_output_activation=einops.rearrange(attention_output_activation, 'l b s (h d) -> l b s h d', h=model.config.num_attention_heads)# 
            """
            对注意力头进行分离，参考此处代码的2.7步: https://medium.com/@yuxiaojian/understand-how-llama3-1-works-a-deep-dive-into-the-model-flow-b149aba04bed
            形状变化为：layer ,batch_size, seq_length, hidden_size -> layer, batch_size, seq_length, num_attention_heads, head_dim
            """
            
            # 注意只要 seq_length 理论上要最后一列的最后一个token
            mlp_output_activation = [ret[mlp].output.detach().cpu() for mlp in MLPS] 
            mlp_output_activation = torch.stack(mlp_output_activation, dim = 0).numpy()
            
            # 只取last token的
            all_attention_ouput_activations.append(torch.tensor(attention_output_activation[:,:,-1:,:,:]))
            all_mlp_output_activations.append(torch.tensor(mlp_output_activation[:,:,-1:,:]))
            all_text_info["output_text"].append(response_text)
            all_text_info["output_label"].append(response_label)
            all_text_info["is_right"].append(ground_truth_labels[idx]==response_label)
            print(idx,response_text,prompt_ids.shape,sum(all_text_info["is_right"])/len(all_text_info["is_right"]))
            
            # print(f"model.config.num_attention_heads",model.config.num_attention_heads,"module name :",HEADS[0] ,"Tensor:",ret[HEADS[0]].output.detach().cpu().numpy().shape)
           
    # 返回 所有注意力头输出激活 和 所有mlp输出激活
    all_attention_ouput_activations=torch.stack(all_attention_ouput_activations,dim=0)
    all_mlp_output_activations=torch.stack(all_mlp_output_activations,dim=0)
    
    return all_attention_ouput_activations,all_mlp_output_activations,all_text_info

def get_activations_baubit_batch(model, prompts, device, ground_truth_labels, question_ids, tokenizer, answer_space, decoding_strategy="greedy", sampling_value=None, max_infer_times=100, batch_size=16):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    answer_space = lower_chars(answer_space)
    all_attention_output_activations = []
    all_mlp_output_activations = []
    all_text_info={"input_text":prompts,"ground_truth_label":ground_truth_labels,"question_id":question_ids,"output_text":[],"output_label":[],"is_right":[]}
    
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches)):
        # 获取当前批次的起始和结束索引
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        current_batch_size = end_idx - start_idx
        # 获取当前批次的prompts、ground_truth_labels和question_ids
        batch_prompts = prompts[start_idx:end_idx]
        batch_ground_truth_labels = ground_truth_labels[start_idx:end_idx]
        batch_question_ids = question_ids[start_idx:end_idx]
        batch_encodings = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        initial_prompt_ids = batch_encodings.input_ids.to(device)  # shape: (batch_size, seq_length)
        initial_attention_mask = batch_encodings.attention_mask.to(device)  # shape: (batch_size, seq_length)
        
        prompt_ids = [initial_prompt_ids[i].tolist() for i in range(current_batch_size)]
        attention_masks = [initial_attention_mask[i].tolist() for i in range(current_batch_size)]
        
        response_texts = [""] * current_batch_size
        response_labels = [None] * current_batch_size
        finished = [False] * current_batch_size  # 标记哪些样本已经生成了标签
        
        # 使用torch.no_grad()上下文管理器，确保在推理过程中不会计算梯度，以节省计算资源
        with torch.no_grad():
            try:
                for infer_time in range(max_infer_times):
                    unfinished_indices = [i for i, done in enumerate(finished) if not done]  # 找出尚未完成的样本索引
                    if not unfinished_indices:
                        break
                    
                    # 准备当前批次的 prompt_ids 和 attention_mask
                    current_prompt_ids_batch = [prompt_ids[i] for i in unfinished_indices]
                    current_attention_mask_batch = [attention_masks[i] for i in unfinished_indices]
                    
                    max_seq_length = max(len(pids) for pids in current_prompt_ids_batch)
                    padded_prompt_ids = [pids + [tokenizer.pad_token_id] * (max_seq_length - len(pids)) for pids in current_prompt_ids_batch]  # 对当前批次进行 padding
                    padded_attention_mask = [mask + [0] * (max_seq_length - len(mask)) for mask in current_attention_mask_batch]
                    
                    input_ids_tensor = torch.tensor(padded_prompt_ids, dtype=torch.long).to(device)  # shape: (unfinished_batch_size, max_seq_length)
                    attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long).to(device)  # shape: (unfinished_batch_size, max_seq_length)
                    
                    # 将当前prompt_ids输入模型，获取模型输出
                    output = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
                    
                    # 获取最后一个token的logits，形状为 (unfinished_batch_size, vocab_size)
                    logits = output.logits[:, -1, :]  # shape: (unfinished_batch_size, vocab_size)
                    
                    # 使用指定的解码策略选择下一个token的ID
                    next_token_ids = LLM_batch_decode_next_token(
                        logits, 
                        strategy=decoding_strategy, 
                        sampling_value=sampling_value
                    )  # shape: (unfinished_batch_size,)
                    
                    next_tokens = [tokenizer.decode(tid, skip_special_tokens=True) for tid in next_token_ids]
                    
                    for idx_in_batch, token in enumerate(next_tokens):
                        prompt_idx = unfinished_indices[idx_in_batch]
                        response_texts[prompt_idx] += token
                        
                        filtered_token = filter_special_chars(token.lower())
                        if filtered_token in answer_space["True"]:
                            response_labels[prompt_idx] = 1
                            finished[prompt_idx] = True
                        elif filtered_token in answer_space["False"]:
                            response_labels[prompt_idx] = 0
                            finished[prompt_idx] = True
                        else:
                            new_token_id = next_token_ids[idx_in_batch].item()
                            prompt_ids[prompt_idx].append(new_token_id)
                            attention_masks[prompt_idx].append(1)
            except Exception as e:
                raise ValueError(f"Batch {batch_idx}, infer_time {infer_time}, responses {response_texts}") from e
            
            if infer_time > 0:
                print(f"Batch {batch_idx} 推理中进行了COT {response_texts}")
                for i, label in enumerate(response_labels):
                    if label is None:
                        print(f"response_text[{i}] 的结果为None，证明推理没有采样到结果\n" * 3)
        
        # 将 prompt_ids 和 attention_masks 转换为张量，并进行 padding
        max_seq_length_final = max(len(pids) for pids in prompt_ids)
        
        padded_prompt_ids_final = [pids + [tokenizer.pad_token_id] * (max_seq_length_final - len(pids)) for pids in prompt_ids]  # 对所有样本进行 padding
        padded_attention_mask_final = [mask + [0] * (max_seq_length_final - len(mask)) for mask in attention_masks]
        
        prompt_ids_tensor_final = torch.tensor(padded_prompt_ids_final, dtype=torch.long).to(device)  # shape: (batch_size, max_seq_length_final)
        attention_mask_tensor_final = torch.tensor(padded_attention_mask_final, dtype=torch.long).to(device)  # shape: (batch_size, max_seq_length_final)
        
        # 使用baukit.TraceDict对模型进行追踪，监控HEADS和MLPS中指定的层
        with baukit.TraceDict(model, HEADS + MLPS) as ret:
            # 将最终的 prompt_ids 和 attention_mask 输入模型，获取输出，此时 TraceDict 会记录指定层的激活
            output = model(input_ids=prompt_ids_tensor_final, attention_mask=attention_mask_tensor_final, output_hidden_states=False)
            """
            设置隐藏状态输出为False，以防止占用过多的内存或显存。
            输出包括 'logits' 和 'past_key_values'，可选的 'hidden_states' 也被禁用。
            """
        
        # 从 TraceDict 的返回值中提取自注意力层的输出激活，并将其从 GPU 转移到 CPU 上
        attention_output_activation = [ret[head].output.detach().cpu() for head in HEADS]
        # 将注意力层的激活列表堆叠成一个张量，形状为 (层数, batch_size, seq_length, head_dim)
        attention_output_activation = torch.stack(attention_output_activation, dim=0).numpy()
        
        # 使用 einops 重新排列张量的形状，将头维度分离出来，形状变为 (层数, batch_size, seq_length, num_attention_heads, head_dim)
        attention_output_activation = einops.rearrange(
            attention_output_activation, 
            'l b s (h d) -> l b s h d', 
            h=model.config.num_attention_heads
        )
        """
        对注意力头进行分离，参考此处代码的2.7步: https://medium.com/@yuxiaojian/understand-how-llama3-1-works-a-deep-dive-into-the-model-flow-b149aba04bed
        形状变化为：layer, batch_size, seq_length, hidden_size -> layer, batch_size, seq_length, num_attention_heads, head_dim
        """
        mlp_output_activation = [ret[mlp].output.detach().cpu() for mlp in MLPS]
        mlp_output_activation = torch.stack(mlp_output_activation, dim=0).numpy()
        
        # 只取序列中最后一个 token 的注意力输出和 MLP 输出激活
        for i in tqdm(range(current_batch_size)):
            attention_act = attention_output_activation[:, i, -1:, :, :]
            all_attention_output_activations.append(torch.tensor(attention_act))
            mlp_act = mlp_output_activation[:, i, -1:, :]
            all_mlp_output_activations.append(torch.tensor(mlp_act))
        
        # 将生成的响应文本、响应标签以及预测是否正确的信息添加到 all_text_info 字典中
        all_text_info["output_text"].extend(response_texts)
        all_text_info["output_label"].extend(response_labels)
        all_text_info["is_right"].extend([gt == resp for gt, resp in zip(batch_ground_truth_labels, response_labels)])
        
        # 计算当前批次的预测正确率
        correct = sum([gt == resp for gt, resp in zip(batch_ground_truth_labels, response_labels) if resp is not None])
        total = sum([resp is not None for resp in response_labels])
        accuracy = correct / total if total > 0 else 0
        print(f"Batch {batch_idx}: 正确率 = {accuracy:.2f}")
        
        # 可选的调试信息，打印注意力头的配置和激活张量的形状
        # print(f"model.config.num_attention_heads", model.config.num_attention_heads, "module name :", HEADS[0], "Tensor:", ret[HEADS[0]].output.detach().cpu().numpy().shape)
    
    all_attention_output_activations = torch.stack(all_attention_output_activations, dim=0)
    all_mlp_output_activations = torch.stack(all_mlp_output_activations, dim=0)
    return all_attention_output_activations, all_mlp_output_activations, all_text_info


def train_probes_with_attention_activations(train_data_idxs, val_data_idxs, attention_output_activations,QA_info:pandas.DataFrame,num_layer,num_head,seed):
    """ 训练探针的函数
    Args:
        train_data_idxs (_type_): _description_
        val_data_idxs (_type_): _description_
        attention_output_activations (_type_): attention_output_activations: 是一个列表，列表中的每个元素是一个tensor，其形态为 layer*batch_size*seq_len*num_heads*head_dim
        QA_info (pandas.DataFrame): _description_
        num_layer (_type_): _description_
        num_head (_type_): _description_
        seed (_type_): _description_

    Returns:
        tuple: (probes, acc_probes) 探针2D列表和对应准确度
    """
    
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
    
    
    print("训练集形状",all_X_train.shape,"训练集GT标签形状",y_train.shape)
    probes=[[None]* num_head for _ in range(num_layer)]
    acc_probes=[[None]* num_head for _ in range(num_layer)]
    for layer in range(num_layer): 
        for head in range(num_head): 
            X_train = all_X_train[:,layer,-1,-1,head,:] # 样本量*特征
            X_val = all_X_val[:,layer,-1,-1,head,:]
    
            lr_probe = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            # lr拟合正负样本
            # y_pred = lr_probe.predict(X_train)#   这是训练集上的生成标签
            y_val_pred = lr_probe.predict(X_val)# 这是验证集上的生成标签
            probes[layer][head]=lr_probe
            acc_probes[layer][head]=accuracy_score(y_val, y_val_pred)# 这是验证集上的准确度

    # all_head_accs_np = np.array(all_head_accs) # 返回所有头的准确度
    return probes, acc_probes


def get_all_head_delta_matrices_cot2judge(all_data_indices, attention_output_activations_cot, attention_output_activations_judge, QA_info: pd.DataFrame, num_layer, num_head):
    """
    Calculate the delta matrices for each attention head between correct and incorrect answers.
    使用GPT使得本函数规范化
    Parameters:
    - data_indices: Indices of the data points to consider.
    - attention_output_activations: List of attention output activations for each data point. Data_size*layer*batch_size*seq_len*num_heads*head_dim  batch_size=1 seq_len=1
    - QA_info: DataFrame containing QA information, including 'is_right' column.
    - num_layer: Number of layers in the model.
    - num_head: Number of attention heads in the model.

    Returns:
    - all_head_delta_matrices: A list of lists containing the delta matrices for each layer and head.
    """
    assert num_layer == attention_output_activations_cot[0].shape[0], f"Expected {num_layer} layers, got {attention_output_activations_cot[0].shape[0]}"
    assert num_layer == attention_output_activations_judge[0].shape[0], f"Expected {num_layer} layers, got {attention_output_activations_judge[0].shape[0]}"
    assert num_head == attention_output_activations_cot[0].shape[-2], f"Expected {num_head} heads, got {attention_output_activations_cot[0].shape}"
    assert num_head == attention_output_activations_judge[0].shape[-2], f"Expected {num_head} heads, got {attention_output_activations_judge[0].shape}"

    # Filter QA info for the current fold
    cur_fold_QA_info = QA_info.iloc[all_data_indices]
    correct_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 1].index.to_numpy()
    incorrect_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 0].index.to_numpy()

    # Stack activations for correct and incorrect answers
    correct_activations = np.stack([attention_output_activations_cot[i] for i in correct_indices], axis=0)
    incorrect_activations = np.stack([attention_output_activations_judge[i] for i in incorrect_indices], axis=0)

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
    

def get_all_head_SAEs(attention_output_activations, top_head_indices, QA_info, all_data_indices, num_layer, num_head, seed,device, num_epochs=50, val_ratio=0.3):
    """
    对每个注意力头训练一个SAE，目标是重建每个头的输入向量。
    
    Parameters:
    - attention_output_activations: 经过注意力机制的激活输出 (numpy array)
    - top_head_indices: 排序后的 (层, 头) 索引，按重要性降序排列
    - QA_info: 问题和答案信息，用于选择正确和错误样本
    - all_data_indices: 当前折的数据索引
    - num_layer: 总的层数
    - num_head: 总的头数
    - seed: 随机种子
    - num_epochs: 训练的轮数
    - val_ratio: 验证集的比例
    """
    assert num_layer == attention_output_activations[0].shape[0], f"Expected {num_layer} layers, got {attention_output_activations[0].shape[0]}"
    assert num_head == attention_output_activations[0].shape[-2], f"Expected {num_head} heads, got {attention_output_activations[0].shape}"

    # 筛选QA信息
    cur_fold_QA_info = QA_info.iloc[all_data_indices]
    correct_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 1].index.to_numpy()
    incorrect_indices = cur_fold_QA_info[cur_fold_QA_info["is_right"] == 0].index.to_numpy()

    # 准备训练数据
    correct_activations = np.stack([attention_output_activations[i] for i in correct_indices], axis=0)
    incorrect_activations = np.stack([attention_output_activations[i] for i in incorrect_indices], axis=0)
    
    from train_utils import train_v6
    top_saes=[]
    # 遍历每个注意力头并训练相应的SAE
    for (layer, head) in top_head_indices:
        pos_samples = correct_activations[:, layer, -1, -1, head, :]  # 获取正确样本的注意力头向量
        neg_samples = incorrect_activations[:, layer, -1, -1, head, :]  # 获取错误样本的注意力头向量

        # 将正负样本拼接在一起
        data_np = np.concatenate([pos_samples, neg_samples], axis=0)
        
        # Shuffle the concatenated data (both positive and negative samples)
        np.random.shuffle(data_np)  # Shuffle the entire dataset

        # 使用train_v6函数训练SAE
        print("layer-",layer,"head-",head,"training","total",len(top_head_indices))
        sae_model = train_v6(data_np=data_np, epochs=num_epochs,val_ratio=val_ratio)
        
        top_saes.append(sae_model)
    all_head_delta_matrices = [[None] * num_head for _ in range(num_layer)]
    for idx,(layer,head) in enumerate(top_head_indices):
        sae_model=top_saes[idx]
        
        pos_samples = torch.tensor(correct_activations[:, layer, -1, -1, head, :]).to(device)  # 获取正确样本的注意力头向量
        neg_samples = torch.tensor(incorrect_activations[:, layer, -1, -1, head, :]).to(device)  # 获取错误样本的注意力头向量
        # sae_model
        with torch.no_grad():
            pos_mean_activations = []
            for data_idx in range(len(pos_samples)):
                latents=sae_model.encode(pos_samples[data_idx])
                pos_mean_activations.append(latents)
            neg_mean_activations = []
            for data_idx in range(len(neg_samples)):
                latents=sae_model.encode(neg_samples[data_idx])
                neg_mean_activations.append(latents)
            pos_mean_activations = torch.stack(pos_mean_activations, dim=0)
            neg_mean_activations = torch.stack(neg_mean_activations, dim=0)
            # 计算 delta_latent
            delta_latent = pos_mean_activations.mean(dim=0) - neg_mean_activations.mean(dim=0)
            steer_matrix=sae_model.decode(delta_latent)
            all_head_delta_matrices[layer][head]=steer_matrix
    return all_head_delta_matrices
    
def get_all_head_delta_matrices(all_data_indices, attention_output_activations, QA_info: pd.DataFrame, num_layer, num_head):
    """
    Calculate the delta matrices for each attention head between correct and incorrect answers.
    得到平均差值矩阵
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
            correct_matrix_mean = np.mean(correct_activations[:, layer,-1,-1, head, :], axis=0) #128*1
            incorrect_matrix_mean = np.mean(incorrect_activations[:, layer,-1,-1, head, :], axis=0) #128*1
            mean_delta_matrix = correct_matrix_mean - incorrect_matrix_mean
            all_head_delta_matrices[layer][head] = mean_delta_matrix

    return all_head_delta_matrices

    

        
    
def get_head_steer_info_list_dict(all_head_delta_matrices,top_head_indices,num_layer,num_head,attention_output_activations,all_data_indices,add_type):
    """只干预top_head_indices中必要的层!!!!
    Args:
        all_head_delta_matrices (_type_): _description_
        top_head_indices (_type_): _description_
    Returns:
        head_vectors_list_dict: dict[layer_name]=[{"head":head1,...vector_info},{"head":"head2"}]
    """
    head_vectors_list_dict={}
    assert add_type in ["matrix","vector"]
    
    for layer in range(num_layer):
        layer_name=f"model.layers.{layer}.self_attn.o_proj"
        head_vectors_list_dict[layer_name]=[None]*num_head
        for head in range(num_head):
            if (layer,head) in top_head_indices:
                if add_type=="vector":
                    delta_matrix=all_head_delta_matrices[layer][head]
                    # 计算变化矩阵
                    
                    magnitude = torch.linalg.norm(delta_matrix)
                    # 计算最重要的解耦方向
                    direction = delta_matrix / magnitude # 获得一个 1*128大小的矩阵
                    mean_attention_activation=attention_output_activations[all_data_indices,layer,-1,-1,head,:]# 计算变化方向上所有矩阵的标准差，用于归一化
                    vals_on_direction = mean_attention_activation @ direction.T# datasize*128 @ 128*1 -> datasize 个数值                
                    std_on_direction = torch.std(vals_on_direction)# 这里的标准差代表，激活值在某个方向上变化的剧烈程度，如果激活值在某一个方向上的变化很大，那么干预的强度也应该增大
                    head_vectors_list_dict[layer_name][head]={"head":head,"direction":direction,"magnitude":magnitude,"std_on_direction":std_on_direction}
                elif add_type=="matrix":
                    delta_matrix=all_head_delta_matrices[layer][head]
                    head_vectors_list_dict[layer_name][head]={"head":head,"delta_matrix":delta_matrix}
                
    return head_vectors_list_dict
        

def inference_with_activation_editing(model,tokenizer,QA_info,device,test_data_idxs, top_intervention_head_dict, intervention_fn, answer_space, start_edit_token_idx, decoding_strategy='greedy', top_p_or_k=None):
    """推理干预函数,在验证集上获取推理结果
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        QA_info (_type_): _description_
        device (_type_): _description_
        test_data_idxs (_type_): _description_
        top_intervention_head_dict (_type_): _description_
        intervention_fn (_type_): _description_
        answer_space (_type_): _description_
        start_edit_token_idx (_type_): _description_
        decoding_strategy (str, optional): _description_. Defaults to 'greedy'.
        sampling_value (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # data
    test_data_info=QA_info.iloc[test_data_idxs]
    test_data_truth_labels=test_data_info["ground_truth_label"].to_numpy()
    test_data_input_prompts=test_data_info["input_text"].to_list()
    
    # all_prompt_ids=tokenizer(test_data_input_prompts, return_tensors = 'pt',padding=True).input_ids  #padding
    
    # parameters
    top_intervention_layers=top_intervention_head_dict.keys()
    intervention_fn = partial(intervention_fn, start_edit_token_idx=start_edit_token_idx)# partial就是固定模型的部分参数返回一个新的参数
    answer_space=lower_chars(answer_space)
    all_results=[]
    all_acc=[]
    with torch.no_grad():
        for idx,prompt in enumerate(test_data_input_prompts):
            try:
                response_text,response_label="",None
                prompt_ids=tokenizer(prompt, return_tensors = 'pt').input_ids.to(device)# 转为token_ids
                for infer_times in range(100):# 预防自动COT
                    with baukit.TraceDict(model, top_intervention_layers, edit_output=intervention_fn) as ret:
                        output= model(prompt_ids)
                        # 只干预最后一个token的输出可能会有自主COT，所以会有100次推理
                    logits = output.logits[0, -1, :]
                    # next_token_id = output.logits[0, -1, :].argmax() 使用指定的解码策略选择下一个 token
                    next_token_id = LLM_decode_next_token(
                        logits, 
                        strategy=decoding_strategy, 
                        sampling_value=top_p_or_k
                    )
                    next_token = tokenizer.decode(next_token_id)
                    response_text+=next_token
                    if filter_special_chars(str(next_token).lower()) in answer_space["True"]:# 如果生成答案在答案空间中
                        response_label=1
                        break
                    elif filter_special_chars(str(next_token).lower()) in answer_space["False"]: # 如果生成答案在答案空间中
                        response_label=0
                        break
                    else:
                        prompt_ids=torch.cat([prompt_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)  # 将新生成的token ID移动到GPU上,注意不能使用+=直接暴力拼接，需要用torch.cat拼接啊
            except Exception as e:
                raise ValueError(f"infer_time {infer_times},response {response_text}, shape {prompt_ids.shape}")
            # 保存结果
            all_results.append({"control_output_label":response_label,"control_output_text":response_text,"control_output_is_right":response_label==test_data_truth_labels[idx]})
            all_acc.append(response_label==test_data_truth_labels[idx])
            print(idx,response_text,"=",test_data_info.iloc[idx]["ground_truth_label"],float(sum(all_acc))/len(all_acc),test_data_info["is_right"].mean())
    

    control_gen_pd=pd.DataFrame(all_results)
    final_result=pd.concat([test_data_info,control_gen_pd],axis=1)
    final_result.to_json('all_result_table.json', orient='table')
    assert len(control_gen_pd)==len(test_data_info)
    return final_result

