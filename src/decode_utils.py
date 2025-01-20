
import torch
def LLM_decode_next_token(logits, strategy='greedy', sampling_value=None):
    if strategy == 'greedy':    # 贪心解码：选择概率最高的 token
        next_token_id = torch.argmax(logits).item()
        return next_token_id
    elif strategy == 'topk':
        if sampling_value is None or isinstance(sampling_value,int)==False :
            raise ValueError("For 'topk' strategy, 'sampling_value' must be set to the desired k value.(int)")
        top_k = min(sampling_value, logits.size(-1))  # 防止 top_k 大于词汇表大小
        # 获取 top_k 的概率和对应的 token IDs
        topk_probs, topk_indices = torch.topk(torch.nn.functional.softmax(logits.float(), dim=-1), top_k, dim=-1)
        topk_probs = topk_probs.squeeze()
        topk_indices = topk_indices.squeeze()
        topk_probs = topk_probs / topk_probs.sum()    # 归一化概率
        next_token_id = torch.multinomial(topk_probs, num_samples=1).item()    # 进行采样
        return topk_indices[next_token_id].item()
    elif strategy == 'topp':
        if sampling_value is None:
            raise ValueError("For 'topp' strategy, 'sampling_value' must be set to the desired p value.")
        sorted_probs, sorted_indices = torch.sort(torch.nn.functional.softmax(logits.float(), dim=-1), descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 保留累积概率 <= sampling_value 的 token
        sorted_indices_to_keep = cumulative_probs <= sampling_value
        # 保留第一个超过 sampling_value 的 token
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1
        # 创建一个 mask
        indices_to_remove = ~sorted_indices_to_keep
        sorted_probs[indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_token_id = torch.multinomial(sorted_probs, num_samples=1).item()    # 进行采样
        return sorted_indices[next_token_id].item()
    else:
        raise ValueError(f"Unsupported decoding strategy: {strategy}")
    

def LLM_batch_decode_next_token(logits, strategy='greedy', sampling_value=None):
    """
    根据给定的解码策略从logits中选择下一个token的ID。
    
    Args:
        logits (torch.Tensor): 模型输出的logits，形状为 (batch_size, vocab_size)。
        strategy (str, optional): 解码策略，默认为 'greedy'。可选值包括 'greedy', 'topk', 'topp' 等。
        sampling_value (float or int, optional): 根据解码策略的不同，可能代表 top_k 或 top_p 的值。
    
    Returns:
        torch.Tensor: 选择的下一个token的ID，形状为 (batch_size,)。
    """
    if strategy == 'greedy':    # 贪心解码：选择概率最高的 token
        next_token_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size,)
        return next_token_ids
    elif strategy == 'topk':
        if sampling_value is None or not isinstance(sampling_value, int):
            raise ValueError("For 'topk' strategy, 'sampling_value' must be set to the desired k value.(int)")
        top_k = min(sampling_value, logits.size(-1))  # 防止 top_k 大于词汇表大小
        # 获取 top_k 的概率和对应的 token IDs
        topk_probs, topk_indices = torch.topk(torch.nn.functional.softmax(logits.float(), dim=-1), top_k, dim=-1)  # shape: (batch_size, top_k)
        # 归一化概率
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # shape: (batch_size, top_k)
        # 进行采样
        sampled_indices = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)  # shape: (batch_size,)
        # 获取对应的 token IDs
        next_token_ids = topk_indices[torch.arange(logits.size(0)), sampled_indices]  # shape: (batch_size,)
        return next_token_ids
    elif strategy == 'topp':
        if sampling_value is None:
            raise ValueError("For 'topp' strategy, 'sampling_value' must be set to the desired p value.")
        # 计算 softmax
        sorted_probs, sorted_indices = torch.sort(torch.nn.functional.softmax(logits.float(), dim=-1), descending=True)  # shape: (batch_size, vocab_size)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # shape: (batch_size, vocab_size)
        # 保留累积概率 <= sampling_value 的 token
        sorted_indices_to_keep = cumulative_probs <= sampling_value  # shape: (batch_size, vocab_size)
        # 保留第一个超过 sampling_value 的 token
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1  # 保留第一个 token
        # 创建一个 mask
        indices_to_remove = ~sorted_indices_to_keep
        sorted_probs[indices_to_remove] = 0
        # 重新归一化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        # 进行采样
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)  # shape: (batch_size,)
        # 获取对应的 token IDs
        next_token_ids = sorted_indices[torch.arange(logits.size(0)), sampled_indices]  # shape: (batch_size,)
        return next_token_ids
    else:
        raise ValueError(f"Unsupported decoding strategy: {strategy}")