from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
def load_models(LLM:str,layer:int,device:str):
    if "llama" in LLM:
        raise ValueError("Some BUGs")
        # logging.info(f"Loading model: {args.LLM}")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="meta-llama/Meta-Llama-3.1-8B",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device
        )
        model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", device=device)
        # logging.info(f"model architecture for {args.LLM} {model}")

    elif "gemma-2-2b" in LLM:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gemma-scope-2b-pt-res-canonical",
            sae_id = f"layer_{layer}/width_16k/canonical",
            device=device)
        # logging.info(f"Loading model: {args.LLM}")
        model = HookedTransformer.from_pretrained(LLM, device=device)
    elif "gemma-2b" in LLM:
        raise ValueError("Some BUGs")
        # logging.info(f"Loading model: {args.LLM}")
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=f"{LLM}-res-jb", sae_id=f"blocks.{layer}.hook_resid_pre", device=device
        )
        model = HookedTransformer.from_pretrained(LLM, device=device)
    
    elif "gpt2-small" in LLM:
        # logging.info(f"Loading model: {args.LLM} SAE gpt2-small-res-jb")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=f"{LLM}-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device
        )
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # 设置填充标记为 EOS token
        tokenizer.pad_token = tokenizer.eos_token
        model = HookedTransformer.from_pretrained(LLM, device=device,tokenizer=tokenizer)
        # tokenizer = AutoTokenizer.from_pretrained(args.LLM)
    else:
        raise ValueError("No Supported")
    return model,sae,tokenizer