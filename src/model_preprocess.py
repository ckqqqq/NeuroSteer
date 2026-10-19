"""Load the language models and matching residual-stream SAEs used in experiments."""
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE


def load_models(LLM: str, layer: int, device: str):
    """Load a supported model/SAE pair; layer numbers follow the SAE hook index.

    GPT-2 has recovered task caches for layer 6. Gemma loading is retained for
    research experiments but does not have recovered four-task demo caches.
    """
    if LLM == "gpt2-small":
        if not 0 <= layer < 12:
            raise ValueError("GPT-2-small SAE layer must be between 0 and 11.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        release = "gpt2-small-res-jb"
        sae_id = f"blocks.{layer}.hook_resid_pre"
    elif LLM == "gemma-2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        release = "gemma-scope-2b-pt-res-canonical"
        sae_id = f"layer_{layer}/width_16k/canonical"
    else:
        raise ValueError(
            f"Unsupported model {LLM!r}. Available research loaders: gpt2-small, gemma-2-2b."
        )

    tokenizer.pad_token = tokenizer.eos_token
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    # Match the preprocessing used to train this SAE; default TransformerLens
    # folding/centering options can otherwise change the residual feature basis.
    model = HookedTransformer.from_pretrained_no_processing(
        LLM, device=device, tokenizer=tokenizer, **sae.cfg.model_from_pretrained_kwargs
    )
    model.eval()
    return model, sae, tokenizer
