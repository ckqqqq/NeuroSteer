import argparse
import os
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
from sae_lens.analysis.feature_statistics import (
    get_all_stats_dfs,
    get_W_U_W_dec_stats_df,
)
from sae_lens.analysis.tsea import (
    get_enrichment_df,
    manhattan_plot_enrichment_scores,
    plot_top_k_feature_projections_by_token_and_category,
    get_baby_name_sets,
    get_letter_gene_sets,
    generate_pos_sets,
    get_test_gene_sets,
    get_gene_set_from_regex,
)
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import plotly_express as px
import logging
from typing import Tuple
import json
from log import setup_logging
from tqdm import tqdm  # 用于显示进度条


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer Lens Analysis Script")
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number to analyze."
    )
    parser.add_argument(
        "--LLM",
        type=str,
        default="gpt2-small",
        help="Name of the pre-trained model to load."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/ckqsudo/code2024/0dataset/emotional_classify/multiclass-sentiment-analysis",
        help="Path to the dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the results."
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env",
        help="Path to the .env file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling the dataset."
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=1000,
        help="Number of samples per class to select."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps","auto"],
        help="Device to run the computations on."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=100,
        help="Steering coefficient."
    )
    parser.add_argument(
        "--steer",
        type=str,
        choices=["pos", "neg", "neu"],
        required=True,
        help="Steer data sentiment preference."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mean", "val_mul"],
        default="val_mul",
        help="Method for computing steering vectors."
    )
    parser.add_argument(
        "--topk_mean",
        type=int,
        default=100,
        help="Number of top elements to select based on nz_mean."
    )
    parser.add_argument(
        "--topk_cnt",
        type=int,
        default=100,
        help="Number of top elements to select based on act_cnt."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing data."
    )
    return parser.parse_args()


def load_environment(env_path: str):
    load_dotenv(env_path)
    hf_endpoint = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
    logging.info(f"HF_ENDPOINT: {hf_endpoint}")


def load_and_prepare_dataset(dataset_path: str, seed: int, num_samples: int):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset["train"] = dataset['train'].shuffle(seed=seed)

    logging.info("Filtering dataset for negative, positive, and neutral samples")
    neg_train_set = dataset['train'].filter(lambda example: example['label'] == 0).select(range(num_samples))
    pos_train_set = dataset['train'].filter(lambda example: example['label'] == 2).select(range(num_samples))
    neu_train_set = dataset['train'].filter(lambda example: example['label'] == 1).select(range(num_samples))

    logging.info(f"Selected {len(neg_train_set)} negative, {len(pos_train_set)} positive, and {len(neu_train_set)} neutral samples")
    return neg_train_set, pos_train_set, neu_train_set


def compute_latents(sae: SAE, model: HookedTransformer, texts: list, hook_point: str, device: str, batch_size: int) -> list:
    """
    计算 latents，支持批次处理。

    Args:
        sae (SAE): SAE 实例。
        model (HookedTransformer): Transformer 模型实例。
        texts (list): 文本列表。
        hook_point (str): 钩子点名称。
        device (str): 计算设备。
        batch_size (int): 每个批次的大小。

    Returns:
        list: 包含每个批次 latents 的张量列表。
    """
    logging.info("Running model with cache to obtain hidden states")
    batch_latents = []

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        sv_logits, cache = model.run_with_cache(batch_texts, prepend_bos=True, device=device)
        batch_hidden_states = cache[hook_point]
        logging.info(f"Batch {i // batch_size + 1}: Hidden states shape: {batch_hidden_states.shape}")

        logging.info(f"Encoding hidden states for batch {i // batch_size + 1}")
        # 假设 sae.encode 支持批量编码
        latents = sae.encode(batch_hidden_states)  # 形状: (batch_size, latent_dim)
        batch_latents.append(latents)
        

    logging.info(f"Total batches processed: {len(batch_latents)}")
    return batch_latents


def analyze_latents(batch_latents: Tensor, top_k_mean: int = 100, top_k_cnt: int = 100) -> Tuple[Tensor, Tensor, Tensor]:
    logging.info("Computing non-zero element counts")
    act_cnt = (batch_latents != 0).sum(dim=(0, 1))

    logging.info("Computing sum of non-zero elements")
    nz_sum = torch.where(batch_latents != 0, batch_latents, torch.tensor(0.0, device=batch_latents.device)).sum(dim=(0, 1))

    logging.info("Computing mean of non-zero elements")
    nz_mean = torch.where(act_cnt != 0, nz_sum / act_cnt, torch.tensor(0.0, device=batch_latents.device))

    logging.info("Selecting top-k indices based on nz_mean")
    nz_act_val, nz_val_indices = torch.topk(nz_mean, top_k_mean)
    logging.info(f"Top {top_k_mean} nz_mean values selected.")

    logging.info("Selecting top-k indices based on act_cnt")
    nz_cnt, cnt_indices = torch.topk(act_cnt, top_k_cnt)
    logging.info(f"Top {top_k_cnt} act_cnt values selected.")

    # logging.info("Finding overlapping indices between nz_mean and act_cnt top-k")
    # overlap_mask = torch.isin(nz_val_indices, cnt_indices)
    # overlap_indices = nz_val_indices[overlap_mask]
    # logging.info(f"Number of overlapping indices: {len(overlap_indices)}")
    # overlap_indices=overlap_indices
    return nz_mean, act_cnt, cnt_indices


def compute_steering_vectors(sae: SAE, overlap_indices: Tensor, nz_mean: Tensor, method: str = "val_mul") -> Tensor:
    logging.info(f"Computing steering vectors using method: {method}")
    if method == "mean":
        steering_vectors = torch.mean(sae.W_dec[overlap_indices], dim=0)
    elif method == "val_mul":
        steering_vectors = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device)
        for important_idx in overlap_indices:
            steering_vectors += nz_mean[important_idx].item() * sae.W_dec[important_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    logging.info(f"Steering vectors computed with shape: {steering_vectors.shape}")
    return steering_vectors


def save_results(output_dir: str, nz_mean: Tensor, act_cnt: Tensor, generated_texts: list, hyperparams: dict):
    os.makedirs(output_dir, exist_ok=True)

    # Save nz_mean and act_cnt
    nz_stats_path = os.path.join(output_dir, 'nz_stats.pt')
    logging.info(f"Saving nz_mean and act_cnt to {nz_stats_path}")
    torch.save({
        'nz_mean': nz_mean,
        'act_cnt': act_cnt
    }, nz_stats_path)

    # Save generated texts
    generated_texts_path = os.path.join(output_dir, 'generated_texts.txt')
    logging.info(f"Saving generated texts to {generated_texts_path}")
    with open(generated_texts_path, 'w') as f:
        for text in generated_texts:
            f.write(text + "\n")

    # Save hyperparameters
    hyperparams_path = os.path.join(output_dir, 'hyperparameters.json')
    logging.info(f"Saving hyperparameters to {hyperparams_path}")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)

    logging.info("All results saved successfully.")


def calcu_similarity(sae, overlap_indices, nz_mean):
    # Example: Compute Euclidean distance and Cosine similarity
    mean_steering_vec = compute_steering_vectors(sae, overlap_indices, nz_mean, method="mean")
    mul_steering_vec = compute_steering_vectors(sae, overlap_indices, nz_mean, method="val_mul")
    euclidean_distance = torch.norm(mean_steering_vec - mul_steering_vec, p=2).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        mean_steering_vec.unsqueeze(0), mul_steering_vec.unsqueeze(0)
    ).item()
    logging.info(f"Euclidean distance between mean and val_mul steering vectors: {euclidean_distance}")
    logging.info(f"Cosine similarity between mean and val_mul steering vectors: {cosine_similarity}")


def main():
    args = parse_arguments()

    # Setup logging
    output_dir_base = os.path.join(
        args.output_dir,
        f"LLM_{args.LLM}_layer_{args.layer}_steer_{args.steer}_alpha_{args.alpha}_cnt_{args.topk_cnt}_mean{args.topk_mean}"
    )
    setup_logging(output_dir_base)

    # Save hyperparameters
    hyperparams = vars(args)

    # Log hyperparameters
    logging.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logging.info(f"  {key}: {value}")

    # Load environment
    load_environment(args.env_path)

    # Load model and SAE
    logging.info(f"Loading model: {args.LLM}")
    model = HookedTransformer.from_pretrained(args.LLM, device=args.device)

    logging.info(f"Loading SAE for layer {args.layer}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{args.layer}.hook_resid_pre",
        device=args.device
    )

    # Load dataset
    neg_train_set, pos_train_set, neu_train_set = load_and_prepare_dataset(
        args.dataset_path, args.seed, args.data_size
    )

    # Select a dataset steer based on steering preference
    if args.steer == "pos":
        selected_set = pos_train_set
    elif args.steer == "neg":
        selected_set = neg_train_set
    elif args.steer=="neu":
        selected_set = neu_train_set

    texts = selected_set["text"][:args.data_size]
    hook_point = sae.cfg.hook_name

    # Compute latents with batch processing
    batch_latents = compute_latents(sae, model, texts, hook_point, args.device, args.batch_size)

    # 将所有批次的 latents 合并为一个张量，以保持与无批处理时的数据一致
    # 这一步是必要的，因为后续的 analyze_latents 函数需要一个单一的张量
    # 如果您希望保留批次列表而不合并，请确保后续函数能够处理列表输入
    
    # 计算第二个维度的最大值
    max_dim1 = max(latent.shape[1] for latent in batch_latents)  # 第二个维度的最大值
    logging.info(f"最大长度:{max_dim1}")
    # 对每个 Tensor 进行填充（仅填充第二个维度）
    padded_latents_right = [
        torch.nn.functional.pad(latent, (0, 0, 0, max_dim1 - latent.size(1)), "constant", 0)
        for latent in batch_latents
    ]

    batch_latents_concatenated = torch.cat(padded_latents_right, dim=0)
    logging.info(f"Concatenated batch latents shape: {batch_latents_concatenated.shape}")

    # Analyze latents
    nz_mean, act_cnt, overlap_indices = analyze_latents(batch_latents_concatenated, top_k_mean=args.topk_mean, top_k_cnt=args.topk_cnt)

    # Compute steering vectors
    steering_vectors = compute_steering_vectors(sae, overlap_indices, nz_mean, method=args.method)

    # Calculate similarity between different steering methods
    calcu_similarity(sae, overlap_indices, nz_mean)

    # Define steering hook
    steering_on = True  # This will be toggled in run_generate
    alpha = args.alpha
    method = args.method  # Store method for clarity

    def steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1:
            return
        if steering_on:
            resid_pre += alpha * steering_vectors

    def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        with model.hooks(fwd_hooks=fwd_hooks):
            tokenized = model.to_tokens(prompt_batch)
            result = model.generate(
                stop_at_eos=True,
                input=tokenized,
                max_new_tokens=50,
                do_sample=True,
                **kwargs,
            )
        return result

    def run_generate(example_prompt: str, sampling_kwargs: dict) -> list:
        model.reset_hooks()
        editing_hooks = [(f"blocks.{args.layer}.hook_resid_post", steering_hook)]
        res = hooked_generate(
            [example_prompt] * 3,
            fwd_hooks=editing_hooks if steering_on else [],
            seed=args.seed,
            **sampling_kwargs
        )
        res_str = model.to_string(res[:, 1:])
        # generated_texts = res_str
        for idx, text in enumerate(res_str):
            logging.info(f"Generated Text {idx+1}: {text}")
            
        return res_str

    # Define sampling parameters
    sampling_kwargs = dict(temperature=1.0, top_p=0.5, freq_penalty=1.0)

    # Example prompt from the selected set
    example_prompt = "It is so"
    logging.info(f"Example prompt: {example_prompt}")

    # Generate without steering
    steering_on = False
    alpha = 0
    logging.info("Generating texts without steering...")
    generated_texts_no_steer = run_generate(example_prompt, sampling_kwargs)

    # Generate with steering
    steering_on = True
    alpha = args.alpha
    logging.info("Generating texts with steering...")
    generated_texts_with_steer = run_generate(example_prompt, sampling_kwargs)

    # Combine generated texts
    all_generated_texts = generated_texts_no_steer + generated_texts_with_steer

    # Save results
    save_results(
        output_dir=output_dir_base,
        nz_mean=nz_mean,
        act_cnt=act_cnt,
        generated_texts=all_generated_texts,
        hyperparams=hyperparams
    )


if __name__ == "__main__":
    main()
