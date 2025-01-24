import argparse
import os
import re
import logging
from typing import Tuple
from tqdm import tqdm  # For progress bars
from log import setup_logging
import logging
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import numpy as np
import argparse
import logging

from data_preprocess import load_and_prepare_triple_dataset,load_and_prepare_COT_dataset,load_and_prepare_debate_triple_dataset
from utils import load_environment
# from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
# from sae_lens.analysis.feature_statistics import (
#     get_all_stats_dfs,
#     get_W_U_W_dec_stats_df,
# )
# from sae_lens.analysis.tsea import (
#     get_enrichment_df,
#     manhattan_plot_enrichment_scores,
#     plot_top_k_feature_projections_by_token_and_category,
#     get_baby_name_sets,
#     get_letter_gene_sets,
#     generate_pos_sets,
#     get_test_gene_sets,
#     get_gene_set_from_regex,
# )
# import plotly_express as px


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='sentiment',choices=['sentiment','cot','polite','toxicity'], help='Task type: sentiment, cot, polite')
parser.add_argument('--layer', type=int, default=6, help='Layer number to analyze')
parser.add_argument('--LLM', type=str, default='gpt2-small', help='LLM model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
from typing import Union
parser.add_argument('--data_size', type=int, default=-1, help='Data size')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use')
parser.add_argument('--alpha', type=int, default=500, help='Alpha parameter')
# parser.add_argument('--steer', type=str, default='neg-pos', help='Steering direction')
parser.add_argument('--method', type=str, default='val_mul', choices=['mean', 'val_mul'], help='Method to use')
parser.add_argument('--topk_mean', type=int, default=100, help='Top K mean selection')
parser.add_argument('--topk_cnt', type=int, default=100, help='Top K count selection')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--source', type=str, default='pos', help='Source class')#赞同、积极情感、礼貌、COT、无毒性
parser.add_argument('--target', type=str, default='neg', help='Target class')#不赞同、消极情感、不礼貌、直接推理、无毒性
parser.add_argument('--mean_type', type=str, default="dif_mean",choices=['dif_mean','tar_mean'], help='Mean type')
parser.add_argument('--steer_type', type=str, default="last",choices=['all','last','last2',"gaussian"], help='Steer type')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
parser.add_argument('--dataset_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/sentiment/sst5", help='Dataset path')
parser.add_argument('--prompt_path', type=str, default="/home/ckqsudo/code2024/0dataset/baseline-acl/data/prompts/sentiment_prompts-10k", help='Prompt path')
parser.add_argument('--env_path', type=str, default="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/.env", help='Environment path')
# 垃圾args，用布尔会有奇怪的东西
parser.add_argument('--debug', type=int, default=0, choices=[0, 1], help='Debug flag: 0 for False, 1 for True')
parser.add_argument("--save_compared", type=int,default=0, choices=[0, 1], help='是否需要比较GPT原先生成的结果')


args = parser.parse_args()

