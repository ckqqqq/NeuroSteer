#!/bin/bash
cd /home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations

export CUDA_VISIBLE_DEVICES=3
EVAL_FILE="/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/results/sentiment_gemma/gemma-2-2b_sentiment_layer_12_datasize_1500_batchsize32_topK_100/alpha_700.0_from_neg_to_pos_prompt_neg_mean_dif_mean_steertype_all_device_cpu/steer_gen_res.jsonl"

/home/ckqsudo/miniconda3/envs/SAE/bin/python evaluation_origin.py --gen_file $EVAL_FILE --out_file steer_eval --metrics ppl-small,dist-n,sentiment --del_end_of_sentence 1 --end_of_sentence "<|endoftext|>"

# sentiment eval