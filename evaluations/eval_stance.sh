#!/bin/bash
RESULT="/home/ckqsudo/code2024/CKQ_ACL2024/NeuroSteer/results"
echo "result path is: $RESULT"

EVAL_FILE=$RESULT"/stance/gpt2-small_debate_layer_6_datasize_ALL_batchsize32_topK_150/alpha_15.0_from_pos_to_neg_prompt_neg_mean_dif_mean_steertype_all_device_cuda/steer_gen_res.jsonl"

/home/ckqsudo/miniconda3/envs/SAE/bin/python evaluation_origin.py --gen_file $EVAL_FILE --out_file steer_eval --metrics stance,ppl-small,dist-n --del_end_of_sentence 1 --end_of_sentence "<|endoftext|>"