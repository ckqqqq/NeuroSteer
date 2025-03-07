#!/bin/bash
RESULT="/home/ckqsudo/code2024/CKQ_ACL2024/NeuroSteer/results"
echo "result path is: $RESULT"

EVAL_FILE=$RESULT"/polite/gpt2-small_polite_layer_6_datasize_ALL_batchsize32_topK_150/alpha_10.0_from_pos_to_neg_prompt_pos_mean_dif_mean_steertype_all_device_cuda/steer_gen_res.jsonl"

/home/ckqsudo/miniconda3/envs/SAE/bin/python evaluation_main.py --gen_file $EVAL_FILE --out_file steer_eval --metrics polite-neg,ppl-small,dist-n --del_end_of_sentence 1 --end_of_sentence "<|endoftext|>"