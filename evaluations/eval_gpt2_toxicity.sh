# script1.sh
RESULT="/home/ckqsudo/code2024/CKQ_ACL2024/NeuroSteer/results"
echo "result path is: $RESULT"

python evaluation_main.py --gen_file $RESULT/toxicity/toxicity_alpha_100_from_neg_to_pos_datasize_1000_layer_5_mean_dif_mean_steertype_last_device_cuda_batchsize16/steer_gen_res.jsonl --out_file steer_eval --metrics ppl-small --del_end_of_sentence 1 --end_of_sentence "<|endoftext|>"

# 减去毒性的操作