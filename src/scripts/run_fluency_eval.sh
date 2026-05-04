#!/bin/bash

model_name="pythia-160m"

defense="none"
save_dir="baseline"
save_dir_1="default_30000_samples"
ckpt_dir_1="models/finetuning/pythia-160m/pythia_fine_tuned/2026-04-30-15-22-36/checkpoint-2000" # example checkpoint path
save_dir_2="default_30000_samples_kas_version"
ckpt_dir_2="models/finetuning/pythia-160m/pythia_fine_tuned_kas_version/2026-05-01-02-41-35/checkpoint-2000" # example checkpoint path



python run_fluency_eval.py \
    --model_name $model_name \
    --batch_size 2 \
    --save_dir $save_dir \
    --decoding_defense $defense \

# python run_fluency_eval.py \
#     --model_name $model_name \
#     --checkpoint_dir $ckpt_dir_1 \
#     --batch_size 2 \
#     --save_dir $save_dir_1 \
#     --decoding_defense $defense \

# python run_fluency_eval.py \
#     --model_name $model_name \
#     --checkpoint_dir $ckpt_dir_2 \
#     --batch_size 2 \
#     --save_dir $save_dir_2 \
#     --decoding_defense $defense \