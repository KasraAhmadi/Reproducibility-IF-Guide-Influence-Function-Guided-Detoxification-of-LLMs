#!/bin/bash

model_name="pythia-160m"
ckpt_dir="models/finetuning/pythia-160m/pythia_fine_tuned_kas_version/2026-05-01-02-41-35/checkpoint-2000" # example checkpoint path

defense="none"
dataset="RTP"

save_dir="pythia_paper_default_30000_samples_kas_version"

python run_toxicity_eval.py \
    --model_name $model_name \
    --checkpoint_dir $ckpt_dir \
    --dataset $dataset \
    --batch_size 64 \
    --save_dir $save_dir \
    --decoding_defense $defense \
    --save_outputs