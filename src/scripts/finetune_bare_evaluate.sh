#!/bin/bash
model="pythia-160m"
save_id="baseline"
defense="none"
save_dir="baseline"
ckpt_dir="models/finetuning/pythia-160m/baseline/2026-05-04/checkpoint-2000" # example checkpoint path
dataset="RTP"
save_dir="pythia_baseline"

python finetune.py \
    --model_name $model \
    --learning_rate 6e-5 \
    --weight_decay 4e-4 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --max_steps 2000 \
    --logging_steps 25 \
    --seed 1004 \
    --save_id $save_id \
    --max_train_samples 30000 \
    --gradient_checkpointing \
    --effective_batch_size 4


python run_fluency_eval.py \
    --model_name $model \
    --batch_size 2 \
    --save_dir $save_dir \
    --checkpoint_dir $ckpt_dir \
    --decoding_defense $defense \



python run_toxicity_eval.py \
    --model_name $model \
    --checkpoint_dir $ckpt_dir \
    --dataset $dataset \
    --batch_size 64 \
    --save_dir $save_dir \
    --decoding_defense $defense \
    --save_outputs
