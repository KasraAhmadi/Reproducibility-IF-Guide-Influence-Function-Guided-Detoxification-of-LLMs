#!/bin/bash

source scripts/helper_scripts/find_open_port.sh
port=$(find_open_port)
echo "Using port $port"

num_gpus=1

model="pythia-160m"
save_id="pythia_fine_tuned_kas_version"

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
    --toxic_token_mask_path "../data/toxic_token_masks/ekfac_RTP_pythia-160m_kas_method/mask_toks=20.0m_p=0.99_w=1.pt" \
    --toxic_lambda 1.0 \
    --max_train_samples 30000 \
    --gradient_checkpointing \
    --effective_batch_size 4

    