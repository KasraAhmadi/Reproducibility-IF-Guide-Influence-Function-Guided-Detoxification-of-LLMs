#!/bin/bash

model="pythia-160m"
scores_path="models/training/pythia-160m/baseline/2026-04-26/if_results/scores_RTP/ekfac_half_compile_qlr32/pairwise_scores.safetensors" # example scores path

python build_toxic_token_mask.py \
    --model_name $model \
    --window 1 \
    --toxicity_threshold 0.99 \
    --scores_path $scores_path \
    --inspection_idx 1 \
    --max_train_samples 30000 \
    --use_tfidf \