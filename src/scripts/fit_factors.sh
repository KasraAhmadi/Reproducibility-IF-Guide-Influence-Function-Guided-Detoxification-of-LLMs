#!/bin/bash

model="gpt-neo-125m"
output_dir="models/training/gpt-neo-125m/baseline/2026-04-28"               # example output path

strategy="ekfac"

# python fit_factors.py \
#     --model_name $model \
#     --factor_strategy $strategy \
#     --output_dir $output_dir \
#     --train_batch_size 2 \
#     --use_half_precision \
python fit_factors.py \
    --model_name $model \
    --factor_strategy $strategy \
    --output_dir $output_dir \
    --train_batch_size 2 \
    --use_half_precision \
    --use_compile \
    --covariance_module_partitions 1 \
    --lambda_module_partitions 1