#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Step 1: Load Trained Models & Data Selection Results

# Step 2: Prediction Cross-lingual Zero-shot NER in target languages
GPU_ID=0
SEEDS=(320)
# SEEDS=(320 550 631 691 985)
TGTS=(es nl)

for seed in ${SEEDS[@]}; do
    for tgt in ${TGTS[@]}; do
        python3 kd.py \
            --seed ${seed} \
            --gpu_id ${GPU_ID} \
            --batch_size 32 \
            --tgt_lang ${tgt} \
            --eval_langs ${tgt} \
            --do_predict True
    done
done
