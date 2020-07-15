#!/bin/bash

GPUNO=0
# SEED=95 495 539 667 806
SEEDS=(95)
# KSHOT=0.01 0.02 0.05
KSHOT=(0.001)
PY_ALIAS=python3

# base model
for seed in ${SEEDS[@]}; do

    # ==> train-baseModel
    ${PY_ALIAS} main.py \
        --no_meta_learning \
        --mask_rate -1.0 \
        --lambda_max_loss 0.0 \
        --result_dir baseModel-seed_${seed} \
        --gpu_device ${GPUNO} \
        --seed ${seed} \
        --py_alias ${PY_ALIAS}

    # ==> 0-shot-baseModel
    ${PY_ALIAS} main.py \
        --zero_shot \
        --no_meta_learning \
        --test_langs es nl de \
        --model_dir models/baseModel-seed_${seed} \
        --gpu_device ${GPUNO} \
        --seed ${seed} \
        --py_alias ${PY_ALIAS}

    for k in ${KSHOT[@]}; do

        # ==> k-shot-baseModel
        ${PY_ALIAS} main.py \
            --k_shot ${k} \
            --test_langs es nl de \
            --lambda_max_loss 0.0 \
            --max_ft_steps 10 \
            --lr_finetune 1e-5 \
            --model_dir models/baseModel-seed_${seed} \
            --gpu_device ${GPUNO} \
            --seed ${seed} \
            --py_alias ${PY_ALIAS}
    done
done

# proposed approach

# mask_rate=-1 => no masking scheme
MASK_RATE=0.2
# lambda_maxloss=0.0 => no max-loss
LAMBDA_MAXLOSS=2.0

for seed in ${SEEDS[@]}; do

    # == >train-ours
    ${PY_ALIAS} main.py \
        --inner_steps 2 \
        --mask_rate ${MASK_RATE} \
        --lambda_max_loss ${LAMBDA_MAXLOSS} \
        --result_dir meta-innerSteps_2-maskRate_${MASK_RATE}-lambdaMaxLoss_${LAMBDA_MAXLOSS}-seed_${seed} \
        --gpu_device ${GPUNO} \
        --seed ${seed} \
        --py_alias ${PY_ALIAS}

    # == >0-shot-ours
    ${PY_ALIAS} main.py \
        --zero_shot \
        --max_ft_steps 1 \
        --test_langs es nl de \
        --lambda_max_loss 0.0 \
        --support_size 2 \
        --lr_finetune 1e-5 \
        --model_dir models/meta-innerSteps_2-maskRate_${MASK_RATE}-lambdaMaxLoss_${LAMBDA_MAXLOSS}-seed_${seed} \
        --gpu_device ${GPUNO} \
        --seed ${seed} \
        --py_alias ${PY_ALIAS}

    for k in ${KSHOT[@]}; do

        # == >k-shot-ours
        ${PY_ALIAS} main.py \
            --k_shot ${k} \
            --test_langs es nl de \
            --lambda_max_loss 0.0 \
            --max_ft_steps 10 \
            --lr_finetune 1e-5 \
            --model_dir models/meta-innerSteps_2-maskRate_${MASK_RATE}-lambdaMaxLoss_${LAMBDA_MAXLOSS}-seed_${seed} \
            --gpu_device ${GPUNO} \
            --seed ${seed} \
            --py_alias ${PY_ALIAS}
    done
done
