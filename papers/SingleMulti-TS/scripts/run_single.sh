#!/bin/bash

GPU_IDS=0
TGT_LANG=es
SRC_LANG=en
SEEDS=(22)

for seed in ${SEEDS[@]}; do

    # STEP1: train teacher model (English: en)

    python main_single.py \
        --do_train \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --learning_rate 5e-5 \
        --data_dir ./data/ner/conll/${SRC_LANG} \
        --output_dir conll-model-${seed}/mono-src-${SRC_LANG}

    # STEP2: single-source teacher-student learning

    python main_single.py \
        --do_train \
        --do_KD \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --data_dir ./data/ner/conll/${TGT_LANG} \
        --src_langs ${SRC_LANG} \
        --src_model_dir_prefix mono-src- \
        --src_model_dir conll-model-${seed} \
        --output_dir conll-model-${seed}/ts-${SRC_LANG}-${TGT_LANG}

    python main_single.py \
        --do_predict \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --data_dir ./data/ner/conll/${TGT_LANG} \
        --output_dir conll-model-${seed}/ts-${SRC_LANG}-${TGT_LANG}
done
