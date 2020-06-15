#!/bin/bash

GPU_IDS=0
TGT_LANGS=(de)
ENCODING=UTF-8
SEEDS=(122 649 705 854 975)
DATA_DIR=data/ner/conll
OUT_NAME=result
LABEL_PATH=data/ner/conll/labels.txt
PRED_MODS=(test dev)

for seed in ${SEEDS[@]}; do
    # M_src
    python3 main.py \
        --do_train \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --learning_rate 5e-5 \
        --data_dir ${DATA_DIR}/en \
        --output_dir ${OUT_NAME}/result-${seed}/mBERT-en \
        --labels ${LABEL_PATH}

    for t in ${TGT_LANGS[@]}; do
        for m in ${PRED_MODS[@]}; do
            python3 main.py \
                --do_predict \
                --gpu_ids ${GPU_IDS} \
                --seed ${seed} \
                --use_viterbi \
                --data_dir ./${DATA_DIR}/${t} \
                --src_model_path ${OUT_NAME}/result-${seed}/mBERT-en \
                --output_dir ${OUT_NAME}/result-${seed}/mBERT-en-${t} \
                --mode ${m} \
                --labels ${LABEL_PATH}
        done
    done

done
