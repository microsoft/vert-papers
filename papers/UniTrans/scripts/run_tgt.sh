#!/bin/bash

GPU_IDS=0
TGT_LANG=(de)
ENCODING=UTF-8
SEEDS=(122 649 705 854 975)
DATA_DIR=data/ner/conll
OUT_NAME=result
LABEL_PATH=data/ner/conll/labels.txt
PRED_MODS=(test dev)

for seed in ${SEEDS[@]}; do
    # M_trans
    python3 main.py \
        --do_train \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --learning_rate 5e-5 \
        --data_dir ${DATA_DIR}/en2${TGT_LANG} \
        --output_dir ${OUT_NAME}/result-${seed}/mBERTtrans-XLData-en2${TGT_LANG} \
        --encoding ${ENCODING} \
        --labels ${LABEL_PATH}

    for m in ${PRED_MODS[@]}; do
        python3 main.py \
            --do_predict \
            --gpu_ids ${GPU_IDS} \
            --seed ${seed} \
            --use_viterbi \
            --data_dir ./${DATA_DIR}/${TGT_LANG} \
            --output_dir ${OUT_NAME}/result-${seed}/mBERTtrans-XLData-en2${TGT_LANG} \
            --encoding ${ENCODING} \
            --mode ${m} \
            --labels ${LABEL_PATH}
    done

    # M_finetune: finetune Msrc
    python3 main.py \
        --do_finetune \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --learning_rate 5e-5 \
        --data_dir ./${DATA_DIR}/en2${TGT_LANG} \
        --src_model_path ${OUT_NAME}/result-${seed}/mBERT-en \
        --output_dir ${OUT_NAME}/result-${seed}/mBERT_finetune_XLData_en2${TGT_LANG} \
        --encoding ${ENCODING} \
        --labels ${LABEL_PATH}

    for m in ${PRED_MODS[@]}; do
        python3 main.py \
            --do_predict \
            --gpu_ids ${GPU_IDS} \
            --seed ${seed} \
            --use_viterbi \
            --data_dir ./${DATA_DIR}/${TGT_LANG} \
            --output_dir ${OUT_NAME}/result-${seed}/mBERT_finetune_XLData_en2${TGT_LANG} \
            --encoding ${ENCODING} \
            --mode ${m} \
            --labels ${LABEL_PATH}
    done

    # filtered hard labels with pfinetune
    # for filtered hard labels only, pls lambda_original_loss to -1.0.
    python3 main.py --do_train --use_KD --do_filter_token --gpu_ids ${GPU_IDS} --seed ${seed} \
        --data_dir ${DATA_DIR}/${TGT_LANG} \
        --src_model_path ${OUT_NAME}/result-${seed}/mBERT-en \
        --src_model_path_assist ${OUT_NAME}/result-${seed}/mBERT_finetune_XLData_en2${TGT_LANG} ${OUT_NAME}/result-${seed}/mBERTtrans-XLData-en2${TGT_LANG} \
        --lambda_original_loss 1.0 --loss_with_crossEntropy \
        --output_dir ${OUT_NAME}/result-${seed}/UniTran-mBERT_${TGT_LANG}_1.0-lossWithCE \
        --encoding ${ENCODING} \
        --labels ${LABEL_PATH}

    for m in ${PRED_MODS[@]}; do
        python3 main.py --do_predict --gpu_ids ${GPU_IDS} --seed ${seed} --use_viterbi \
            --data_dir ./${DATA_DIR}/${TGT_LANG} \
            --output_dir ${OUT_NAME}/result-${seed}/UniTran-mBERT_${TGT_LANG}_1.0-lossWithCE \
            --encoding ${ENCODING} \
            --mode ${m} \
            --labels ${LABEL_PATH}
    done

    # filtered hard labels with pfinetune
    # for filtered hard labels only, pls lambda_original_loss to -1.0.
    python3 main.py --do_train --use_KD --do_filter_token --gpu_ids ${GPU_IDS} --seed ${seed} \
        --data_dir ${DATA_DIR}/${TGT_LANG} \
        --src_model_path ${OUT_NAME}/result-${seed}/mBERT_finetune_XLData_en2${TGT_LANG} \
        --src_model_path_assist ${OUT_NAME}/result-${seed}/mBERT-en ${OUT_NAME}/result-${seed}/mBERTtrans-XLData-en2${TGT_LANG} \
        --lambda_original_loss 1.0 --loss_with_crossEntropy \
        --output_dir ${OUT_NAME}/result-${seed}/UniTran-fineTran_${TGT_LANG}_1.0-lossWithCE \
        --encoding ${ENCODING} \
        --labels ${LABEL_PATH}

    for m in ${PRED_MODS[@]}; do
        python3 main.py --do_predict --gpu_ids ${GPU_IDS} --seed ${seed} --use_viterbi \
            --data_dir ./${DATA_DIR}/${TGT_LANG} \
            --output_dir ${OUT_NAME}/result-${seed}/UniTran-fineTran_${TGT_LANG}_1.0-lossWithCE \
            --encoding ${ENCODING} \
            --mode ${m} \
            --labels ${LABEL_PATH}
    done
done

# statistical results
python scripts/statistical.py \
    --src_dir=${OUT_NAME}/result-122/mBERT-en-${TGT_LANG} \
    --trans_dir=${OUT_NAME}/result-122/mBERTtrans-XLData-en2${TGT_LANG} \
    --finetune_dir=${OUT_NAME}/result-122/mBERT_finetune_XLData_en2${TGT_LANG} \
    --unitrans_finetune_dir=${OUT_NAME}/result-122/UniTran-fineTran_${TGT_LANG}_1.0-lossWithCE \
    --unitrans_src_dir=${OUT_NAME}/result-122/UniTran-mBERT_${TGT_LANG}_1.0-lossWithCE \
    --seeds=122,649,705,854,975
