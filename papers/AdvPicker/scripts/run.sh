#!/bin/bash
GPU_ID=0
SEEDS=(320 550 631 691 985)
TGTS=(de es nl)
LR=5e-3
DH=500
TGTS_STR=`echo $TGTS[@] | sed "s/ /,/g"`
SEEDS_STR=`echo $SEEDS[@] | sed "s/ /,/g"`

PYTHON=python3

for seed in ${SEEDS[@]}; do
    # Step 1: Train mBERT-TLADV
    ${PYTHON} train_wl_db.py \
        --seed ${seed} \
        --disc_hidden_size ${DH} \
        --gpu_id ${GPU_ID} \
        --lr_lm 6e-5 \
        --lr_d ${LR} \
        --lr_gen 6e-7 \
        --batch_size 32 \
        --eval_batch_size 32 \
        --num_epoches 10

    # Step 2: Predict Langauge-Discriminator scores
    for tgt in ${TGTS[@]}; do
        ${PYTHON} get_xl_data.py \
            --seed ${seed} \
            --gpu_id ${GPU_ID} \
            --batch_size 32 \
            --pth_dir result/${DH}-0.005-${seed}-es_de_nl \
            --tgt_lang ${tgt} \
            --train_type train
    done   
done

# Step 3: Ensemble Data Selections
${PYTHON} overlap_all.py \
    --seed ${SEEDS_STR} \
    --gpu_id ${GPU_ID} \
    --tgt_langs ${TGTS_STR}


# Step 4: Knowledge Distillation
for seed in ${SEEDS[@]}; do
    for tgt in ${TGTS[@]}; do
        ${PYTHON} kd.py \
            --seed ${seed} \
            --gpu_id ${GPU_ID} \
            --batch_size 32 \
            --tgt_lang ${tgt} \
            --eval_langs ${tgt}
    done
done
