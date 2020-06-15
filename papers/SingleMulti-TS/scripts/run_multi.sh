#!/bin/bash

GPU_IDS=0
SEEDS=(22)
GAMMAS=(0.01)
LOW_RANKS=(64)
LANGS=(en es nl de)
TGT_LANG=en
MAX_EPOCH=10

# STEP1: train teacher models for each source language, as in `run_single.sh`
for seed in ${SEEDS[@]}; do
    for lan in ${LANGS[@]}; do
        python main_single.py \
            --do_train \
            --gpu_ids ${GPU_IDS} \
            --seed ${seed} \
            --data_dir ./data/ner/conll/${lan} \
            --output_dir conll-model-${seed}/mono-src-${lan}
    done

    for rank in ${LOW_RANK[@]}; do
        for g in ${GAMMA[@]}; do

            # STEP2: train domain model
            python domain_learner.py \
                --do_train \
                --gpu_ids ${GPU_IDS} \
                --seed ${seed} \
                --data_dir ./data/ner/conll \
                --src_langs ${LANGS} \
                --tgt_lang ${TGT_LANG} \
                --gamma_R ${g} \
                --low_rank_size ${rank} \
                --tau_metric var \
                --num_train_epochs %MAX_EPOCH% \
                --output_dir domain-model/${TGT_LANG}-rank_${rank}-gamma_${g}-seed_${seed}

            # STEP3: multi-source teacher-student learning
            python main.py \
                --do_train \
                --gpu_ids ${GPU_IDS} \
                --seed ${seed} \
                --tgt_lang ${TGT_LANG} \
                --src_langs ${LANGS} \
                --src_model_dir conll-model-${seed} \
                --sim_dir domain-model \
                --low_rank_size ${rank} \
                --gamma_R ${g} \
                --sim_level domain \
                --tau_metric var \
                --sim_type learn \
                --output_dir result-${seed}/ts-learn-var-domain-${TGT_LANG}-rank_${rank}-gamma_${g}

            python main.py \
                --do_predict \
                --gpu_ids ${GPU_IDS} \
                --seed ${seed} \
                --tgt_lang ${TGT_LANG} \
                --low_rank_size ${rank} \
                --gamma_R ${g} \
                --sim_level domain \
                --tau_metric var \
                --sim_type learn \
                --output_dir result-${seed}/ts-learn-var-domain-${TGT_LANG}-rank_${rank}-gamma_${g}
        done
    done
done
