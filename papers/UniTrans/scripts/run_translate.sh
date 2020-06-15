#!/bin/bash
GPU_ID=0
SRC_LANG=en
TGT_LANGS=de
EMB_PATH=data/embedding
DICT_PATH=data/dict
DATA_DIR=data/ner/conll

for t in ${TGT_LANGS[@]}; do
    python3 MUSE/supervised.py \
        --src_lang ${SRC_LANG} \
        --tgt_lang ${t} \
        --src_emb ${EMB_PATH}/wiki.${SRC_LANG}.vec \
        --tgt_emb ${EMB_PATH}/wiki.${t}.vec \
        --n_refinement 3 \
        --dico_train identical_char \
        --max_vocab 100000 \
        --dico_eval ${DICT_PATH}/${SRC_LANG}-${t}.5000-6500.txt \
        --exp_path ${DATA_DIR} \
        --exp_name en2${t} \
        --exp_id muse \
        --export ""

    python3 translate.py --tgt_lang=${t} --embed_dir ${EMB_PATH} --gpu_id=${GPU_ID} --data_dir=${DATA_DIR}
done
