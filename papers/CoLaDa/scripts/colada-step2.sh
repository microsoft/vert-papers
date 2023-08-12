# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
exp_dir=./exp
data_dir=./data

SEED=$1
tgt=$2
cur_iter=$3

save_ckpt=True

# data file config
if [ $tgt == 'ar' ] || [ $tgt == 'zh' ] || [ $tgt == 'hi' ]
then
data=wikiann
trans_sym=m2m100
else
data=conll
trans_sym=word
fi
src=en
train_file=${data_dir}/${data}-lingual/${src}/dup-train.txt
src_dev_file=${data_dir}/${data}-lingual/${tgt}/dev.txt
unlabel_file=${data_dir}/${data}-lingual/${tgt}/dup-train.txt
test_file=${data_dir}/${data}-lingual/${tgt}/test.txt
trans_file=${data_dir}/${data}-lingual/${src}/${trans_sym}-trans/dup-trans-train.${tgt}.conll


mode=step2
do_train=True
wandb_log=False

# ner parameter
use_crf=True  #just apply viterbi decoding
# mname=xlm-large
mname=bert
T=0
train_ner_epochs=10
ner_lr=2e-5
ner_drop=0.1
ner_max_grad_norm=1
frz_bert_layers=3
if [ $mname == 'bert' ]
then
    model_name_or_path=bert-base-multilingual-cased
elif [ $mname == 'xlm-large' ]
then
    model_name_or_path=xlm-roberta-large
else
    model_name_or_path=xlm-roberta-base
fi


#knn combine parameter
filter_tgt=reweight
knn_pooling=avg
knn_lid=3
ealpha=6
emu=1
K=500
knnname=${filter_tgt}${K}-m${emu}-a${ealpha}-l${knn_lid}

# set ckptdir and outputdir
if [ $cur_iter == '1' ]
then
ckptname=it${cur_iter}-s1-${trans_sym}-lam0.5-${knnname}-${mname}
name=it${cur_iter}-s2-${trans_sym}-lam0.5-T${T}-${knnname}-${mname}
else
ckptname=it${cur_iter}-s1-${trans_sym}-lam0.5-T${T}-lam0.1-${knnname}-${mname}
name=it${cur_iter}-s2-${trans_sym}-lam0.5-T${T}-lam0.1-${knnname}-${mname}
fi
ckpt_dir=${exp_dir}/${data}/${src}-${tgt}/${ckptname}/seed${SEED}
out_dir=${exp_dir}/${data}/${src}-${tgt}/${name}/seed${SEED}


mkdir -p ${out_dir}

python train.py --mode ${mode} \
    --train_file ${train_file} \
    --trans_file ${trans_file} \
    --src_dev_file ${src_dev_file}  \
    --unlabel_file ${unlabel_file} \
    --test_file ${test_file} \
    --do_train ${do_train} --do_eval True \
    --ckpt_dir ${ckpt_dir} \
    --output_dir ${out_dir} \
    --model_name_or_path ${model_name_or_path} \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --frz_bert_layers ${frz_bert_layers} \
    --ner_max_grad_norm ${ner_max_grad_norm} \
    --train_ner_epochs ${train_ner_epochs} \
    --use_crf ${use_crf} \
    --ner_lr ${ner_lr} \
    --ner_drop ${ner_drop} \
    --T ${T} \
    --select_ckpt True \
    --save_ckpt ${save_ckpt} \
    --filter_tgt ${filter_tgt} \
    --knn_lid ${knn_lid} \
    --knn_pooling ${knn_pooling} \
    --K ${K} \
    --ealpha ${ealpha} \
    --emu ${emu} \
    --seed ${SEED} \
    --save_log ${wandb_log} \
    --logging_steps 20 \
    --val_steps -1 | tee ${out_dir}/log.txt