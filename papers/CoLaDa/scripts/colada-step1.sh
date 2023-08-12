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


mode=step1
do_train=True
wandb_log=False

# ner parameter
use_crf=True
# mname=xlm-large
mname=bert
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
filter_trans=reweight
knn_lid=3
ealpha=6
emu=1
K=500
knn_pooling=avg
knnname=${filter_trans}${K}-m${emu}-a${ealpha}-l${knn_lid}

if [ $cur_iter == '1' ]
then
trans_lam=0.5
ckptname=src-${mname}
name=it1-s1-${trans_sym}-lam${trans_lam}-${knnname}-${mname}
elif [ $cur_iter == '2' ]
then
prev_iter=`expr $cur_iter - 1`
trans_lam=0.1
ckptname=it${prev_iter}-s2-${trans_sym}-lam0.5-T0-${knnname}-${mname}
name=it${cur_iter}-s1-${trans_sym}-lam0.5-T0-lam${trans_lam}-${knnname}-${mname}
else
prev_iter=`expr $cur_iter - 1`
trans_lam=0.1
ckptname=it${prev_iter}-s2-${trans_sym}-lam0.5-T0-lam${trans_lam}-${knnname}-${mname}
name=it${cur_iter}-s1-${trans_sym}-lam0.5-T0-lam${trans_lam}-${knnname}-${mname}
fi



echo "exp $src-$tgt: $name"


if [ $ckptname == src-${mname} ]
then
ckpt_dir=${exp_dir}/${data}/${src}/${ckptname}/seed${SEED}
else
ckpt_dir=${exp_dir}/${data}/${src}-${tgt}/${ckptname}/seed${SEED}
fi
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
    --filter_trans ${filter_trans} \
    --K ${K} \
    --knn_pooling ${knn_pooling} \
    --knn_lid ${knn_lid} \
    --ealpha ${ealpha} \
    --emu ${emu} \
    --trans_lam ${trans_lam} \
    --save_log ${wandb_log} \
    --logging_steps 20 \
    --save_ckpt ${save_ckpt} \
    --val_steps -1 \
    --select_ckpt True \
    --seed ${SEED} | tee ${out_dir}/log.txt