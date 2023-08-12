# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exp_dir=./exp
data_dir=./data

SEED=42

mode=ner
do_train=True
wandb_log=False

dataset=conll
src=en
tgt=de
train_file=${data_dir}/${dataset}-lingual/${src}/dup-train.txt
src_dev_file=${data_dir}/${dataset}-lingual/${tgt}/dev.txt
unlabel_file=${data_dir}/${dataset}-lingual/${tgt}/dup-train.txt 
test_file=${data_dir}/${dataset}-lingual/${tgt}/test.txt


# ner learning params
mname=bert
use_crf=True
name=src-${mname}
if [ $mname == 'bert' ]
then
    model_name_or_path=bert-base-multilingual-cased
elif [ $mname == 'xlm-large' ]
then
    model_name_or_path=xlm-roberta-large
else
    model_name_or_path=xlm-roberta-base
fi
train_ner_epochs=3
ner_lr=5e-5
ner_drop=0.1
ner_max_grad_norm=1
frz_bert_layers=3
out_dir=${exp_dir}/${dataset}/${src}/${name}/seed${SEED}


mkdir -p ${out_dir}

python train.py --mode ${mode} \
    --train_file ${train_file} \
    --src_dev_file ${src_dev_file}  \
    --unlabel_file ${unlabel_file} \
    --test_file ${test_file} \
    --do_train ${do_train} --do_eval True \
    --output_dir ${out_dir} \
    --save_ckpt True \
    --val_steps -1 \
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
    --save_log ${wandb_log} \
    --logging_steps 20 \
    --seed ${SEED} | tee ${out_dir}/log.txt
