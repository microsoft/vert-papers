seed=${1:-12}
d=${2:-1}
K=${3:-1}
dataset=${4:-"ner"}
schema=${5:-"BIO"}
mloss=${6:-2}
ft_steps=${7:-30}

base_dir=data

wp_inner=0
lr=5e-5
batch_size=16

echo $batch_size

if [ $dataset == 'snips' ] || [ $dataset == "ner" ]
then
root=${base_dir}/${dataset}/${K}shot
train_fn=train_domain${d}.txt
val_fn=valid_domain${d}.txt
test_fn=test_domain${d}.txt
ep_dir=$root
ep_train_fn=train_domain${d}_id.jsonl
ep_val_fn=valid_domain${d}_id.jsonl
ep_test_fn=test_domain${d}_id.jsonl
bio=True
train_iter=800
dev_iter=-1
val_steps=100
elif [ $dataset == 'inter' ] || [ $dataset == "intra" ]
then
root=${base_dir}/few-nerd/${dataset}
ep_dir=${base_dir}/few-nerd/episode-${dataset}
train_fn=train.txt
val_fn=dev.txt
test_fn=test.txt
ep_train_fn=train_${d}_${K}_id.jsonl
ep_val_fn=dev_${d}_${K}_id.jsonl
ep_test_fn=test_${d}_${K}_id.jsonl
bio=False
train_iter=1000
dev_iter=500
val_steps=500
else
root=${base_dir}/fewevent
train_fn=train.txt
val_fn=dev.txt
test_fn=test.txt
ep_dir=${root}
ep_train_fn=train_${d}_${K}_id.jsonl
ep_val_fn=dev_${d}_${K}_id.jsonl
ep_test_fn=test_${d}_${K}_id.jsonl
bio=True
train_iter=1000
dev_iter=-1
val_steps=500
fi

if [ $schema == 'IO' ]
then
crf=False
else
crf=True
fi



name=sqlment_maml${ft_steps}_max${mloss}_${schema}_bz${batch_size}
inner_steps=2

output_dir=outputs/${dataset}/${d}-${K}shot/${name}/seed${seed}
mkdir -p ${output_dir}
log_name=${dataset}-${d}-${K}-${name}-${seed}-train.log

python train_sql.py --mode train \
    --seed ${seed} \
    --root ${root} \
    --train ${train_fn} \
    --val ${val_fn} \
    --test ${test_fn} \
    --ep_dir ${ep_dir} \
    --ep_train ${ep_train_fn} \
    --ep_val ${ep_val_fn} \
    --ep_test ${ep_test_fn} \
    --output_dir ${output_dir} \
    --encoder_name_or_path bert-base-uncased \
    --last_n_layer -1 \
    --max_length 128 \
    --word_encode_choice first \
    --warmup_step 0 \
    --learning_rate 0.001 \
    --bert_learning_rate ${lr} \
    --bert_weight_decay 0.01 \
    --log_steps 50 \
    --train_iter ${train_iter} \
    --dev_iter ${dev_iter} \
    --val_steps ${val_steps} \
    --train_batch_size ${batch_size} \
    --eval_batch_size 1 \
    --max_loss ${mloss} \
    --use_crf ${crf} \
    --schema ${schema} \
    --dropout 0.5 \
    --max_grad_norm 5 \
    --eval_all_after_train True \
    --bio ${bio} \
    --use_maml True \
    --train_inner_lr 2e-5 \
    --train_inner_steps ${inner_steps} \
    --warmup_prop_inner ${wp_inner} \
    --eval_inner_lr 2e-5 \
    --eval_inner_steps ${ft_steps} | tee ${log_name}

mv ${log_name} ${output_dir}/train.log