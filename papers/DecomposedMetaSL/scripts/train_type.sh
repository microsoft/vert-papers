seed=${1:-12}
d=${2:-1}
K=${3:-1}
dataset=${4:-"ner"}
ft_steps=${5:-20}

base_dir=data
batch_size=16


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



name=type_mamlcls${ft_steps}_bz${batch_size}
inner_steps=1

output_dir=outputs/${dataset}/${d}-${K}shot/${name}/seed${seed}
mkdir -p ${output_dir}
log_name=${dataset}-${d}-${K}-${name}-${seed}-train.log

python train_type.py --mode train \
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
    --N ${d} \
    --K ${K} \
    --Q 1 \
    --bio ${bio} \
    --encoder_name_or_path bert-base-uncased \
    --last_n_layer -4 \
    --max_length 128 \
    --word_encode_choice first \
    --span_encode_choice avg \
    --learning_rate 0.001 \
    --bert_learning_rate 5e-5 \
    --bert_weight_decay 0.01 \
    --train_iter ${train_iter} \
    --dev_iter ${dev_iter} \
    --val_steps ${val_steps} \
    --warmup_step 0 \
    --log_steps 50 \
    --train_batch_size ${batch_size} \
    --eval_batch_size 1 \
    --use_width False \
    --use_case False \
    --dropout 0.5 \
    --max_grad_norm 5 \
    --dot False \
    --normalize l2 \
    --temperature 0.1 \
    --use_focal False \
    --use_att False \
    --att_hidden_dim 100 \
    --use_oproto False \
    --hou_eval_ep ${hou_eval_ep} \
    --overlap False \
    --type_threshold 0 \
    --use_maml True \
    --train_inner_lr 2e-5 \
    --train_inner_steps ${inner_steps} \
    --warmup_prop_inner 0 \
    --eval_inner_lr 2e-5 \
    --eval_inner_steps ${ft_steps} | tee ${log_name}

mv ${log_name} ${output_dir}/train.log