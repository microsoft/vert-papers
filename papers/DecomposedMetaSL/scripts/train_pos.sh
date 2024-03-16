seed=${1:-12}
d=${2:-2}
K=${3:-1}
mloss=${4:-2}
ft_steps=${5:-20}

base_dir=data
batch_size=16
dataset=postag

root=${base_dir}/postag/${d}-${K}shot
train_fn=train.txt
val_fn=dev.txt
test_fn=test.txt
ep_dir=$root
ep_train_fn=train_id.jsonl
ep_val_fn=dev_id.jsonl
ep_test_fn=test_id.jsonl
train_iter=1000
dev_iter=-1
val_steps=100


name=pos_mamlcls${ft_steps}_max${mloss}_bz${batch_size}
inner_steps=1

output_dir=outputs/${dataset}/${d}-${K}shot/${name}/seed${seed}
mkdir -p ${output_dir}
log_name=${dataset}-${d}-${K}-${name}-${seed}-train.log

python train_pos.py --mode train \
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
    --encoder_name_or_path bert-base-uncased \
    --last_n_layer -1 \
    --max_length 128 \
    --word_encode_choice first \
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
    --dropout 0.5 \
    --max_grad_norm 5 \
    --dot False \
    --normalize l2 \
    --temperature 0.1 \
    --max_loss ${mloss} \
    --use_maml True \
    --train_inner_lr 2e-5 \
    --train_inner_steps ${inner_steps} \
    --warmup_prop_inner 0.1 \
    --eval_inner_lr 2e-5 \
    --eval_inner_steps ${ft_steps}