# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
tgt=de
gid=0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${gid}
export TOKENIZERS_PARALLELISM=false


slf_iter=8

for SEED in 122 42 666 22 705;
do

for (( i=1; i<=$slf_iter; i++ ))
do
    bash scripts/colada-step1.sh ${SEED} ${tgt} ${i}
    bash scripts/colada-step2.sh ${SEED} ${tgt} ${i}
done

done