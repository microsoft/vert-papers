# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

exp_name=$1
round=$2
dump_dir="${BASE_DATA_DIR}/SemTab/Output/"
echo "running ${exp_name} for Round ${round}..."
python TableAnnotator/multi_process_offline_predict.py --dump_dir ${dump_dir}${exp_name}
bash eval.sh ${exp_name} ${round}