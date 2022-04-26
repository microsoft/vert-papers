# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

exp_name=$1
round=$2
echo "Evaluating Round ${round} ..."
submission_dir="${BASE_DATA_DIR}/SemTab/Submission/${exp_name}"
echo "Evaluating SemTab ..."
echo "---------------${submission_dir}---------------"
python Evaluator/Evaluator_2020/CEA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CEA"
python Evaluator/Evaluator_2020/CTA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CTA"
python Evaluator/Evaluator_2020/CPA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CPA"
if [ $round == 4 ]
then
  echo "Evaluate 2T ..."
  python Evaluator/Evaluator_2020_2T/evaluator.py --cea_submission_file "${submission_dir}/CEA/CEA.csv" --cta_submission_file "${submission_dir}/CTA/CTA.csv"
fi