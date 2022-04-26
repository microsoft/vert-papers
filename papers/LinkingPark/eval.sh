# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

exp_name=$1
round=$2
echo "Evaluating Round ${round}..."
log_fn="${BASE_DATA_DIR}/SemTab/Output/${exp_name}/all_output_tables.jsonl"
echo "generate submission files..."
python Benchmark/eval/gen_submission_file.py --log_fn ${log_fn} --round ${round}
submission_dir="${BASE_DATA_DIR}/SemTab/Submission/${exp_name}"
echo "evaluating SemTab..."
echo "---------------${submission_dir}---------------"
python Evaluator/Evaluator_2020/CEA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CEA"
python Evaluator/Evaluator_2020/CTA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CTA"
python Evaluator/Evaluator_2020/CPA_Evaluator.py --round ${round} --data_dir "${submission_dir}/CPA"
if [ $round == 4 ]
then
  echo "evaluate 2T..."
  python Evaluator/Evaluator_2020_2T/evaluator.py --cea_submission_file "${submission_dir}/CEA/CEA.csv" --cta_submission_file "${submission_dir}/CTA/CTA.csv"
  #generate analysis file
  echo "generate 2T analysis file"
  python Benchmark/analysis/gen_2T_analysis_file_with_gold.py --log-fn ${log_fn} --round "2T"
fi
echo "generate round${round} analysis file"
python Benchmark/analysis/gen_analysis_file_with_gold.py --log-fn ${log_fn} --round ${round}