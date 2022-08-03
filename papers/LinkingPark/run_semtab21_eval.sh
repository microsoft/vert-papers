# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

submission_dir="${BASE_DATA_DIR}/SemTab21-Data/SemTab/Submission"
benchmark_dir="${BASE_DATA_DIR}/SemTab21-Data/SemTab/Benchmark"

echo "Evaluating Round2 HardTables ..."
echo "CEA ..."
python Evaluator/Evaluator_2021/HardTable_Round2_CEA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/HardTable/gt/HardTablesR2_CEA_WD_gt.csv" --submission-fn "${submission_dir}/Round2_HardTable/CEA/CEA.csv"

echo "CTA ..."
python Evaluator/Evaluator_2021/HardTable_Round2_CTA_WD_Evaluator.py --gt-dir "${benchmark_dir}/Round2/HardTable/gt/" --submission-fn "${submission_dir}/Round2_HardTable/CTA/CTA.csv"

echo "CPA ..."
python Evaluator/Evaluator_2021/HardTable_Round2_CPA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/HardTable/gt/HardTablesR2_CPA_WD_gt.csv" --submission-fn "${submission_dir}/Round2_HardTable/CPA/CPA.csv"

echo "Evaluating Round2 BioTables ..."
echo "CEA ..."
python Evaluator/Evaluator_2021/BioTable_CEA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/BioTable/gt/cea_gt.csv" --submission-fn "${submission_dir}/Round2_BioTab/CEA/CEA.csv"

echo "CTA ..."
echo "Due to missing released resources, please use the online aicrowd evaluator (https://www.aicrowd.com/challenges/semtab-2021/problems/biotable-column-type-annotation-by-wikidata-biotable-cta-wd/submissions) to evaluate the CTA task"

echo "CPA ..."
python Evaluator/Evaluator_2021/BioTable_CPA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/BioTable/gt/cpa_gt.csv" --submission-fn "${submission_dir}/Round2_BioTab/CPA/CPA.csv"

echo "Evaluating Round3 BioDivTable ..."
echo "CEA ..."
python Evaluator/Evaluator_2021/BioDivTable_CEA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round3/BioDivTab/gt/CEA_biodivtab_2021_gt.csv" --submission-fn "${submission_dir}/Round3_BioDivTable_header/CEA/CEA.csv"

echo "CTA ..."
python Evaluator/Evaluator_2021/BioDivTable_CTA_WD_Evaluator.py --gt-dir "${benchmark_dir}/Round3/BioDivTab/gt" --submission-fn "${submission_dir}/Round3_BioDivTable_header/CTA/CTA.csv"

echo "Evaluating Round3 HardTables ..."
echo "CEA ..."
python Evaluator/Evaluator_2021/HardTable_Round3_CEA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round3/HardTable/gt/HardTablesR3_CEA_WD_gt.csv" --submission-fn "${submission_dir}/Round3_HardTable/CEA/CEA.csv"

echo "CTA ..."
python Evaluator/Evaluator_2021/HardTable_Round3_CTA_WD_Evaluator.py --gt-dir "${benchmark_dir}/Round3/HardTable/gt/" --submission-fn "${submission_dir}/Round3_HardTable/CTA/CTA.csv"

echo "CPA ..."
python Evaluator/Evaluator_2021/HardTable_Round3_CPA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round3/HardTable/gt/HardTablesR3_CPA_WD_gt.csv" --submission-fn "${submission_dir}/Round3_HardTable/CPA/CPA.csv"

echo "Evaluating multi-subject extension version Round2 BioTables ..."
echo "CEA ..."
python Evaluator/Evaluator_2021/BioTable_CEA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/BioTable/gt/cea_gt.csv" --submission-fn "${submission_dir}/Round2_BioTab_mul_subj/CEA/CEA.csv"

echo "CTA ..."
echo "Due to missing released resources, please use the online aicrowd evaluator (https://www.aicrowd.com/challenges/semtab-2021/problems/biotable-column-type-annotation-by-wikidata-biotable-cta-wd/submissions) to evaluate the CTA task"

echo "CPA ..."
python Evaluator/Evaluator_2021/BioTable_CPA_WD_Evaluator.py --gt-fn "${benchmark_dir}/Round2/BioTable/gt/cpa_gt.csv" --submission-fn "${submission_dir}/Round2_BioTab_mul_subj/CPA/CPA.csv"