# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

for dname in "Semtab21_round2_hard_table" "Semtab21_round2_biotab_table" "Semtab21_round3_hard_table" "Semtab21_round3_biodivtab_table"
do
python TableAnnotator/multi_process_offline_predict.py --in_table_fn "${BASE_DATA_DIR}/SemTab/Input/${dname}.json" --dump_dir "${BASE_DATA_DIR}/SemTab/Output/${dname}-output" --index_name "wikidata_20210830_keep_disambiguation" --in_links_fn "${BASE_DATA_DIR}/SemTab21-Data/wikidata/wikidata-20210830/incoming_links/in_coming_links_num.pkl" --alias_map_fn "${BASE_DATA_DIR}/SemTab21-Data/wikidata/wikidata-20210830/merged_alias_map/alias_map_keep_disambiguation.pkl"
if [ ${dname} == "Semtab21_round2_hard_table" ]
then
  python Benchmark/eval/gen_semtab21_round2_hardtable_submission.py --data-dir "${BASE_DATA_DIR}/SemTab21-Data/SemTab/Benchmark/Round2/HardTable/targets" --log_fn "${BASE_DATA_DIR}/SemTab/Output/${dname}-output/all_output_tables.jsonl" --result_dir "${BASE_DATA_DIR}/SemTab/Submission"
elif [ ${dname} == "Semtab21_round2_biotab_table" ]
then
  python Benchmark/eval/gen_semtab21_biotab_submission.py --data-dir "${BASE_DATA_DIR}/SemTab21-Data/SemTab/Benchmark/Round2/BioTable/targets" --log_fn "${BASE_DATA_DIR}/SemTab/Output/${dname}-output/all_output_tables.jsonl" --result_dir "${BASE_DATA_DIR}/SemTab/Submission"
elif [ ${dname} == "Semtab21_round3_hard_table" ]
then
  python Benchmark/eval/gen_semtab21_round3_hardtable_submission.py --data-dir "${BASE_DATA_DIR}/SemTab21-Data/SemTab/Benchmark/Round3/HardTable/targets" --log_fn "${BASE_DATA_DIR}/SemTab/Output/${dname}-output/all_output_tables.jsonl" --result_dir "${BASE_DATA_DIR}/SemTab/Submission"
elif [ ${dname} == "Semtab21_round3_biodivtab_table" ]
then
  # CTA - match with header
  python Benchmark/eval/gen_semtab21_biodivtab_header_submission.py --data-dir "${BASE_DATA_DIR}/SemTab21-Data/SemTab/Benchmark/Round3/BioDivTab/targets" --log_fn "${BASE_DATA_DIR}/SemTab/Output/${dname}-output/all_output_tables.jsonl" --result_dir "${BASE_DATA_DIR}/SemTab/Submission"
fi
done