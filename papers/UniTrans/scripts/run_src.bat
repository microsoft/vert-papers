set GPU_IDS=0
set TGT_LANGS=de
set ENCODING=UTF-8
set SEEDS=122 649 705 854 975
set DATA_DIR=data/ner/conll
set OUT_NAME=result
set PRED_MODS=test dev
set LABEL_PATH=data/ner/conll/labels.txt

for %%e in (%SEEDS%) do (

: M src
python main.py --do_train --gpu_ids %GPU_IDS% --seed %%e --learning_rate 5e-5 ^
--data_dir %DATA_DIR%\en ^
--output_dir %OUT_NAME%\result-%%e\mBERT-en ^
--labels %LABEL_PATH%

for %%t in (%TGT_LANGS%) do (
for %%m in (%PRED_MODS%) do (
python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e --use_viterbi ^
--data_dir .\%DATA_DIR%\%%t ^
--src_model_path %OUT_NAME%\result-%%e\mBERT-en ^
--output_dir %OUT_NAME%\result-%%e\mBERT-en-%%t ^
--mode %%m ^
--labels %LABEL_PATH%
)
)
)
