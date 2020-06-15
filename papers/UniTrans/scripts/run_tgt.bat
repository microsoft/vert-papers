set GPU_IDS=0
set TGT_LANG=de
set ENCODING=UTF-8
set PRED_MODS=test dev
set SEEDS=122 649 705 854 975
set DATA_DIR=data/ner/conll
set OUT_NAME=result
set LABEL_PATH=data/ner/conll/labels.txt

: for majority voting based pseudo hard labels
for %%e in (%SEEDS%) do (

: M trans
python main.py --do_train --gpu_ids %GPU_IDS% --seed %%e --learning_rate 5e-5 ^
--data_dir %DATA_DIR%\en2%TGT_LANG% ^
--output_dir %OUT_NAME%\result-%%e\mBERTtrans-XLData-en2%TGT_LANG% ^
--encoding %ENCODING% ^
--labels %LABEL_PATH%

for %%m in (%PRED_MODS%) do (
python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e --use_viterbi ^
--data_dir .\%DATA_DIR%\%TGT_LANG% ^
--output_dir %OUT_NAME%\result-%%e\mBERTtrans-XLData-en2%TGT_LANG% ^
--encoding %ENCODING% ^
--mode %%m ^
--labels %LABEL_PATH%
)

: M finetune: finetune Msrc
python main.py --do_finetune --gpu_ids %GPU_IDS% --seed %%e --learning_rate 5e-5 ^
--data_dir .\%DATA_DIR%\en2%TGT_LANG% ^
--src_model_path %OUT_NAME%\result-%%e\mBERT-en ^
--output_dir %OUT_NAME%\result-%%e\mBERT_finetune_XLData_en2%TGT_LANG% ^
--encoding %ENCODING% ^
--labels %LABEL_PATH%

for %%m in (%PRED_MODS%) do (
python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e --use_viterbi ^
--data_dir .\%DATA_DIR%\%TGT_LANG% ^
--output_dir %OUT_NAME%\result-%%e\mBERT_finetune_XLData_en2%TGT_LANG% ^
--encoding %ENCODING% ^
--mode %%m ^
--labels %LABEL_PATH%
)

: filtered hard labels with pfinetune
: for filtered hard labels only, pls set lambda_original_loss to -1.0.
python main.py --do_train --use_KD --do_filter_token --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir %DATA_DIR%\%TGT_LANG% ^
--src_model_path %OUT_NAME%\result-%%e\mBERT-en ^
--src_model_path_assist %OUT_NAME%\result-%%e\mBERT_finetune_XLData_en2%TGT_LANG% %OUT_NAME%\result-%%e\mBERTtrans-XLData-en2%TGT_LANG% ^
--lambda_original_loss 1.0 --loss_with_crossEntropy ^
--output_dir %OUT_NAME%\result-%%e\UniTran-mBERT_%TGT_LANG%_1.0-lossWithCE ^
--encoding %ENCODING% ^
--labels %LABEL_PATH%

for %%m in (%PRED_MODS%) do (
python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e --use_viterbi ^
--data_dir .\%DATA_DIR%\%TGT_LANG% ^
--output_dir %OUT_NAME%\result-%%e\UniTran-mBERT_%TGT_LANG%_1.0-lossWithCE ^
--encoding %ENCODING% ^
--mode %%m ^
--labels %LABEL_PATH%
)

: filtered hard labels with pfinetune
: for filtered hard labels only, pls set lambda_original_loss to -1.0.
python main.py --do_train --use_KD --do_filter_token --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir %DATA_DIR%\%TGT_LANG% ^
--src_model_path %OUT_NAME%\result-%%e\mBERT_finetune_XLData_en2%TGT_LANG% ^
--src_model_path_assist %OUT_NAME%\result-%%e\mBERT-en %OUT_NAME%\result-%%e\mBERTtrans-XLData-en2%TGT_LANG% ^
--lambda_original_loss 1.0 --loss_with_crossEntropy ^
--output_dir %OUT_NAME%\result-%%e\UniTran-fineTran_%TGT_LANG%_1.0-lossWithCE ^
--encoding %ENCODING% ^
--labels %LABEL_PATH%

for %%m in (%PRED_MODS%) do (
python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e --use_viterbi ^
--data_dir .\%DATA_DIR%\%TGT_LANG% ^
--output_dir %OUT_NAME%\result-%%e\UniTran-fineTran_%TGT_LANG%_1.0-lossWithCE ^
--encoding %ENCODING% ^
--mode %%m ^
--labels %LABEL_PATH%
)
)

: statistical results 
python scripts\statistical.py --src_dir=%OUT_NAME%\result-122\mBERT-en-%TGT_LANG% ^
--trans_dir=%OUT_NAME%\result-122\mBERTtrans-XLData-en2%TGT_LANG% ^
--finetune_dir=%OUT_NAME%\result-122\mBERT_finetune_XLData_en2%TGT_LANG% ^
--unitrans_finetune_dir=%OUT_NAME%\result-122\UniTran-fineTran_%TGT_LANG%_1.0-lossWithCE ^
--unitrans_src_dir=%OUT_NAME%\result-122\UniTran-mBERT_%TGT_LANG%_1.0-lossWithCE ^
--seeds=122,649,705,854,975
