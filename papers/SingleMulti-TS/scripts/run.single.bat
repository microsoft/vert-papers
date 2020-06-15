set GPU_IDS=0
set TGT_LANG=es
set SRC_LANG=en
set SEED=22

for %%e in (%SEED%) do (

rem STEP1: train teacher model (English: en)

python main_single.py --do_train --gpu_ids %GPU_IDS% --seed %%e --learning_rate 5e-5 ^
--data_dir .\data\ner\conll\%SRC_LANG% ^
--output_dir conll-model-%%e\mono-src-%SRC_LANG%

rem STEP2: single-source teacher-student learning

python main_single.py --do_train --do_KD --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir .\data\ner\conll\%TGT_LANG% ^
--src_langs %SRC_LANG% --src_model_dir_prefix mono-src- --src_model_dir conll-model-%%e ^
--output_dir conll-model-%%e\ts-%SRC_LANG%-%TGT_LANG%

python main_single.py --do_predict --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir .\data\ner\conll\%TGT_LANG% ^
--output_dir conll-model-%%e\ts-%SRC_LANG%-%TGT_LANG%
)