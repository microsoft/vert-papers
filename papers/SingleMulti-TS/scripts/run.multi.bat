set GPU_IDS=0

set SEED=22
set GAMMA=0.01
set LOW_RANK=64
set LANGS=en es nl de
set TGT_LANG=en
set MAX_EPOCH=10

rem STEP1: train teacher models for each source language, as in `run.single.bat`

for %%e in (%SEED%) do (
for %%l in (%LANGS%) do (
python main_single.py --do_train --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir .\data\ner\conll\%%l ^
--output_dir conll-model-%%e\mono-src-%%l
)
)

for %%e in (%SEED%) do (
for %%r in (%LOW_RANK%) do (
for %%g in (%GAMMA%) do (

rem STEP2: train domain model

python domain_learner.py --do_train --gpu_ids %GPU_IDS% --seed %%e ^
--data_dir .\data\ner\conll --src_langs %LANGS% --tgt_lang %TGT_LANG% ^
--gamma_R %%g --low_rank_size %%r --tau_metric var --num_train_epochs %MAX_EPOCH% ^
--output_dir domain-model\%TGT_LANG%-rank_%%r-gamma_%%g-seed_%%e

rem STEP3: multi-source teacher-student learning

python main.py --do_train --gpu_ids %GPU_IDS% --seed %%e ^
--tgt_lang %TGT_LANG% --src_langs %LANGS% --src_model_dir conll-model-%%e ^
--sim_dir domain-model --low_rank_size %%r --gamma_R %%g --sim_level domain --tau_metric var ^
--sim_type learn ^
--output_dir result-%%e\ts-learn-var-domain-%TGT_LANG%-rank_%%r-gamma_%%g

python main.py --do_predict --gpu_ids %GPU_IDS% --seed %%e ^
--tgt_lang %TGT_LANG% --low_rank_size %%r --gamma_R %%g --sim_level domain --tau_metric var ^
--sim_type learn ^
--output_dir result-%%e\ts-learn-var-domain-%TGT_LANG%-rank_%%r-gamma_%%g

)
)
)