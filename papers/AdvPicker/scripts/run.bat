set GPU_ID=0
set SEEDS=320 550 631 691 985
set TGTS=de es nl
set LR=5e-3
set DH=500

set PYTHON=python3

for %%e in (%SEEDS%) do (
: Step 1: Train mBERT-TLADV
%PYTHON% train_wl_db.py ^
--seed=%%e ^
--disc_hidden_size=%DH% ^
--gpu_id=%GPU_ID% ^
--lr_lm=6e-5 ^
--lr_d=%LR% ^
--lr_gen=6e-7 ^
--batch_size=32 ^
--eval_batch_size=32 ^
--num_epoches=10

: Step 2: Predict Langauge-Discriminator scores
for %%t in (%TGTS%) do (
%PYTHON% get_xl_data.py ^
--seed=%%e ^
--gpu_id=%GPU_ID% ^
--batch_size=32 ^
--pth_dir=result/%DH%-0.005-%%e-es_de_nl ^
--tgt_lang=%%t ^
--train_type=train  
)
)

: Step 3: Ensemble Data Selections
%PYTHON% overlap_all.py ^
--seed=%SEEDS: =,% ^
--gpu_id=%GPU_ID% ^
--tgt_langs=%TGTS: =,%

: Step 4: Knowledge Distillation
for %%e in (%SEEDS%) do (
for %%t in (%TGTS%) do (
%PYTHON% kd.py ^
--seed=%%e ^
--gpu_id=%GPU_ID% ^
--batch_size=32 ^
--tgt_lang=%%t ^
--eval_langs=%%t
)
)