: Step 1: Load Trained Models & Data Selection Results

: Step 2: Prediction Cross-lingual Zero-shot NER in target languages
set GPU_ID=0
set SEEDS=320
: set SEEDS=320 550 631 691 985
set TGTS=es nl

for %%e in (%SEEDS%) do (
for %%t in (%TGTS%) do (
python3 kd.py ^
--seed=%%e ^
--gpu_id=%GPU_ID% ^
--batch_size=32 ^
--tgt_lang=%%t ^
--eval_langs=%%t ^
--do_predict
)
)