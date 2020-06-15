set GPUNO=0
: set SEED=95 495 539 667 806
set SEED=95
: set KSHOT=0.01 0.02 0.05
set KSHOT=0.001

:::::::::::::::::::::::
: base model
:::::::::::::::::::::::

for %%s in (%SEED%) do (

:: ==> train-baseModel
python main.py --no_meta_learning --mask_rate -1.0 --lambda_max_loss 0.0 ^
--result_dir baseModel-seed_%%s --gpu_device %GPUNO% --seed %%s

:: ==> 0-shot-baseModel
python main.py --zero_shot --no_meta_learning --test_langs es nl de ^
--model_dir models\baseModel-seed_%%s --gpu_device %GPUNO% --seed %%s

for %%k in (%KSHOT%) do (

:: ==> k-shot-baseModel
python main.py --k_shot %%k --test_langs es nl de --lambda_max_loss 0.0 --max_ft_steps 10 --lr_finetune 1e-5 ^
--model_dir models\baseModel-seed_%%s --gpu_device %GPUNO% --seed %%s

)

)


::::::::::::::::::::::::
: proposed approach
::::::::::::::::::::::::
: mask_rate=-1 => no masking scheme
set MASK_RATE=0.2
: lambda_maxloss=0.0 => no max-loss
set LAMBDA_MAXLOSS=2.0

for %%s in (%SEED%) do (

:: ==> train-ours
python main.py --inner_steps 2 --mask_rate %MASK_RATE% --lambda_max_loss %LAMBDA_MAXLOSS% ^
--result_dir meta-innerSteps_2-maskRate_%MASK_RATE%-lambdaMaxLoss_%LAMBDA_MAXLOSS%-seed_%%s ^
--gpu_device %GPUNO% --seed %%s


:: ==> 0-shot-ours
python main.py --zero_shot --max_ft_steps 1 --test_langs es nl de ^
--lambda_max_loss 0.0 --support_size 2 --lr_finetune 1e-5 ^
--model_dir models\meta-innerSteps_2-maskRate_%MASK_RATE%-lambdaMaxLoss_%LAMBDA_MAXLOSS%-seed_%%s ^
--gpu_device %GPUNO% --seed %%s

for %%k in (%KSHOT%) do (

:: ==> k-shot-ours
python main.py --k_shot %%k --test_langs es nl de --lambda_max_loss 0.0 --max_ft_steps 10 --lr_finetune 1e-5 ^
--model_dir models\meta-innerSteps_2-maskRate_%MASK_RATE%-lambdaMaxLoss_%LAMBDA_MAXLOSS%-seed_%%s ^
--gpu_device %GPUNO% --seed %%s

)

)