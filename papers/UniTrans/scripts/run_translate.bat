set GPU_ID=0
set SRC_LANG=en
set TGT_LANGS=de
set EMB_PATH=data/embedding
set DICT_PATH=data/dict
set DATA_DIR=data\ner\conll

for %%t in (%TGT_LANGS%) do (
python MUSE\supervised.py --src_lang %SRC_LANG% --tgt_lang %%t ^
--src_emb %EMB_PATH%/wiki.%SRC_LANG%.vec --tgt_emb %EMB_PATH%/wiki.%%t.vec  ^
--n_refinement 3 --dico_train identical_char --max_vocab 100000 --dico_eval %DICT_PATH%/%SRC_LANG%-%%t.5000-6500.txt ^
--exp_path %DATA_DIR% --exp_name en2%%t --exp_id muse --export ""

python translate.py --tgt_lang=%%t --embed_dir %EMB_PATH% --gpu_id=%GPU_ID% --data_dir=%DATA_DIR%
)
