# coding=utf-8
# checked
from __future__ import print_function

import optparse
import os
import pickle
import time

import torch

from data_loader_conll import DataLoaderCoNLL
from evaluation import evaluating
from constants import Constants

t = time.time()

optparser = optparse.OptionParser()
optparser.add_option(
    "--test", default="datasets/english/conll2003/eng.testb.test",
    help="Test set file path"
)
optparser.add_option(
    '--use_gpu', default='1',
    type='int', help='Whether or not to use gpu'
)
optparser.add_option(
    '--model_path', default='models/GRN.model',
    help='Model path'
)
optparser.add_option(
    '--map_path', default='models/GRN.mappings.pkl',
    help='Mapping file path'
)
optparser.add_option(
    '--cuda', default='0',
    type='int', help='Index of the cuda device'
)

opts = optparser.parse_args()[0]

mapping_file = opts.map_path

assert os.path.exists(mapping_file)
assert os.path.exists(opts.model_path)
assert os.path.exists(opts.test)
assert os.path.isfile(opts.test)
assert opts.cuda >= 0

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = mappings['id_to_tag']
char_to_id = mappings['char_to_id']

use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()
device_count = torch.cuda.device_count() if use_gpu else 0
assert (not use_gpu) or opts.cuda < device_count
device_name = "cuda:{0}".format(opts.cuda) if use_gpu else "cpu"
device = torch.device(device_name)

eval_path = Constants.Eval_Folder
eval_temp = Constants.Eval_Temp_Folder
eval_script = Constants.Eval_Script

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)

word_to_lower = mappings['word_to_lower']
digits_to_zeros = mappings['digits_to_zeros']
label_schema = mappings['label_schema']

test_set = DataLoaderCoNLL(opts.test, mappings)

model = torch.load(opts.model_path, map_location=device_name)
model.use_gpu = use_gpu
model_name = opts.model_path.split('/')[-1].split('.')[0]

char_mode = model.char_mode

if use_gpu:
    model = model.to(device)

model.eval()
with torch.no_grad():
    best_test_F, new_test_F, _, __ = evaluating(model, test_set, 0.0, model_name, mappings, char_mode, use_gpu, device, 1)
    print("F1: {0:.4f}".format(new_test_F))

print(time.time() - t)
