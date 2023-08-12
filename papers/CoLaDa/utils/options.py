# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from ast import parse
import os
import random
import torch
import numpy as np

def add_args():
    def str2bool(arg):
        if arg.lower() == "true":
            return True
        return False
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--mode", type=str, choices=["ner", "step1", "step2"], default="ner")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--src_dev_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--trans_file", default=None, type=str)
    parser.add_argument("--unlabel_file", type=str)
    parser.add_argument("--output_dir",
                        default='out',
                        type=str,
                        help="the output directory where the final model checkpoint will be saved.")
    parser.add_argument("--ckpt_dir", default=None, type=str)
    parser.add_argument("--ckpt_trans_dir", default=None, type=str)
    parser.add_argument("--ckpt_tgt_dir", default=None, type=str)

    parser.add_argument("--model_name_or_path",
                        default='bert-base-multilingual-cased',
                        type=str,
                        help="pre-trained language model, default to roberta base.")
    # ner model parameters
    parser.add_argument("--use_crf",
                        type=str2bool,
                        default=True,
                        help="whether to use bert-crf")
    parser.add_argument("--schema", default="bio", type=str, choices=['bio', 'iobes'])
    parser.add_argument("--word_pooling",
                        default="first",
                        choices=["first", "avg"],
                        help="how to encoder word")
    parser.add_argument("--ner_drop",
                        type=float,
                        default=0.1)
    parser.add_argument("--ner_lr",
                        default=3e-5,
                        type=float,
                        help="the peak learning rate for noise robust training.")
    parser.add_argument("--train_ner_epochs",
                        default=3,
                        type=int,
                        help="total number of training epochs for noise robust training.")
    parser.add_argument("--ner_max_grad_norm", default=1.0, type=float)
    parser.add_argument("--frz_bert_layers", default=-1, type=int)
    parser.add_argument("--do_train",
                        type=str2bool,
                        default=False,
                        help="whether to run training.")
    parser.add_argument("--do_eval",
                        type=str2bool,
                        default=False,
                        help="whether to run eval on eval set or not.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="effective batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="batch size for eval.")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="the maximum input sequence length.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="proportion of learning rate warmup.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="weight decay for model training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for training")


    parser.add_argument('--lower_case', default=False, type=str2bool)
    parser.add_argument('--select_ckpt', default=False, type=str2bool)
    parser.add_argument('--val_steps', default=-1, type=int)
    parser.add_argument('--save_ckpt', default=False, type=str2bool)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument("--save_log", type=str2bool, default=False)
    
    parser.add_argument("--trans_lam", default=0, type=float)
    parser.add_argument("--T", default=-1, type=float)
    parser.add_argument("--knn_lid", default=3, type=int)
    parser.add_argument("--knn_pooling", default="avg", choices=["first", "avg"], help="how to encoder word")
    parser.add_argument("--ealpha", default=1, type=float)
    parser.add_argument("--emu", default=1, type=float)
    parser.add_argument("--filter_tgt", default="none", type=str, choices=["none", "reweight"])
    parser.add_argument("--filter_trans", default="none", type=str, choices=["none", "reweight"])
    parser.add_argument("--smK", default=0, type=int)
    parser.add_argument("--K", default=50, type=int)


    # parser.add_argument('--src_policy', default='ignore', type=str, choices=['ignore', 'merge', 'srcft'])

    args = parser.parse_args()
    for k, v in args.__dict__.items():
        if isinstance(v, str):
            if v == 'None' or v == 'none':
                args.__dict__[k] = None
    return args