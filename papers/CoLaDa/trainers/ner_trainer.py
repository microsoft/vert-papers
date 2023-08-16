# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path
import json
from seqeval.metrics import precision_score, recall_score
from seqeval.scheme import IOB2

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.seqtagger import NERTagger
from utils.utils_ner import *
from transformers import AutoTokenizer, AutoConfig

from .base_trainer import *

class SrcNERTrainer(NERTrainer):
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.lower_case)
        processor = DataUtils(tokenizer, args.max_seq_length)
        src_ner_bert_dataset = processor.get_ner_dataset(args.train_file)
        super().__init__(args, processor)
        self.src_ner_bert_dataset = src_ner_bert_dataset
        self.src_dev_dataset = self.processor.get_ner_dataset(self.args.src_dev_file)
        if args.do_eval: # do test
            self.eval_dataset = self.processor.get_ner_dataset(self.args.test_file)
        return

    def train_ner(self):
        return super().train_ner(self.src_ner_bert_dataset, self.src_dev_dataset, self.eval_dataset, self.args.output_dir)