# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, random
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from itertools import groupby
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, BatchSampler)
from seqeval.metrics import classification_report, f1_score
from collections import defaultdict
from seqeval.scheme import IOB2
import re


def set_seed(seed):
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

class NERDataset(Dataset):
    def __init__(self, tensor_mp):
        super(NERDataset, self).__init__()
        self.data = tensor_mp
        return

    def __getitem__(self, idx):
        inputs = ()
        for key in ['idx', 'words', 'word_masks', 'word_to_piece_inds', 'word_to_piece_ends', 'sent_lens', 'labels', 'weights', 'valid_masks']:
            if (key in self.data) and (self.data[key] is not None):
                inputs = inputs + (self.data[key][idx],)
        return inputs

    def __len__(self):
        return len(self.data["words"])   

class DataUtils:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_length = max_seq_length
        self.label2idx = None
        self.idx2label = None
        return

    def read_conll(self, file_name):
        def is_empty_line(line_pack):
            return len(line_pack.strip()) == 0
        sent_len = 0
        ent_len = 0
        sentences = []
        labels = []
        type_mp = defaultdict(int)
        token_num = 0
        bio_err = 0
        with open(file_name, mode="r", encoding="utf-8") as fp:
            lines = [line.strip("\n") for line in fp]
            groups = groupby(lines, is_empty_line)
            for is_empty, pack in groups:
                if is_empty is False:
                    cur_sent = []
                    cur_labels = []
                    prev = "O"
                    for wt_pair in pack:
                        tmp = wt_pair.split("\t")
                        if len(tmp) == 1:
                            w, t = tmp[0].split()
                        else:
                            w, t = tmp
                        if t != "O":
                            ent_len += 1
                        if prev == "O" and t[0] != "B" and t != "O":
                            bio_err += 1
                        cur_sent.append(w)
                        cur_labels.append(t)
                        token_num += 1
                        prev = t
                        if t[0] == "B" or t[0] == "S":
                            type_mp[t[2:]] += 1
                    sent_len += len(cur_sent)
                    sentences.append(cur_sent)
                    labels.append(cur_labels)
        ent_num = sum(type_mp.values())
        sent_len /= len(sentences)
        ent_len /= ent_num
        print("{}: ent {} token {} sent {}".format(file_name, ent_num, token_num, len(sentences)))
        print("avg sent len: {} avg ent len: {}".format(sent_len, ent_len))
        print("bio error {}".format(bio_err))
        print(sorted([(k, v) for k, v in type_mp.items()], key=lambda x: x[0]))
        if self.label2idx is None:
            print("get label map !!!!!!!!!!!!!!!")
            self.get_label_map(labels)
            print(self.label2idx)
        return sentences, labels

    def read_examples_from_file(self, file_name):
        guid_index = 0
        sentences = []
        labels = []
        with open(file_name, encoding="utf-8") as f:
            cur_words = []
            cur_labels = []
            for line in f:
                line = line.strip().replace("\t", " ")
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if cur_words:
                        sentences.append(cur_words)
                        labels.append(cur_labels)
                        guid_index += 1
                        cur_words = []
                        cur_labels = []
                else:
                    splits = line.split(" ")
                    cur_words.append(splits[0])
                    cur_labels.append(splits[-1].replace("\n", ""))
            if cur_words:
                sentences.append(cur_words)
                labels.append(cur_labels)
                guid_index += 1
        print("number of sentences : {}".format(len(sentences)))
        return sentences, labels

    def get_label_map(self, labels):
        labels = sorted(set([y for x in labels for y in x]))
        label2id_mp = {}
        id2label_mp = {}
        for i, tag in enumerate(labels):
            label2id_mp[tag] = i
            id2label_mp[i] = tag
        self.label2idx = label2id_mp
        self.idx2label = id2label_mp
        return
    
    def get_ner_dataset(self, filename):
        sentences, labels = self.read_conll(filename)
        print(f"reading {filename}")
        dataset = self.process_data(sentences, labels, mode='ner')
        return dataset

    def get_loader(self, dataset, shuffle=True, batch_size=32):
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return data_loader

    def process_data(self, sentences, labels=None, mode='ner'):
        sent_num = len(sentences)
        data_word = np.full((sent_num, self.max_length), self.tokenizer.pad_token_id, dtype=np.int32)
        data_mask = np.zeros((sent_num, self.max_length), dtype=np.int32)
        data_w2p_ind = np.zeros((sent_num, self.max_length), dtype=np.int32)
        data_w2p_end = np.zeros((sent_num, self.max_length), dtype=np.int32)
        data_length = np.zeros((sent_num), dtype=np.int32)
        data_labels = np.full((sent_num, self.max_length), -100, dtype=np.int32)
        for i in range(sent_num):
            cur_tokens = [self.tokenizer.cls_token]
            cur_word_to_piece_ind = []
            cur_word_to_piece_end = []
            cur_seq_len = None
            for j, word in enumerate(sentences[i]):
                tag = labels[i][j] if labels else "O"
                if j > 0:
                    word = ' ' + word
                data_labels[i][j] = self.label2idx[tag]
                word_tokens = self.tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = [self.tokenizer.unk_token]
                    print("error token")
                if len(cur_tokens) + len(word_tokens) + 1 > self.max_length:
                    part_len = self.max_length - 1 - len(cur_tokens)
                    if part_len > 0:
                        cur_word_to_piece_ind.append(len(cur_tokens))
                        cur_tokens.extend(word_tokens[:part_len])
                        cur_word_to_piece_end.append(len(cur_tokens) - 1)
                    cur_tokens.append(self.tokenizer.sep_token)
                    cur_seq_len = len(cur_word_to_piece_ind)
                    break
                else:
                    cur_word_to_piece_ind.append(len(cur_tokens))
                    cur_tokens.extend(word_tokens)
                    cur_word_to_piece_end.append(len(cur_tokens) - 1)
            if cur_seq_len is None: # not full, append padding tokens
                assert len(cur_tokens) + 1 <= self.max_length
                cur_tokens.append(self.tokenizer.sep_token)
                cur_seq_len = len(cur_word_to_piece_ind)
            assert cur_seq_len == len(cur_word_to_piece_end)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(cur_tokens)
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = np.array(indexed_tokens)
            while len(cur_word_to_piece_ind) < self.max_length:
                cur_word_to_piece_ind.append(0)
                cur_word_to_piece_end.append(0)
            data_word[i, :] = indexed_tokens
            data_mask[i, :len(cur_tokens)] = 1
            data_w2p_ind[i, :] = cur_word_to_piece_ind
            data_w2p_end[i, :] = cur_word_to_piece_end
            data_length[i] = cur_seq_len
        return NERDataset({'idx': torch.arange(0, len(sentences)), 'words': torch.LongTensor(data_word), 'word_masks': torch.tensor(data_mask),
                'word_to_piece_inds': torch.LongTensor(data_w2p_ind), 'word_to_piece_ends': torch.LongTensor(data_w2p_end),
                'sent_lens': torch.LongTensor(data_length), 'labels': torch.LongTensor(data_labels), 
                'ori_sents': sentences, 'ori_labels': labels})

    def performance_report(self, y_pred_id, y_true_id, y_pred_probs=None, print_log=False):
        # convert id to tags
        y_pred_id = y_pred_id.detach().cpu().tolist()
        y_true_id = y_true_id.detach().cpu().tolist()
        if y_pred_probs is not None:
            y_pred_probs = y_pred_probs.detach().cpu().tolist()
        y_true = []
        y_pred = []
        y_probs = []
        sid = 0
        for tids, pids in zip(y_true_id, y_pred_id):
            y_true.append([])
            y_pred.append([])
            y_probs.append([])
            for k in range(len(tids)):
                if tids[k] == -100:
                    break
                y_true[-1].append(self.idx2label[tids[k]])
                if k < len(pids):
                    y_pred[-1].append(self.idx2label[pids[k]])
                    y_probs[-1].append(y_pred_probs[sid][k] if y_pred_probs is not None else None)
                else:
                    y_pred[-1].append("O")
                    y_probs[-1].append(None)
            sid += 1
        if print_log:
            print("============conlleval mode==================")
            report = classification_report(y_true, y_pred, digits=4)
            print(report)
        f1 = f1_score(y_true, y_pred, scheme=IOB2)
        if y_pred_probs is not None:
            return y_pred, y_true, y_probs, f1
        return y_pred, y_true, f1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = None
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
