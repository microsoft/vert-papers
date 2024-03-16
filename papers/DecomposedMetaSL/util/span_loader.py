import json
import torch
import torch.utils.data as data
import os
from .fewshotsampler import FewshotSampler, DebugSampler
from .span_sample import SpanSample
import numpy as np
import random
from collections import defaultdict
import logging, pickle, gzip
from tqdm import tqdm


class QuerySpanSample(SpanSample):
    def __init__(self, index, filelines, ment_prob_list, bio=True):
        super(QuerySpanSample, self).__init__(index, filelines, bio)
        self.query_ments = []
        self.query_probs = []
        for i in range(len(ment_prob_list)):
            self.query_ments.append(ment_prob_list[i][0])
            self.query_probs.append(ment_prob_list[i][1])
        return

class QuerySpanBatcher:
    def __init__(self, iou_thred, use_oproto):
        self.iou_thred = iou_thred
        self.use_oproto = use_oproto
        return

    def _get_span_weights(self, batch_triples, seq_len, span_indices, span_tags, thred=0.6, alpha=1):
        span_weights = []
        for k in range(len(span_indices)):
            seq_tags = np.zeros(seq_len[k], dtype=np.int)
            span_st_inds = np.zeros(seq_len[k], dtype=np.int)
            span_ed_inds = np.zeros(seq_len[k], dtype=np.int)
            span_weights.append([1] * len(span_indices[k]))
            for [r, i, j] in batch_triples[k]:
                seq_tags[i:j + 1] = r
                span_st_inds[i:j + 1] = i
                span_ed_inds[i:j + 1] = j
            for sp_idx in range(len(span_indices[k])):
                sp_st = span_indices[k][sp_idx][0]
                sp_ed = span_indices[k][sp_idx][1]
                sp_tag = span_tags[k][sp_idx]
                if sp_tag != 0:
                    continue
                cur_token_tags = list(seq_tags[sp_st: sp_ed + 1])
                max_tag = max(cur_token_tags, key=cur_token_tags.count)
                anchor_idx = cur_token_tags.index(max_tag) + sp_st
                if max_tag == 0:
                    continue
                cur_ids = set(range(sp_st, sp_ed + 1))
                ref_ids = set(range(span_st_inds[anchor_idx], span_ed_inds[anchor_idx] + 1))
                tag_percent = len(cur_ids & ref_ids) / len(cur_ids | ref_ids)
                if tag_percent >= thred:
                    span_tags[k][sp_idx] = max_tag
                    span_weights[k][sp_idx] = tag_percent ** alpha
                else:
                    span_weights[k][sp_idx] = (1 - tag_percent) ** alpha
        return span_tags, span_weights

    @staticmethod
    def _make_golden_batch(batch_triples, seq_len):
        span_indices = []
        span_tags = []
        for k in range(len(batch_triples)):
            per_sp_indices = []
            per_sp_tags = []
            for (r, i, j) in batch_triples[k]:
                if j < seq_len[k]:
                    per_sp_indices.append([i, j])
                    per_sp_tags.append(r)
                else:
                    print("something error")
            span_indices.append(per_sp_indices)
            span_tags.append(per_sp_tags)
        return span_indices, span_tags

    @staticmethod
    def _make_train_query_batch(query_ments, query_probs, golden_triples, seq_len):
        span_indices = []
        ment_probs = []
        span_tags = []
        for k in range(len(query_ments)):
            per_sp_indices = []
            per_sp_tags = []
            per_ment_probs = []
            per_tag_mp = {(i, j): tag for tag, i, j in golden_triples[k]}
            for [i, j], prob in zip(query_ments[k], query_probs[k]):
                if j < seq_len[k]:
                    per_sp_indices.append([i, j])
                    per_sp_tags.append(per_tag_mp.get((i, j), 0))
                    per_ment_probs.append(prob)
                else:
                    print("something error")
            #add golden mentions into query batch
            for [r, i, j] in golden_triples[k]:
                if [i, j] not in per_sp_indices:
                    per_sp_indices.append([i, j])
                    per_sp_tags.append(r)
                    per_ment_probs.append(0)
            span_indices.append(per_sp_indices)
            span_tags.append(per_sp_tags)
            ment_probs.append(per_ment_probs)
        return span_indices, span_tags, ment_probs

    @staticmethod
    def _make_test_query_batch(query_ments, query_probs, golden_triples, seq_len):
        span_indices = []
        ment_probs = []
        span_tags = []
        for k in range(len(query_ments)):
            per_sp_indices = []
            per_sp_tags = []
            per_ment_probs = []
            per_tag_mp = {(i, j): tag for tag, i, j in golden_triples[k]}
            for [i, j], prob in zip(query_ments[k], query_probs[k]):
                if j < seq_len[k]:
                    per_sp_indices.append([i, j])
                    per_sp_tags.append(per_tag_mp.get((i, j), 0))
                    per_ment_probs.append(prob)
                else:
                    print("something error")
            span_indices.append(per_sp_indices)
            span_tags.append(per_sp_tags)
            ment_probs.append(per_ment_probs)
        return span_indices, span_tags, ment_probs

    def make_support_batch(self, batch):
        span_indices, span_tags = self._make_golden_batch(batch["spans"], batch["seq_len"])
        if self.use_oproto:
            used_token_flag = []
            for k in range(len(batch["spans"])):
                used_token_flag.append(np.zeros(batch["seq_len"][k]))
                for [r, sp_st, sp_ed] in batch["spans"][k]:
                    used_token_flag[k][sp_st: sp_ed + 1] = 1
            for k in range(len(batch["seq_len"])):
                for j in range(batch["seq_len"][k]):
                    if used_token_flag[k][j] == 1:
                        continue
                    if [j, j] not in span_indices[k]:
                        span_indices[k].append([j, j])
                        span_tags[k].append(0)
                    else:
                        print("something error")
        span_nums = [len(x) for x in span_indices]
        max_n_spans = max(span_nums)
        span_masks = np.zeros(shape=(len(span_indices), max_n_spans), dtype='int')
        for k in range(len(span_indices)):
            span_masks[k, :span_nums[k]] = 1
            while len(span_tags[k]) < max_n_spans:
                span_indices[k].append([0, 0])
                span_tags[k].append(-1)
        batch["span_indices"] = span_indices
        batch["span_mask"] = span_masks
        batch["span_tag"] = span_tags
        # all example with equal weights
        batch["span_weights"] = np.ones(shape=span_masks.shape, dtype='float')
        return batch

    def make_query_batch(self, batch, is_train):
        if is_train:
            span_indices, span_tags, ment_probs = self._make_train_query_batch(batch["query_ments"],
                                                                               batch["query_probs"], batch["spans"],
                                                                               batch["seq_len"])
            if self.iou_thred is None:
                span_weights = None
            else:
                span_tags, span_weights = self._get_span_weights(batch["spans"], batch["seq_len"], span_indices,
                                                                 span_tags, thred=self.iou_thred)

        else:
            span_indices, span_tags, ment_probs = self._make_test_query_batch(batch["query_ments"],
                                                                              batch["query_probs"], batch["spans"],
                                                                              batch["seq_len"])
            span_weights = None

        span_nums = [len(x) for x in span_indices]
        max_n_spans = max(span_nums)
        span_masks = np.zeros(shape=(len(span_indices), max_n_spans), dtype='int')
        new_span_weights = np.ones(shape=span_masks.shape, dtype='float')
        for k in range(len(span_indices)):
            span_masks[k, :span_nums[k]] = 1
            if span_weights is not None:
                new_span_weights[k, :span_nums[k]] = span_weights[k]
            while len(span_tags[k]) < max_n_spans:
                span_indices[k].append([0, 0])
                span_tags[k].append(-1)
                ment_probs[k].append(-100)
        batch["span_indices"] = span_indices
        batch["span_mask"] = span_masks
        batch["span_tag"] = span_tags
        batch["span_probs"] = ment_probs
        batch["span_weights"] = new_span_weights
        return batch

    def batchnize_episode(self, data, mode):
        support_sets, query_sets = zip(*data)
        if mode == "train":
            is_train = True
        else:
            is_train = False
        batch_support = self.batchnize_sent(support_sets, "support", is_train)
        batch_query = self.batchnize_sent(query_sets, "query", is_train)
        batch_query["label2tag"] = []
        for i in range(len(query_sets)):
            batch_query["label2tag"].append(query_sets[i]["label2tag"])
        return {"support": batch_support, "query": batch_query}

    def batchnize_sent(self, data, mode, is_train):
        batch = {"index": [], "word": [], "word_mask": [], "word_to_piece_ind": [], "word_to_piece_end": [],
                 "seq_len": [],
                 "spans": [], "sentence_num": [], "query_ments": [], "query_probs": [], "subsentence_num": [],
                 'split_words': []}
        for i in range(len(data)):
            for k in batch.keys():
                if k == 'sentence_num':
                    batch[k].append(data[i][k])
                else:
                    batch[k] += data[i][k]

        max_n_piece = max([sum(x) for x in batch['word_mask']])

        max_n_words = max(batch["seq_len"])
        word_to_piece_ind = np.zeros(shape=(len(batch["seq_len"]), max_n_words))
        word_to_piece_end = np.zeros(shape=(len(batch["seq_len"]), max_n_words))
        for k, slen in enumerate(batch['seq_len']):
            assert len(batch['word_to_piece_ind'][k]) == slen
            assert len(batch['word_to_piece_end'][k]) == slen
            word_to_piece_ind[k, :slen] = batch['word_to_piece_ind'][k]
            word_to_piece_end[k, :slen] = batch['word_to_piece_end'][k]
        batch['word_to_piece_ind'] = word_to_piece_ind
        batch['word_to_piece_end'] = word_to_piece_end
        if mode == "support":
            batch = self.make_support_batch(batch)
        else:
            batch = self.make_query_batch(batch, is_train)
        for k, v in batch.items():
            if k not in ['spans', 'sentence_num', 'label2tag', 'index', "query_ments", "query_probs", "subsentence_num",
                         "split_words"]:
                v = np.array(v)
                if k == "span_weights":
                    batch[k] = torch.tensor(v).float()
                else:
                    batch[k] = torch.tensor(v).long()
        batch['word'] = batch['word'][:, :max_n_piece]
        batch['word_mask'] = batch['word_mask'][:, :max_n_piece]
        return batch

    def __call__(self, batch, mode):
        return self.batchnize_episode(batch, mode)



class QuerySpanNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepath, encoder, N, K, Q, max_length, \
                 bio=True, debug_file=None, query_file=None, hidden_query_label=False, labelname_fn=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.cur_t = 0
        self.samples, self.classes = self.__load_data_from_file__(filepath, bio)
        if labelname_fn:
            self.class2name = self.__load_names__(labelname_fn)
        else:
            self.class2name = None
        self.sampler = FewshotSampler(N, K, Q, self.samples, classes=self.classes)
        if debug_file:
            if query_file is None:
                print("use golden mention for typing !!! input_fn: {}".format(filepath))
            self.sampler = DebugSampler(debug_file, query_file)
        self.max_length = max_length
        self.tag2label = None
        self.label2tag = None
        self.hidden_query_label = hidden_query_label

        if self.hidden_query_label:
            print("Attention! This dataset hidden query set labels !")

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
        return

    def __load_names__(self, label_mp_fn):
        class2str = {}
        with open(label_mp_fn, mode="r", encoding="utf-8") as fp:
            for line in fp:
                label, lstr = line.strip("\n").split(":")
                class2str[label.strip()] = lstr.strip()
        return class2str

    def __load_data_from_file__(self, filepath, bio):
        print("load input text from {}".format(filepath))
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip("\n")
            if len(line):
                samplelines.append(line)
            else:
                cur_ments = []
                sample = QuerySpanSample(index, samplelines, cur_ments, bio)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        if len(samplelines):
            cur_ments = []
            sample = QuerySpanSample(index, samplelines, cur_ments, bio)
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            self.__insert_sample__(index, sample_classes)
            classes += sample_classes
        classes = list(set(classes))
        max_span_len = -1
        long_ent_num = 0
        tot_ent_num = 0
        tot_tok_num = 0

        for sample in samples:
            max_span_len = max(max_span_len, sample.get_max_ent_len())
            long_ent_num += sample.get_num_of_long_ent(10)
            tot_ent_num += len(sample.spans)
            tot_tok_num += len(sample.words)
        print("Sentence num {}, token num {}, entity num {} in file {}".format(len(samples), tot_tok_num, tot_ent_num,
                                                                               filepath))
        print("Total classes {}: {}".format(len(classes), str(classes)))
        print("The max golden entity len in the dataset is ", max_span_len)
        print("The max golden entity len in the dataset is greater than 10", long_ent_num)
        print("The total coverage of spans: {:.5f}".format(1 - long_ent_num / tot_ent_num))
        return samples, classes


    def __getraw__(self, sample, add_split):
        word, mask, word_to_piece_ind, word_to_piece_end, word_shape_inds, seq_lens = self.encoder.tokenize(sample.words, true_token_flags=None)
        sent_st_id = 0
        split_spans = []
        split_querys = []
        split_probs = []

        split_words = []
        cur_wid = 0
        for k in range(len(seq_lens)):
            split_words.append(sample.words[cur_wid: cur_wid + seq_lens[k]])
            cur_wid += seq_lens[k]

        for cur_len in seq_lens:
            sent_ed_id = sent_st_id + cur_len
            split_spans.append([])
            for tag, span_st, span_ed in sample.spans:
                if (span_st >= sent_ed_id) or (span_ed < sent_st_id):  # span totally not in subsent
                    continue
                if (span_st >= sent_st_id) and (span_ed < sent_ed_id):  # span totally in subsent
                    split_spans[-1].append([self.tag2label[tag], span_st - sent_st_id, span_ed - sent_st_id])
                elif add_split:
                    if span_st >= sent_st_id:
                        split_spans[-1].append([self.tag2label[tag], span_st - sent_st_id, sent_ed_id - 1 - sent_st_id])
                    else:
                        split_spans[-1].append([self.tag2label[tag], 0, span_ed - sent_st_id])
            split_querys.append([])
            split_probs.append([])
            for [span_st, span_ed], span_prob in zip(sample.query_ments, sample.query_probs):
                if (span_st >= sent_ed_id) or (span_ed < sent_st_id):  # span totally not in subsent
                    continue
                if (span_st >= sent_st_id) and (span_ed < sent_ed_id):  # span totally in subsent
                    split_querys[-1].append([span_st - sent_st_id, span_ed - sent_st_id])
                    split_probs[-1].append(span_prob)
                elif add_split:
                    if span_st >= sent_st_id:
                        split_querys[-1].append([span_st - sent_st_id, sent_ed_id - 1 - sent_st_id])
                        split_probs[-1].append(span_prob)
                    else:
                        split_querys[-1].append([0, span_ed - sent_st_id])
                        split_probs[-1].append(span_prob)
            sent_st_id += cur_len

        item = {"word": word, "word_mask": mask, "word_to_piece_ind": word_to_piece_ind,
                "word_to_piece_end": word_to_piece_end, "spans": split_spans, "query_ments": split_querys, "query_probs": split_probs, "seq_len": seq_lens,
                "subsentence_num": len(seq_lens), "split_words": split_words}
        return item

    def __additem__(self, index, d, item):
        d['index'].append(index)
        d['word'] += item['word']
        d['word_mask'] += item['word_mask']
        d['seq_len'] += item['seq_len']
        d['word_to_piece_ind'] += item['word_to_piece_ind']
        d['word_to_piece_end'] += item['word_to_piece_end']
        d['spans'] += item['spans']
        d['query_ments'] += item['query_ments']
        d['query_probs'] += item['query_probs']
        d['subsentence_num'].append(item['subsentence_num'])
        d['split_words'] += item['split_words']
        return

    def __populate__(self, idx_list, query_ment_mp=None, savelabeldic=False, add_split=False):
        dataset = {'index': [], 'word': [], 'word_mask': [], 'spans': [], 'word_to_piece_ind': [],
                   "word_to_piece_end": [], "seq_len": [], "query_ments": [], "query_probs": [], "subsentence_num": [],
                   'split_words': []}
        for idx in idx_list:
            if query_ment_mp is not None:
                self.samples[idx].query_ments = [x[0] for x in query_ment_mp[str(self.samples[idx].index)]]
                self.samples[idx].query_probs = [x[1] for x in query_ment_mp[str(self.samples[idx].index)]]
            else:
                self.samples[idx].query_ments = [[x[1], x[2]] for x in self.samples[idx].spans]
                self.samples[idx].query_probs = [1 for x in self.samples[idx].spans]               
            item = self.__getraw__(self.samples[idx], add_split)
            self.__additem__(idx, dataset, item)
        dataset['sentence_num'] = len(dataset["seq_len"])
        assert len(dataset['word']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_ind']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_end']) == len(dataset['seq_len'])
        assert len(dataset['spans']) == len(dataset['seq_len'])
        if savelabeldic:
            dataset['label2tag'] = self.label2tag
        return dataset

    def __getitem__(self, index):
        tmp = self.sampler.__next__(index)
        if len(tmp) == 3:
            target_classes, support_idx, query_idx = tmp
            query_ment_mp = None
        else:
            target_classes, support_idx, query_idx, query_ment_mp = tmp
        target_classes = sorted(target_classes)
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, add_split=True)
        query_set = self.__populate__(query_idx, query_ment_mp=query_ment_mp, add_split=True, savelabeldic=True)
        if self.hidden_query_label:
            query_set['spans'] = [[] for i in range(query_set['sentence_num'])]
        return support_set, query_set

    def __len__(self):
        return 1000000

    def batch_to_device(self, batch, device):
        for k, v in batch.items():
            if k in ['sentence_num', 'label2tag', 'spans', 'index', 'query_ments', 'query_probs', "subsentence_num", 'split_words']:
                continue
            batch[k] = v.to(device)
        return batch

def get_query_loader(filepath, mode, encoder, N, K, Q, batch_size, max_length,
               bio, shuffle, num_workers=8, debug_file=None, query_file=None,
               iou_thred=None, hidden_query_label=False, label_fn=None, use_oproto=False):
    batcher = QuerySpanBatcher(iou_thred, use_oproto)
    dataset = QuerySpanNERDataset(filepath, encoder, N, K, Q, max_length, bio, 
                                  debug_file=debug_file, query_file=query_file, 
                                  hidden_query_label=hidden_query_label, labelname_fn=label_fn)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=lambda x: batcher(x, mode))
    return dataloader