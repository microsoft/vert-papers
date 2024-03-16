import json
import torch
import torch.utils.data as data
import os
from .fewshotsampler import DebugSampler, FewshotSampleBase
import numpy as np
import random
from collections import defaultdict
from .span_sample import SpanSample
from .span_loader import QuerySpanBatcher


class JointBatcher(QuerySpanBatcher):
    def __init__(self, iou_thred, use_oproto):
        super().__init__(iou_thred, use_oproto)
        return

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
                 'split_words': [],
                 "ment_labels": []}
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
        ment_labels = np.full(shape=(len(batch['seq_len']), max_n_words), fill_value=-1, dtype='int')
        for k, slen in enumerate(batch['seq_len']):
            assert len(batch['word_to_piece_ind'][k]) == slen
            assert len(batch['word_to_piece_end'][k]) == slen
            word_to_piece_ind[k, :slen] = batch['word_to_piece_ind'][k]
            word_to_piece_end[k, :slen] = batch['word_to_piece_end'][k]
            ment_labels[k, :slen] = batch['ment_labels'][k]
        batch['word_to_piece_ind'] = word_to_piece_ind
        batch['word_to_piece_end'] = word_to_piece_end
        batch['ment_labels'] = ment_labels
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


class JointNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, encoder, N, K, Q, max_length, schema, \
                 bio=True, debug_file=None, query_file=None, hidden_query_label=False, labelname_fn=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.schema = schema # this means the meta-train/test schema
        if self.schema == 'BIO':
            self.ment_tag2label = {"O": 0, "B-X": 1, "I-X": 2}
        elif self.schema == 'IO':
            self.ment_tag2label = {"O": 0, "I-X": 1}
        elif self.schema == 'BIOES':
            self.ment_tag2label = {"O": 0, "B-X": 1, "I-X": 2, "E-X": 3, "S-X": 4}
        else:
            raise ValueError
        self.ment_label2tag = {lidx: tag for tag, lidx in self.ment_tag2label.items()}
        self.label2tag = None
        self.tag2label = None
        self.sql_label2tag = None
        self.sql_tag2label = None
        self.samples, self.classes = self.__load_data_from_file__(filepath, bio)
        if debug_file:
            if query_file is None:
                print("use golden mention for typing !!! input_fn: {}".format(filepath))
            self.sampler = DebugSampler(debug_file, query_file)
        self.max_length = max_length
        return

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
        return

    def __load_data_from_file__(self, filepath, bio):
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
                sample = SpanSample(index, samplelines, bio)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        if len(samplelines):
            sample = SpanSample(index, samplelines, bio)
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            self.__insert_sample__(index, sample_classes)
            classes += sample_classes
        classes = list(set(classes))
        max_span_len = -1
        long_ent_num = 0
        tot_ent_num = 0
        tot_tok_num = 0
        for eid, sample in enumerate(samples):
            max_span_len = max(max_span_len, sample.get_max_ent_len())
            long_ent_num += sample.get_num_of_long_ent(10)
            tot_ent_num += len(sample.spans)
            tot_tok_num += len(sample.words)
            # convert seq labels to target schema
            new_tags = ['O' for _ in range(len(sample.words))]
            for sp in sample.spans:
                stype = sp[0]
                sp_st = sp[1]
                sp_ed = sp[2]
                assert stype != "O"
                if self.schema == 'IO':
                    for k in range(sp_st, sp_ed + 1):
                        new_tags[k] = "I-" + stype
                elif self.schema == 'BIO':
                    new_tags[sp_st] = "B-" + stype
                    for k in range(sp_st + 1, sp_ed + 1):
                        new_tags[k] = "I-" + stype
                elif self.schema == 'BIOES':
                    if sp_st == sp_ed:
                        new_tags[sp_st] = "S-" + stype
                    else:
                        new_tags[sp_st] = "B-" + stype
                        new_tags[sp_ed] = "E-" + stype
                        for k in range(sp_st + 1, sp_ed):
                            new_tags[k] = "I-" + stype
                else:
                    raise ValueError
            assert len(new_tags) == len(samples[eid].tags)
            samples[eid].tags = new_tags
        print("Sentence num {}, token num {}, entity num {} in file {}".format(len(samples), tot_tok_num, tot_ent_num,
                                                                               filepath))
        print("Total classes {}: {}".format(len(classes), str(classes)))
        print("The max golden entity len in the dataset is ", max_span_len)
        print("The max golden entity len in the dataset is greater than 10", long_ent_num)
        print("The total coverage of spans: {:.5f}".format(1 - long_ent_num / tot_ent_num))
        return samples, classes

    def get_ment_word_tag(self, wtag):
        if wtag == "O":
            return wtag
        return wtag[:2] + "X"

    def __getraw__(self, sample, add_split):
        word, mask, word_to_piece_ind, word_to_piece_end, word_shape_ids, seq_lens = self.encoder.tokenize(sample.words)
        sent_st_id = 0
        split_seqs = []
        ment_labels = []
        word_labels = []

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
            split_seqs.append(sample.tags[sent_st_id: sent_ed_id])
            cur_ment_seqs = []
            cur_word_seqs = []
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

            for wtag in split_seqs[-1]:
                cur_ment_seqs.append(self.ment_tag2label[self.get_ment_word_tag(wtag)])
                cur_word_seqs.append(-1) # not used
            ment_labels.append(cur_ment_seqs)
            word_labels.append(cur_word_seqs)
            sent_st_id += cur_len
        item = {"word": word, "word_mask": mask, "word_to_piece_ind": word_to_piece_ind, "word_to_piece_end": word_to_piece_end,
                "seq_len": seq_lens, "word_shape_ids": word_shape_ids, "ment_labels": ment_labels, "word_labels": word_labels,
                "subsentence_num": len(seq_lens), "spans": split_spans, "query_ments": split_querys, "query_probs": split_probs,
                "split_words": split_words}
        return item

    def __additem__(self, index, d, item):
        d['index'].append(index)
        d['word'] += item['word']
        d['word_mask'] += item['word_mask']
        d['seq_len'] += item['seq_len']
        d['word_to_piece_ind'] += item['word_to_piece_ind']
        d['word_to_piece_end'] += item['word_to_piece_end']
        d['word_shape_ids'] += item['word_shape_ids']
        d['spans'] += item['spans']
        d['query_ments'] += item['query_ments']
        d['query_probs'] += item['query_probs']
        d['ment_labels'] += item['ment_labels']
        d['word_labels'] += item['word_labels']
        d['subsentence_num'].append(item['subsentence_num'])
        d['split_words'] += item['split_words']
        return

    def __populate__(self, idx_list, query_ment_mp=None, savelabeldic=False, add_split=False):
        dataset = {'index': [], 'word': [], 'word_mask': [], 'ment_labels': [], 'word_labels': [], 'word_to_piece_ind': [],
                   "word_to_piece_end": [], "seq_len": [], "word_shape_ids": [], "subsentence_num": [], 'spans': [], "query_ments": [], "query_probs": [], 'split_words': []}
        for idx in idx_list:
            if query_ment_mp is not None:
                self.samples[idx].query_ments = [x[0] for x in query_ment_mp[str(self.samples[idx].index)]]
                self.samples[idx].query_probs = [x[1] for x in query_ment_mp[str(self.samples[idx].index)]]
            else:
                self.samples[idx].query_ments = [[x[1], x[2]] for x in self.samples[idx].spans]
                self.samples[idx].query_probs = [1 for x in self.samples[idx].spans]          
            item = self.__getraw__(self.samples[idx], add_split)
            self.__additem__(idx, dataset, item)
        if savelabeldic:
            dataset['label2tag'] = self.label2tag
            dataset['sql_label2tag'] = self.sql_label2tag
        dataset['sentence_num'] = len(dataset["seq_len"])
        assert len(dataset['word']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_ind']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_end']) == len(dataset['seq_len'])
        assert len(dataset['ment_labels']) == len(dataset['seq_len'])
        assert len(dataset['word_labels']) == len(dataset['seq_len'])
        return dataset

    def __getitem__(self, index):
        tmp = self.sampler.__next__(index)
        if len(tmp) == 3:
            target_classes, support_idx, query_idx = tmp
            query_ment_mp = None
        else:
            target_classes, support_idx, query_idx, query_ment_mp = tmp
        target_tags = ['O']
        for cname in target_classes:
            if self.schema == 'IO':
                target_tags.append(f"I-{cname}")
            elif self.schema == 'BIO':
                target_tags.append(f"B-{cname}")
                target_tags.append(f"I-{cname}")
            elif self.schema == 'BIOES':
                target_tags.append(f"B-{cname}")
                target_tags.append(f"I-{cname}")
                target_tags.append(f"E-{cname}")
                target_tags.append(f"S-{cname}")
            else:
                raise ValueError              
        self.sql_tag2label = {tag: idx for idx, tag in enumerate(target_tags)}
        self.sql_label2tag = {idx: tag for idx, tag in enumerate(target_tags)}
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, add_split=True, savelabeldic=False)
        query_set = self.__populate__(query_idx, query_ment_mp=query_ment_mp, add_split=True, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return 1000000

    def batch_to_device(self, batch, device):
        for k, v in batch.items():
            if k in ['sentence_num', 'label2tag', 'spans', 'index', 'query_ments', 'query_probs', "subsentence_num", 'split_words']:
                continue
            batch[k] = v.to(device)
        return batch


def get_joint_loader(filepath, mode, encoder, N, K, Q, batch_size, max_length, schema, 
               bio, shuffle, num_workers=8, debug_file=None, query_file=None,
               iou_thred=None, hidden_query_label=False, label_fn=None, use_oproto=False):
    batcher = JointBatcher(iou_thred=iou_thred, use_oproto=use_oproto)
    dataset = JointNERDataset(filepath, encoder, N, K, Q, max_length, bio=bio, schema=schema, 
                                  debug_file=debug_file, query_file=query_file, 
                                  hidden_query_label=hidden_query_label, labelname_fn=label_fn)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=lambda x: batcher(x, mode))
    return dataloader