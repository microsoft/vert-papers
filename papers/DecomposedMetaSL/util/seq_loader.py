import json

import torch
import torch.utils.data as data
import os
from .fewshotsampler import DebugSampler, FewshotSampleBase
import numpy as np
import random
from collections import defaultdict
from .span_sample import SpanSample

class SeqBatcher:
    def __init__(self):
        return

    def batchnize_sent(self, data):
        batch = {"index": [], "word": [], "word_mask": [], "word_to_piece_ind": [], "word_to_piece_end": [],
                 "word_shape_ids": [], "word_labels": [],
                 "seq_len": [], "ment_labels": [], "sentence_num": [], "subsentence_num": []}
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
        word_shape_ids = np.zeros(shape=(len(batch["seq_len"]), max_n_words))
        ment_labels = np.full(shape=(len(batch['seq_len']), max_n_words), fill_value=-1, dtype='int')
        word_labels = np.full(shape=(len(batch['seq_len']), max_n_words), fill_value=-1, dtype='int') 
        for k, slen in enumerate(batch['seq_len']):
            assert len(batch['word_to_piece_ind'][k]) == slen
            assert len(batch['word_to_piece_end'][k]) == slen
            assert len(batch['word_shape_ids'][k]) == slen
            word_to_piece_ind[k, :slen] = batch['word_to_piece_ind'][k]
            word_to_piece_end[k, :slen] = batch['word_to_piece_end'][k]
            word_shape_ids[k, :slen] = batch['word_shape_ids'][k]
            ment_labels[k, :slen] = batch['ment_labels'][k]
            word_labels[k, :slen] = batch['word_labels'][k]
        batch['word_to_piece_ind'] = word_to_piece_ind
        batch['word_to_piece_end'] = word_to_piece_end
        batch['word_shape_ids'] = word_shape_ids
        batch['ment_labels'] = ment_labels
        batch['word_labels'] = word_labels
        for k, v in batch.items():
            if k not in ['sentence_num', 'index', 'subsentence_num', 'label2tag']:
                v = np.array(v, dtype=np.int32)
                batch[k] = torch.tensor(v, dtype=torch.long)
        batch['word'] = batch['word'][:, :max_n_piece]
        batch['word_mask'] = batch['word_mask'][:, :max_n_piece]
        return batch

    def __call__(self, batch):
        support_sets, query_sets = zip(*batch)
        batch_support = self.batchnize_sent(support_sets)
        batch_query = self.batchnize_sent(query_sets)
        batch_query["label2tag"] = []
        for i in range(len(query_sets)):
            batch_query["label2tag"].append(query_sets[i]["label2tag"])
        return {"support": batch_support, "query": batch_query}

class SeqNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, encoder, max_length, schema, bio=True, debug_file=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
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
        self.samples, self.classes = self.__load_data_from_file__(filepath, bio)
        if debug_file is not None:
            self.sampler = DebugSampler(debug_file)
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

    def __getraw__(self, sample):
        word, mask, word_to_piece_ind, word_to_piece_end, word_shape_ids, seq_lens = self.encoder.tokenize(sample.words)
        sent_st_id = 0
        split_seqs = []
        ment_labels = []
        word_labels = []
        for cur_len in seq_lens:
            sent_ed_id = sent_st_id + cur_len
            split_seqs.append(sample.tags[sent_st_id: sent_ed_id])
            cur_ment_seqs = []
            cur_word_seqs = []
            for wtag in split_seqs[-1]:
                cur_ment_seqs.append(self.ment_tag2label[self.get_ment_word_tag(wtag)])
                if wtag not in self.tag2label:
                    print(wtag, self.tag2label)
                cur_word_seqs.append(self.tag2label[wtag])
            ment_labels.append(cur_ment_seqs)
            word_labels.append(cur_word_seqs)
            sent_st_id += cur_len
        item = {"word": word, "word_mask": mask, "word_to_piece_ind": word_to_piece_ind, "word_to_piece_end": word_to_piece_end,
                "seq_len": seq_lens, "word_shape_ids": word_shape_ids, "ment_labels": ment_labels, "word_labels": word_labels,
                "subsentence_num": len(seq_lens)}
        return item

    def __additem__(self, index, d, item):
        d['index'].append(index)
        d['word'] += item['word']
        d['word_mask'] += item['word_mask']
        d['seq_len'] += item['seq_len']
        d['word_to_piece_ind'] += item['word_to_piece_ind']
        d['word_to_piece_end'] += item['word_to_piece_end']
        d['word_shape_ids'] += item['word_shape_ids']
        d['ment_labels'] += item['ment_labels']
        d['word_labels'] += item['word_labels']
        d['subsentence_num'].append(item['subsentence_num'])
        return

    def __populate__(self, idx_list, savelabeldic=False):
        dataset = {'index': [], 'word': [], 'word_mask': [], 'ment_labels': [], 'word_labels': [], 'word_to_piece_ind': [],
                   "word_to_piece_end": [], "seq_len": [], "word_shape_ids": [], "subsentence_num": []}
        for idx in idx_list:
            item = self.__getraw__(self.samples[idx])
            self.__additem__(idx, dataset, item)
        if savelabeldic:
            dataset['label2tag'] = self.label2tag
        dataset['sentence_num'] = len(dataset["seq_len"])
        assert len(dataset['word']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_ind']) == len(dataset['seq_len'])
        assert len(dataset['word_to_piece_end']) == len(dataset['seq_len'])
        assert len(dataset['ment_labels']) == len(dataset['seq_len'])
        assert len(dataset['word_labels']) == len(dataset['seq_len'])
        return dataset

    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__(index)
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
        self.tag2label = {tag: idx for idx, tag in enumerate(target_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(target_tags)}
        support_set = self.__populate__(support_idx, savelabeldic=False)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return 1000000

    def batch_to_device(self, batch, device):
        for k, v in batch.items():
            if k == 'sentence_num' or k == 'index' or k == 'subsentence_num' or k == 'label2tag':
                continue
            batch[k] = v.to(device)
        return batch


def get_seq_loader(filepath, mode, encoder, batch_size, max_length, schema, bio, shuffle, debug_file=None, num_workers=8):
    batcher = SeqBatcher()
    dataset = SeqNERDataset(filepath, encoder, max_length, schema, bio, debug_file=debug_file)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=batcher)
    return dataloader