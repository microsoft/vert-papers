import json

import torch
import torch.utils.data as data
import os
from .fewshotsampler import DebugSampler, FewshotSampleBase
import numpy as np
import random
from collections import defaultdict

class TokenSample(FewshotSampleBase):
    def __init__(self, idx, filelines):
        super(TokenSample, self).__init__()
        self.index = idx
        filelines = [line.split('\t') for line in filelines]
        if len(filelines[0]) == 2:
            self.words, self.tags = zip(*filelines)
        else:
            self.words, self.postags, self.tags = zip(*filelines)
        return

    def __count_entities__(self):
        self.class_count = {}
        for tag in self.tags:
            if tag in self.class_count:
                self.class_count[tag] += 1
            else:
                self.class_count[tag] = 1
        return

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        return list(set(self.tags))

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and \
               not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        return


class SeqBatcher:
    def __init__(self):
        return

    def batchnize_sent(self, data):
        batch = {"index": [], "word": [], "word_mask": [], "word_to_piece_ind": [], "word_to_piece_end": [],
                 "word_shape_ids": [], "word_labels": [],
                 "seq_len": [], "sentence_num": [], "subsentence_num": []}
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
        word_labels = np.full(shape=(len(batch['seq_len']), max_n_words), fill_value=-1, dtype='int') 
        for k, slen in enumerate(batch['seq_len']):
            assert len(batch['word_to_piece_ind'][k]) == slen
            assert len(batch['word_to_piece_end'][k]) == slen
            assert len(batch['word_shape_ids'][k]) == slen
            word_to_piece_ind[k, :slen] = batch['word_to_piece_ind'][k]
            word_to_piece_end[k, :slen] = batch['word_to_piece_end'][k]
            word_shape_ids[k, :slen] = batch['word_shape_ids'][k]
            word_labels[k, :slen] = batch['word_labels'][k]
        batch['word_to_piece_ind'] = word_to_piece_ind
        batch['word_to_piece_end'] = word_to_piece_end
        batch['word_shape_ids'] = word_shape_ids
        batch['word_labels'] = word_labels
        for k, v in batch.items():
            if k not in ['sentence_num', 'index', 'subsentence_num', 'label2tag']:
                v = np.array(v, dtype=np.int32)
                batch[k] = torch.tensor(v, dtype=torch.long)
        batch['word'] = batch['word'][:, :max_n_piece]
        batch['word_mask'] = batch['word_mask'][:, :max_n_piece]
        return batch

    def __call__(self, batch, use_episode):
        if use_episode:
            support_sets, query_sets = zip(*batch)
            batch_support = self.batchnize_sent(support_sets)
            batch_query = self.batchnize_sent(query_sets)
            batch_query["label2tag"] = []
            for i in range(len(query_sets)):
                batch_query["label2tag"].append(query_sets[i]["label2tag"])
            return {"support": batch_support, "query": batch_query}
        else:
            batch_query = self.batchnize_sent(batch)
            batch_query["label2tag"] = []
            for i in range(len(batch)):
                batch_query["label2tag"].append(batch[i]["label2tag"])
            return batch_query

class SeqNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, encoder, max_length, debug_file=None, use_episode=True):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.encoder = encoder
        self.label2tag = None
        self.tag2label = None
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        if debug_file is not None:
            self.sampler = DebugSampler(debug_file)
        self.max_length = max_length
        self.use_episode = use_episode
        return

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
        return

    def __load_data_from_file__(self, filepath):
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
                sample = TokenSample(index, samplelines)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1

        if len(samplelines):
            sample = TokenSample(index, samplelines)
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            self.__insert_sample__(index, sample_classes)
            classes += sample_classes
        classes = list(set(classes))
        return samples, classes

    def __getraw__(self, sample):
        word, mask, word_to_piece_ind, word_to_piece_end, word_shape_ids, seq_lens = self.encoder.tokenize(sample.words)
        sent_st_id = 0
        split_seqs = []
        word_labels = []
        for cur_len in seq_lens:
            sent_ed_id = sent_st_id + cur_len
            split_seqs.append(sample.tags[sent_st_id: sent_ed_id])
            cur_word_seqs = []
            for wtag in split_seqs[-1]:
                if wtag not in self.tag2label:
                    print(wtag, self.tag2label)
                cur_word_seqs.append(self.tag2label[wtag])
            word_labels.append(cur_word_seqs)
            sent_st_id += cur_len
        item = {"word": word, "word_mask": mask, "word_to_piece_ind": word_to_piece_ind, "word_to_piece_end": word_to_piece_end,
                "seq_len": seq_lens, "word_shape_ids": word_shape_ids, "word_labels": word_labels,
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
        d['word_labels'] += item['word_labels']
        d['subsentence_num'].append(item['subsentence_num'])
        return

    def __populate__(self, idx_list, savelabeldic=False):
        dataset = {'index': [], 'word': [], 'word_mask': [], 'word_labels': [], 'word_to_piece_ind': [],
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
        assert len(dataset['word_labels']) == len(dataset['seq_len'])
        return dataset

    def __getitem__(self, index):
        target_tags, support_idx, query_idx = self.sampler.__next__(index)
        if self.use_episode:            
            self.tag2label = {tag: idx for idx, tag in enumerate(target_tags)}
            self.label2tag = {idx: tag for idx, tag in enumerate(target_tags)}
            support_set = self.__populate__(support_idx, savelabeldic=False)
            query_set = self.__populate__(query_idx, savelabeldic=True)
            return support_set, query_set
        else:
            if self.tag2label is None:
                self.tag2label = {tag: idx for idx, tag in enumerate(target_tags)}
                self.label2tag = {idx: tag for idx, tag in enumerate(target_tags)}
            return self.__populate__(support_idx + query_idx, savelabeldic=True)


    def __len__(self):
        return 1000000

    def batch_to_device(self, batch, device):
        for k, v in batch.items():
            if k == 'sentence_num' or k == 'index' or k == 'subsentence_num' or k == 'label2tag':
                continue
            batch[k] = v.to(device)
        return batch


def get_seq_loader(filepath, mode, encoder, batch_size, max_length, shuffle, debug_file=None, num_workers=8, use_episode=True):
    batcher = SeqBatcher()
    dataset = SeqNERDataset(filepath, encoder, max_length, debug_file=debug_file, use_episode=use_episode)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=lambda x: batcher(x, use_episode))
    return dataloader