import os, torch

import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from copy import deepcopy

LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
# Replace with: LABEL_LIST = [LABEL_1, LABEL_2, ..., LABEL_N, "X", "[CLS]", "[SEP]"]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.representation = None

class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    def get_examples(self, language, mode):
        """
        :param mode: one element in ['train', 'valid', 'test']
        """
        sentences = self._read_tsv(os.path.join("data", language, "{}.txt".format(mode)))
        return self._create_examples(sentences, mode) # [list of labels, list of words(no split)]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, label=label))
        return examples

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        ## read file
        # return format :
        # [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        f = open(input_file, 'r', encoding='utf-8')
        data = []
        sentence = []
        label = []
        sentence_id = 0
        for i, line in enumerate(f):
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                    sentence_id += 1
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data #, type_idxs

def compute_represenation(sents, bert_model, logger, device="cuda", reprer=None):
    if reprer is None:
        model = BertModel.from_pretrained(bert_model).to(device)
    else:
        model = reprer.model
    model.eval()
    batch_size = 100
    for i in range(0, len(sents), batch_size):
        items = sents[i : min(len(sents), i + batch_size)]
        with torch.no_grad():
            input_ids = torch.tensor([item.input_ids for item in items], dtype=torch.long).to(device)
            segment_ids = torch.tensor([item.segment_ids for item in items], dtype=torch.long).to(device)
            input_mask = torch.tensor([item.input_mask for item in items], dtype=torch.long).to(device)
            all_encoder_layers, _ = model(input_ids, segment_ids, input_mask)  # batch_size x seq_len x target_size
        layer_output = all_encoder_layers[-1].detach().cpu().numpy() # batch_size x seq_len x target_size
        for j, item in enumerate(items):
            item.representation = layer_output[j][0]
        # item.representation = layer_output
        if i % (10 * batch_size) == 0:
            logger.info('  Compute sentence representation. To {}...'.format(i))
    logger.info('  Finish.')

class Reprer():
    def __init__(self, bert_model, device="cuda"):
        self.device = device
        self.model = BertModel.from_pretrained(bert_model).to(device)

class Corpus(object):
    def __init__(self, bert_model, max_seq_length, logger, language, mode, load_data=False, support_size=-1,
                 base_features=None, mask_rate=-1.0, compute_repr=False, shuffle=True, k_shot_prop=-1.0, reprer=None):
        self.processor = NerProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

        self.label_list = LABEL_LIST
        self.num_labels = len(self.label_list) + 1 # if not +1, then the loss returned may be `nan`

        self.max_seq_length = max_seq_length
        self.language = language
        self.mode = mode
        self.logger = logger
        self.load_data = load_data
        self.mask_rate = mask_rate

        # get original feature list (do not consider [MASK] scheme)
        self.original_features = self.build_original_features(language, mode, load_data=load_data)

        if k_shot_prop > 0:
            n_tmp = len(self.original_features)
            kept_idxs = np.random.permutation(n_tmp).tolist()[: int(n_tmp * k_shot_prop)] if k_shot_prop < 1.0 else np.random.permutation(n_tmp).tolist()[: int(k_shot_prop)]
            logger.info('  The kept {}-shot-prop idxs are: {}'.format(k_shot_prop, ', '.join([str(i) for i in kept_idxs])))
            self.original_features = [self.original_features[i] for i in kept_idxs]

        # compute representations for original features (in-place operation)
        if compute_repr:
            compute_represenation(self.original_features, bert_model, logger, reprer=reprer)

        # build query set
        if mask_rate < 0: # NO [MASK] scheme
            self.query_features = self.original_features
        else:
            self.query_features = self.build_query_features_with_mask(mask_rate) # (masked)

        # build support set
        assert isinstance(support_size, int)
        if support_size > 0: # build support set (NOT masked)
            if base_features is None:
                self.support_features = self.build_support_features_(self.original_features, support_size=support_size)
            else:
                self.support_features = self.build_support_features_(base_features, support_size=support_size)

        self.n_total = len(self.query_features)
        self.batch_start_idx = 0
        self.batch_idxs = np.random.permutation(self.n_total) if shuffle else np.array([i for i in range(self.n_total)]) # for batch sampling in training


    def reset_batch_info(self, shuffle=False):
        self.batch_start_idx = 0
        self.batch_idxs = np.random.permutation(self.n_total) if shuffle else np.array([i for i in range(self.n_total)]) # for batch sampling in training


    def build_original_features(self, lang, mode, load_data=True):
        self.logger.info("Build original features for [{}-{}]...".format(lang, mode))

        # examples: a list of sentences. each item is a tuple of a list of words and a list of tags > (['words'], ['tags'])
        examples = self.processor.get_examples(lang, mode)  # 'en', 'train'

        # prepare data (max length limitation...)
        if not load_data:
            tokens_labels = self._prepare_data(examples, 'data/{}-{}-processed.txt'.format(lang, mode))
        else:
            tokens_labels = self._load_data('data/{}-{}-processed.txt'.format(lang, mode))

        # convert texts and labels to ids, add segments and mask
        features = []
        label_map = {label: i for i, label in enumerate(self.label_list, 1)}

        for item in tokens_labels:
            tokens, labels = item.split('\t|\t')
            tokens = tokens.strip().split()
            labels = labels.strip().split()

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            label_ids = [label_map[label] for label in labels]

            if len(label_ids) != len(input_ids):
                assert False
            input_mask = [1] * len(input_ids) + [0] * (self.max_seq_length - len(input_ids))
            segment_ids = [0] * len(input_ids) + [0] * (self.max_seq_length - len(input_ids))
            label_ids += [0] * (self.max_seq_length - len(input_ids))
            input_ids += [0] * (self.max_seq_length - len(input_ids))

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids))

        self.logger.info('  Num examples: {}, Num examples after split: {}'.format(len(examples), len(features)))
        return features

    def build_query_features_with_mask(self, mask_rate):
        self.logger.info("Build query features with MASK for [{}-{}]...".format(self.language, self.mode))

        assert mask_rate > 0
        features = deepcopy(self.original_features)
        mask_id = self.tokenizer.vocab['[MASK]']

        n_BIs = 0
        n_masked = 0
        for item in features:
            for i, label_id in enumerate(item.label_id):
                if label_id == 0: # [PAD] token
                    break
                label = self.label_list[label_id-1]
                if len(label) > 1 and label[1] == '-': # -: both B-XXX and I-XXX have a '-'
                    n_BIs += 1
                    if np.random.random() < mask_rate:
                        item.input_ids[i] = mask_id
                        n_masked += 1

        self.logger.info('  Masked {}/{} tokens in total.'.format(n_masked, n_BIs))

        return features

    def _prepare_data(self, examples, fn):

        def output_item(tokens, labels, res_list, fw):
            item = ' '.join(['[CLS]'] + tokens + ['[SEP]']) + '\t|\t' + ' '.join(['[CLS]'] + labels + ['[SEP]'])
            res_list.append(item)
            fw.write(item + '\n')

        fw = open(fn, 'w', encoding='utf-8')
        res = []

        for (ex_index, example) in enumerate(examples):
            textList = example.text_a
            labelList = example.label
            tokens = []
            labels = []
            for i, word in enumerate(textList):
                token = self.tokenizer.tokenize(word)
                label = [labelList[i]] + ['X'] * (len(token) - 1)
                if len(token) != len(label):
                    assert False
                tokens.extend(token)
                labels.extend(label)

            if len(tokens) >= self.max_seq_length - 1:
                tokens_ = tokens[0:(self.max_seq_length - 2)]
                labels_ = labels[0:(self.max_seq_length - 2)]
                output_item(tokens_, labels_, res, fw)

                curr_idx = self.max_seq_length - 2
                while len(tokens) >= curr_idx + self.max_seq_length // 2 - 2:
                    tokens_ = tokens[curr_idx - self.max_seq_length // 2: curr_idx + self.max_seq_length // 2 - 2]
                    labels_ = labels[curr_idx - self.max_seq_length // 2: curr_idx + self.max_seq_length // 2 - 2]
                    output_item(tokens_, labels_, res, fw)
                    curr_idx += self.max_seq_length // 2 - 2

                tokens_ = tokens[curr_idx - self.max_seq_length // 2:]
                labels_ = labels[curr_idx - self.max_seq_length // 2:]
                output_item(tokens_, labels_, res, fw)

            else:
                output_item(tokens, labels, res, fw)

        return res

    def _load_data(self, fn):
        res = []
        with open(fn, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                res.append(line)
        return res


    def get_support_ids(self, base_features, support_size=2):
        self.logger.info("Getting support feature ids for [{}-{}]...".format(self.language, self.mode))
        target_features = self.query_features

        target_reprs = np.stack([item.representation for item in target_features])
        base_reprs = np.stack([item.representation for item in base_features])  # sample_num x feature_dim

        # compute pairwise cosine distance
        dis = np.matmul(target_reprs, base_reprs.T)  # target_num x base_num

        base_norm = np.linalg.norm(base_reprs, axis=1)  # base_num
        base_norm = np.stack([base_norm] * len(target_features), axis=0)  # target_num x base_num

        dis = dis / base_norm  # target_num x base_num
        relevance = np.argsort(dis, axis=1)

        support_id_set = []
        for i, item in enumerate(target_features):
            chosen_ids = relevance[i][-1 * (support_size + 1): -1]
            if i <= 9:
                self.logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
            support_id_set.extend(chosen_ids)

        support_id_set = set(support_id_set)

        self.logger.info('  size of support ids: {}'.format(len(support_id_set)))
        return list(support_id_set)

    def reset_query_features(self, feature_ids, shuffle=True):
        self.logger.info("Reset query features of [{}-{}]...".format(self.language, self.mode))
        self.query_features = [self.original_features[i] for i in feature_ids]

        self.n_total = len(self.query_features)
        self.reset_batch_info(shuffle=shuffle)

        self.logger.info('  size of current query features: {}'.format(self.n_total))

    def build_support_features_(self, base_features, support_size=2):
        self.logger.info("Build support features for [{}-{}]...".format(self.language, self.mode))
        target_features = self.query_features

        target_reprs = np.stack([item.representation for item in target_features])
        base_reprs = np.stack([item.representation for item in base_features]) # sample_num x feature_dim

        # compute pairwise cosine distance
        dis = np.matmul(target_reprs, base_reprs.T) # target_num x base_num

        base_norm = np.linalg.norm(base_reprs, axis=1) # base_num
        base_norm = np.stack([base_norm] * len(target_features), axis=0) # target_num x base_num

        dis = dis / base_norm # target_num x base_num
        relevance = np.argsort(dis, axis=1)

        support_set = []
        for i, item in enumerate(target_features):
            chosen_ids = relevance[i][-1 * (support_size + 1) : -1]
            self.logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
            support = [base_features[id] for id in chosen_ids]
            support_set.append(support)

        return support_set


    def get_batch_meta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size >= self.n_total:
            self.reset_batch_info(shuffle=shuffle)
            if self.mask_rate >= 0:
                self.query_features = self.build_query_features_with_mask(mask_rate=self.mask_rate)


        query_batch = []
        support_batch = []
        start_id = self.batch_start_idx

        for i in range(start_id, start_id + batch_size):
            idx = self.batch_idxs[i]
            query_i = self.query_features[idx]
            query_item = {
                'input_ids': torch.tensor([query_i.input_ids], dtype=torch.long).to(device), # 1 x max_seq_len
                'input_mask': torch.tensor([query_i.input_mask], dtype=torch.long).to(device),
                'segment_ids': torch.tensor([query_i.segment_ids], dtype=torch.long).to(device),
                'label_ids': torch.tensor([query_i.label_id], dtype=torch.long).to(device) #,
                # 'flag_ids': torch.tensor([query_i.flag], dtype=torch.long).to(device)
            }
            query_batch.append(query_item)

            support_i = self.support_features[idx]
            support_item = {
                'input_ids': torch.tensor([f.input_ids for f in support_i], dtype=torch.long).to(device),
                'input_mask': torch.tensor([f.input_mask for f in support_i], dtype=torch.long).to(device),
                'segment_ids': torch.tensor([f.segment_ids for f in support_i], dtype=torch.long).to(device),
                'label_ids': torch.tensor([f.label_id for f in support_i], dtype=torch.long).to(device) #,
                # 'flag_ids': torch.tensor([f.flag for f in support_i], dtype=torch.long).to(device)
            }
            support_batch.append(support_item)

        self.batch_start_idx += batch_size

        return query_batch, support_batch

    def get_batch_NOmeta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size >= self.n_total:
            self.reset_batch_info(shuffle=shuffle)
            if self.mask_rate >= 0:
                self.query_features = self.build_query_features_with_mask(mask_rate=self.mask_rate)

        idxs = self.batch_idxs[self.batch_start_idx : self.batch_start_idx + batch_size]
        batch_features = [self.query_features[idx] for idx in idxs]
        # batch_features = self.query_features[self.batch_start_idx : self.batch_start_idx + batch_size]

        batch = {
            'input_ids': torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(device),
            'input_mask': torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(device),
            'segment_ids': torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(device),
            'label_ids': torch.tensor([f.label_id for f in batch_features], dtype=torch.long).to(device) #,
            # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
        }

        self.batch_start_idx += batch_size

        return batch

    def get_batches(self, batch_size, device="cuda", shuffle=False):
        batches = []

        if shuffle:
            idxs = np.random.permutation(self.n_total)
            features = [self.query_features[i] for i in idxs]
        else:
            features = self.query_features

        for i in range(0, self.n_total, batch_size):
            batch_features = features[i : min(self.n_total, i + batch_size)]

            batch = {
                'input_ids': torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(device),
                'input_mask': torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(device),
                'segment_ids': torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(device),
                'label_ids': torch.tensor([f.label_id for f in batch_features], dtype=torch.long).to(device) #,
                # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
            }

            batches.append(batch)

        return batches
