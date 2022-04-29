# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import bisect
import json
import logging

import numpy as np
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from utils import load_file


logger = logging.getLogger(__file__)


class EntityTypes(object):
    def __init__(self, types_path: str, negative_types_number: int, negative_mode: str):
        self.types = {}
        self.types_map = {}
        self.O_id = 0
        self.types_embedding = None
        self.negative_mode = negative_mode
        self.load_entity_types(types_path)

    def load_entity_types(self, types_path: str):
        self.types = load_file(types_path, "json")
        types_list = sorted([jj for ii in self.types.values() for jj in ii])
        self.types_map = {jj: ii for ii, jj in enumerate(types_list)}
        self.O_id = self.types_map["O"]
        self.types_list = types_list
        logger.info("Load %d entity types from %s.", len(types_list), types_path)

    def build_types_embedding(
        self,
        model: str,
        do_lower_case: bool,
        device,
        types_mode: str = "cls",
        init_type_embedding_from_bert: bool = False,
    ):
        types_list = sorted([jj for ii in self.types.values() for jj in ii])
        if init_type_embedding_from_bert:
            tokenizer = BertTokenizer.from_pretrained(
                model, do_lower_case=do_lower_case
            )

            tokens = [
                [tokenizer.cls_token_id]
                + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ii))
                + [tokenizer.sep_token_id]
                for ii in types_list
            ]
            token_max = max([len(ii) for ii in tokens])
            mask = [[1] * len(ii) + [0] * (token_max - len(ii)) for ii in tokens]
            ids = [
                ii + [tokenizer.pad_token_id] * (token_max - len(ii)) for ii in tokens
            ]
            mask = torch.tensor(np.array(mask), dtype=torch.long).to(device)
            ids = torch.tensor(np.array(ids), dtype=torch.long).to(
                device
            )  # len(type_list) x token_max

            model = BertModel.from_pretrained(model).to(device)
            outs = model(ids, mask)
        else:
            outs = [0, torch.rand((len(types_list), 768)).to(device)]
        self.types_embedding = nn.Embedding(*outs[1].shape).to(device)
        if types_mode.lower() == "cls":
            self.types_embedding.weight = nn.Parameter(outs[1])
        self.types_embedding.requires_grad = False
        logger.info("Built the types embedding.")

    def generate_negative_types(
        self, labels: list, types: list, negative_types_number: int
    ):
        N = len(labels)
        data = np.zeros((N, 1 + negative_types_number), np.int)
        if self.negative_mode == "batch":
            batch_labels = set(types)
            if negative_types_number > len(batch_labels):
                other = list(
                    set(range(len(self.types_map))) - batch_labels - set([self.O_id])
                )
                other_size = negative_types_number - len(batch_labels)
            else:
                other, other_size = [], 0
            b_size = min(len(batch_labels) - 1, negative_types_number)
            o_set = [self.O_id] if negative_types_number > len(batch_labels) - 1 else []
            for idx, l in enumerate(labels):
                data[idx][0] = l
                data[idx][1:] = np.concatenate(
                    [
                        np.random.choice(list(batch_labels - set([l])), b_size, False),
                        o_set,
                        np.random.choice(other, other_size, False),
                    ]
                )
        return data

    def get_types_embedding(self, labels: torch.Tensor):
        return self.types_embedding(labels)

    def update_type_embedding(self, e_out, e_type_ids, e_type_mask):
        labels = e_type_ids[:, :, 0][e_type_mask[:, :, 0] == 1]
        hiddens = e_out[e_type_mask[:, :, 0] == 1]
        label_set = set(labels.detach().cpu().numpy())
        for ii in label_set:
            self.types_embedding.weight.data[ii] = hiddens[labels == ii].mean(0)


class InputExample(object):
    def __init__(
        self, guid: str, words: list, labels: list, types: list, entities: list
    ):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.types = types
        self.entities = entities


class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        e_mask,
        e_type_ids,
        e_type_mask,
        types,
        entities,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.e_mask = e_mask
        self.e_type_ids = e_type_ids
        self.e_type_mask = e_type_mask
        self.types = types
        self.entities = entities


class Corpus(object):
    def __init__(
        self,
        logger,
        data_fn,
        bert_model,
        max_seq_length,
        label_list,
        entity_types: EntityTypes,
        do_lower_case=True,
        shuffle=True,
        tagging="BIO",
        viterbi="none",
        device="cuda",
        concat_types: str = "None",
        dataset: str = "FewNERD",
        negative_types_number: int = -1,
    ):
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=do_lower_case
        )
        self.max_seq_length = max_seq_length
        self.entity_types = entity_types
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.n_labels = len(label_list)
        self.tagging_scheme = tagging
        self.max_len_dict = {"entity": 0, "type": 0, "sentence": 0}
        self.max_entities_length = 50
        self.viterbi = viterbi
        self.dataset = dataset
        self.negative_types_number = negative_types_number

        logger.info(
            "Construct the transition matrix via [{}] scheme...".format(viterbi)
        )
        # M[ij]: p(j->i)
        if viterbi == "none":
            self.transition_matrix = None
            update_transition_matrix = False
        elif viterbi == "hard":
            self.transition_matrix = torch.zeros(
                [self.n_labels, self.n_labels], device=device
            )  # pij: p(j -> i)
            if self.n_labels == 3:
                self.transition_matrix[2][0] = -10000  # p(O -> I) = 0
            elif self.n_labels == 5:
                for (i, j) in [
                    (2, 0),
                    (3, 0),
                    (0, 1),
                    (1, 1),
                    (4, 1),
                    (0, 2),
                    (1, 2),
                    (4, 2),
                    (2, 3),
                    (3, 3),
                    (2, 4),
                    (3, 4),
                ]:
                    self.transition_matrix[i][j] = -10000
            else:
                raise ValueError()
            update_transition_matrix = False
        elif viterbi == "soft":
            self.transition_matrix = (
                torch.zeros([self.n_labels, self.n_labels], device=device) + 1e-8
            )
            update_transition_matrix = True
        else:
            raise ValueError()

        self.tasks = self.read_tasks_from_file(
            data_fn, update_transition_matrix, concat_types, dataset
        )

        self.n_total = len(self.tasks)
        self.batch_start_idx = 0
        self.batch_idxs = (
            np.random.permutation(self.n_total)
            if shuffle
            else np.array([i for i in range(self.n_total)])
        )  # for batch sampling in training

    def read_tasks_from_file(
        self,
        data_fn,
        update_transition_matrix=False,
        concat_types: str = "None",
        dataset: str = "FewNERD",
    ):
        """
        return: List[task]
            task['support'] = List[InputExample]
            task['query'] = List[InputExample]
        """

        self.logger.info("Reading tasks from {}...".format(data_fn))
        self.logger.info(
            "  update_transition_matrix = {}".format(update_transition_matrix)
        )
        self.logger.info("  concat_types = {}".format(concat_types))

        with open(data_fn, "r", encoding="utf-8") as json_file:
            json_list = list(json_file)
        output_tasks = []
        all_labels = [] if update_transition_matrix else None
        if dataset == "Domain":
            json_list = self._convert_Domain2FewNERD(json_list)

        for task_id, json_str in enumerate(json_list):
            if task_id % 1000 == 0:
                self.logger.info("Reading tasks %d of %d", task_id, len(json_list))
            task = json.loads(json_str) if dataset != "Domain" else json_str

            support = task["support"]
            types = task["types"]
            if self.negative_types_number == -1:
                self.negative_types_number = len(types) - 1
            tmp_support, entities, tmp_support_tokens, tmp_query_tokens = [], [], [], []
            self.max_len_dict["type"] = max(self.max_len_dict["type"], len(types))

            if concat_types != "None":
                types = set()
                for l_list in support["label"]:
                    types.update(l_list)
                types.remove("O")
                tokenized_types = self.__tokenize_types__(types, concat_types)
            else:
                tokenized_types = None

            for i, (words, labels) in enumerate(zip(support["word"], support["label"])):
                entities = self._convert_label_to_entities_(labels)
                self.max_len_dict["entity"] = max(
                    len(entities), self.max_len_dict["entity"]
                )
                if self.tagging_scheme == "BIOES":
                    labels = self._convert_label_to_BIOES_(labels)
                elif self.tagging_scheme == "BIO":
                    labels = self._convert_label_to_BIO_(labels)
                elif self.tagging_scheme == "IO":
                    labels = self._convert_label_to_IO_(labels)
                else:
                    raise ValueError("Invalid tagging scheme!")
                guid = "task[%s]-%s" % (task_id, i)
                feature, token_sum = self._convert_example_to_feature_(
                    InputExample(
                        guid=guid,
                        words=words,
                        labels=labels,
                        types=types,
                        entities=entities,
                    ),
                    tokenized_types=tokenized_types,
                    concat_types=concat_types,
                )
                tmp_support.append(feature)
                tmp_support_tokens.append(token_sum)
                if update_transition_matrix:
                    all_labels.append(labels)

            query = task["query"]
            tmp_query = []
            for i, (words, labels) in enumerate(zip(query["word"], query["label"])):
                entities = self._convert_label_to_entities_(labels)
                self.max_len_dict["entity"] = max(
                    len(entities), self.max_len_dict["entity"]
                )
                if self.tagging_scheme == "BIOES":
                    labels = self._convert_label_to_BIOES_(labels)
                elif self.tagging_scheme == "BIO":
                    labels = self._convert_label_to_BIO_(labels)
                elif self.tagging_scheme == "IO":
                    labels = self._convert_label_to_IO_(labels)
                else:
                    raise ValueError("Invalid tagging scheme!")
                guid = "task[%s]-%s" % (task_id, i)
                feature, token_sum = self._convert_example_to_feature_(
                    InputExample(
                        guid=guid,
                        words=words,
                        labels=labels,
                        types=types,
                        entities=entities,
                    ),
                    tokenized_types=tokenized_types,
                    concat_types=concat_types,
                )
                tmp_query.append(feature)
                tmp_query_tokens.append(token_sum)
                if update_transition_matrix:
                    all_labels.append(labels)

            output_tasks.append(
                {
                    "support": tmp_support,
                    "query": tmp_query,
                    "support_token": tmp_support_tokens,
                    "query_token": tmp_query_tokens,
                }
            )
        self.logger.info(
            "%s Max Entities Lengths: %d, Max batch Types Number: %d, Max sentence Length: %d",
            data_fn,
            self.max_len_dict["entity"],
            self.max_len_dict["type"],
            self.max_len_dict["sentence"],
        )
        if update_transition_matrix:
            self._count_transition_matrix_(all_labels)
        return output_tasks

    def _convert_Domain2FewNERD(self, data: list):
        def decode_batch(batch: dict):
            word = batch["seq_ins"]
            label = [
                [jj.replace("B-", "").replace("I-", "") for jj in ii]
                for ii in batch["seq_outs"]
            ]
            return {"word": word, "label": label}

        data = json.loads(data[0])
        res = []
        for domain in data.keys():
            d = data[domain]
            labels = self.entity_types.types[domain]
            res.extend(
                [
                    {
                        "support": decode_batch(ii["support"]),
                        "query": decode_batch(ii["batch"]),
                        "types": labels,
                    }
                    for ii in d
                ]
            )
        return res

    def __tokenize_types__(self, types, concat_types: str = "past"):
        tokens = []
        for t in types:
            if "embedding" in concat_types:
                t_tokens = [f"[unused{self.entity_types.types_map[t]}]"]
            else:
                t_tokens = self.tokenizer.tokenize(t)
            if len(t_tokens) == 0:
                continue
            tokens.extend(t_tokens)
            tokens.append(",")  # separate different types with a comma ','.
        tokens.pop()  # pop the last comma
        return tokens

    def _count_transition_matrix_(self, labels):
        self.logger.info("Computing transition matrix...")
        for sent_labels in labels:
            for i in range(len(sent_labels) - 1):
                start = self.label_map[sent_labels[i]]
                end = self.label_map[sent_labels[i + 1]]
                self.transition_matrix[end][start] += 1
        self.transition_matrix /= torch.sum(self.transition_matrix, dim=0)
        self.transition_matrix = torch.log(self.transition_matrix)
        self.logger.info("Done.")

    def _convert_label_to_entities_(self, label_list: list):
        N = len(label_list)
        S = [
            ii
            for ii in range(N)
            if label_list[ii] != "O"
            and (not ii or label_list[ii] != label_list[ii - 1])
        ]
        E = [
            ii
            for ii in range(N)
            if label_list[ii] != "O"
            and (ii == N - 1 or label_list[ii] != label_list[ii + 1])
        ]
        return [(s, e, label_list[s]) for s, e in zip(S, E)]

    def _convert_label_to_BIOES_(self, label_list):
        res = []
        label_list = ["O"] + label_list + ["O"]
        for i in range(1, len(label_list) - 1):
            if label_list[i] == "O":
                res.append("O")
                continue
            # for S
            if (
                label_list[i] != label_list[i - 1]
                and label_list[i] != label_list[i + 1]
            ):
                res.append("S")
            elif (
                label_list[i] != label_list[i - 1]
                and label_list[i] == label_list[i + 1]
            ):
                res.append("B")
            elif (
                label_list[i] == label_list[i - 1]
                and label_list[i] != label_list[i + 1]
            ):
                res.append("E")
            elif (
                label_list[i] == label_list[i - 1]
                and label_list[i] == label_list[i + 1]
            ):
                res.append("I")
            else:
                raise ValueError("Some bugs exist in your code!")
        return res

    def _convert_label_to_BIO_(self, label_list):
        precursor = ""
        label_output = []
        for label in label_list:
            if label == "O":
                label_output.append("O")
            elif label != precursor:
                label_output.append("B")
            else:
                label_output.append("I")
            precursor = label

        return label_output

    def _convert_label_to_IO_(self, label_list):
        label_output = []
        for label in label_list:
            if label == "O":
                label_output.append("O")
            else:
                label_output.append("I")

        return label_output

    def _convert_example_to_feature_(
        self,
        example,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
        sequence_b_segment_id=1,
        tokenized_types=None,
        concat_types: str = "None",
    ):
        """
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        tokens, label_ids, token_sum = [], [], [1]
        if tokenized_types is None:
            tokenized_types = []
        if "before" in concat_types:
            token_sum[-1] += 1 + len(tokenized_types)

        for words, labels in zip(example.words, example.labels):
            word_tokens = self.tokenizer.tokenize(words)
            token_sum.append(token_sum[-1] + len(word_tokens))
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [self.label_map[labels]]
                + [ignore_token_label_id] * (len(word_tokens) - 1)
            )
        self.max_len_dict["sentence"] = max(self.max_len_dict["sentence"], len(tokens))
        e_ids = [(token_sum[s], token_sum[e + 1] - 1) for s, e, _ in example.entities]
        e_mask = np.zeros((self.max_entities_length, self.max_seq_length), np.int8)
        e_type_mask = np.zeros(
            (self.max_entities_length, 1 + self.negative_types_number), np.int8
        )
        e_type_mask[: len(e_ids), :] = np.ones(
            (len(e_ids), 1 + self.negative_types_number), np.int8
        )
        for idx, (s, e) in enumerate(e_ids):
            e_mask[idx][s : e + 1] = 1
        e_type_ids = [self.entity_types.types_map[t] for _, _, t in example.entities]
        entities = [(s, e, t) for (s, e), t in zip(e_ids, e_type_ids)]
        batch_types = [self.entity_types.types_map[ii] for ii in example.types]
        # e_type_ids[i, 0] is the possitive label, while e_type_ids[i, 1:] are negative labels
        e_type_ids = self.entity_types.generate_negative_types(
            e_type_ids, batch_types, self.negative_types_number
        )
        if len(e_type_ids) < self.max_entities_length:
            e_type_ids = np.concatenate(
                [
                    e_type_ids,
                    [[0] * (1 + self.negative_types_number)]
                    * (self.max_entities_length - len(e_type_ids)),
                ]
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > self.max_seq_length - special_tokens_count - len(
            tokenized_types
        ):
            tokens = tokens[
                : (self.max_seq_length - special_tokens_count - len(tokenized_types))
            ]
            label_ids = label_ids[
                : (self.max_seq_length - special_tokens_count - len(tokenized_types))
            ]

        types = [self.entity_types.types_map[t] for t in example.types]

        if "before" in concat_types:
            # OPTION 1: Concatenated tokenized types at START
            len_sentence = len(tokens)
            tokens = [cls_token] + tokenized_types + [sep_token] + tokens
            label_ids = [ignore_token_label_id] * (len(tokenized_types) + 2) + label_ids
            segment_ids = (
                [cls_token_segment_id]
                + [sequence_a_segment_id] * (len(tokenized_types) + 1)
                + [sequence_b_segment_id] * len_sentence
            )
        else:
            # OPTION 2: Concatenated tokenized types at END
            tokens += [sep_token]
            label_ids += [ignore_token_label_id]
            if sep_token_extra:
                raise ValueError("Unexpected path!")
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [ignore_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                raise ValueError("Unexpected path!")
                tokens += [cls_token]
                label_ids += [ignore_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [ignore_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            if "past" in concat_types:
                tokens += tokenized_types
                label_ids += [ignore_token_label_id] * len(tokenized_types)
                segment_ids += [sequence_b_segment_id] * len(tokenized_types)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length

        return (
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                e_mask=e_mask,  # max_entities_length x 128
                e_type_ids=e_type_ids,  # max_entities_length x 5 (n_types)
                e_type_mask=np.array(e_type_mask),  # max_entities_length x 5 (n_types)
                types=np.array(types),
                entities=entities,
            ),
            token_sum,
        )

    def reset_batch_info(self, shuffle=False):
        self.batch_start_idx = 0
        self.batch_idxs = (
            np.random.permutation(self.n_total)
            if shuffle
            else np.array([i for i in range(self.n_total)])
        )  # for batch sampling in training

    def get_batch_meta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size > self.n_total:
            self.reset_batch_info(shuffle=shuffle)

        query_batch = []
        support_batch = []
        start_id = self.batch_start_idx

        for i in range(start_id, start_id + batch_size):
            idx = self.batch_idxs[i]
            task_curr = self.tasks[idx]

            query_item = {
                "input_ids": torch.tensor(
                    [f.input_ids for f in task_curr["query"]], dtype=torch.long
                ).to(
                    device
                ),  # 1 x max_seq_len
                "input_mask": torch.tensor(
                    [f.input_mask for f in task_curr["query"]], dtype=torch.long
                ).to(device),
                "segment_ids": torch.tensor(
                    [f.segment_ids for f in task_curr["query"]], dtype=torch.long
                ).to(device),
                "label_ids": torch.tensor(
                    [f.label_ids for f in task_curr["query"]], dtype=torch.long
                ).to(device),
                "e_mask": torch.tensor(
                    [f.e_mask for f in task_curr["query"]], dtype=torch.int
                ).to(device),
                "e_type_ids": torch.tensor(
                    [f.e_type_ids for f in task_curr["query"]], dtype=torch.long
                ).to(device),
                "e_type_mask": torch.tensor(
                    [f.e_type_mask for f in task_curr["query"]], dtype=torch.int
                ).to(device),
                "types": [f.types for f in task_curr["query"]],
                "entities": [f.entities for f in task_curr["query"]],
                "idx": idx,
            }
            query_batch.append(query_item)

            support_item = {
                "input_ids": torch.tensor(
                    [f.input_ids for f in task_curr["support"]], dtype=torch.long
                ).to(device),
                # 1 x max_seq_len
                "input_mask": torch.tensor(
                    [f.input_mask for f in task_curr["support"]], dtype=torch.long
                ).to(device),
                "segment_ids": torch.tensor(
                    [f.segment_ids for f in task_curr["support"]], dtype=torch.long
                ).to(device),
                "label_ids": torch.tensor(
                    [f.label_ids for f in task_curr["support"]], dtype=torch.long
                ).to(device),
                "e_mask": torch.tensor(
                    [f.e_mask for f in task_curr["support"]], dtype=torch.int
                ).to(device),
                "e_type_ids": torch.tensor(
                    [f.e_type_ids for f in task_curr["support"]], dtype=torch.long
                ).to(device),
                "e_type_mask": torch.tensor(
                    [f.e_type_mask for f in task_curr["support"]], dtype=torch.int
                ).to(device),
                "types": [f.types for f in task_curr["support"]],
                "entities": [f.entities for f in task_curr["support"]],
                "idx": idx,
            }
            support_batch.append(support_item)

        self.batch_start_idx += batch_size

        return query_batch, support_batch

    def get_batch_NOmeta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size >= self.n_total:
            self.reset_batch_info(shuffle=shuffle)
            if self.mask_rate >= 0:
                self.query_features = self.build_query_features_with_mask(
                    mask_rate=self.mask_rate
                )

        idxs = self.batch_idxs[self.batch_start_idx : self.batch_start_idx + batch_size]
        batch_features = [self.query_features[idx] for idx in idxs]

        batch = {
            "input_ids": torch.tensor(
                [f.input_ids for f in batch_features], dtype=torch.long
            ).to(device),
            "input_mask": torch.tensor(
                [f.input_mask for f in batch_features], dtype=torch.long
            ).to(device),
            "segment_ids": torch.tensor(
                [f.segment_ids for f in batch_features], dtype=torch.long
            ).to(device),
            "label_ids": torch.tensor(
                [f.label_id for f in batch_features], dtype=torch.long
            ).to(device),
            "e_mask": torch.tensor(
                [f.e_mask for f in batch_features], dtype=torch.int
            ).to(device),
            "e_type_ids": torch.tensor(
                [f.e_type_ids for f in batch_features], dtype=torch.long
            ).to(device),
            "e_type_mask": torch.tensor(
                [f.e_type_mask for f in batch_features], dtype=torch.int
            ).to(device),
            "types": [f.types for f in batch_features],
            "entities": [f.entities for f in batch_features],
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
                "input_ids": torch.tensor(
                    [f.input_ids for f in batch_features], dtype=torch.long
                ).to(device),
                "input_mask": torch.tensor(
                    [f.input_mask for f in batch_features], dtype=torch.long
                ).to(device),
                "segment_ids": torch.tensor(
                    [f.segment_ids for f in batch_features], dtype=torch.long
                ).to(device),
                "label_ids": torch.tensor(
                    [f.label_id for f in batch_features], dtype=torch.long
                ).to(device),
                "e_mask": torch.tensor(
                    [f.e_mask for f in batch_features], dtype=torch.int
                ).to(device),
                "e_type_ids": torch.tensor(
                    [f.e_type_ids for f in batch_features], dtype=torch.long
                ).to(device),
                "e_type_mask": torch.tensor(
                    [f.e_type_mask for f in batch_features], dtype=torch.int
                ).to(device),
                "types": [f.types for f in batch_features],
                "entities": [f.entities for f in batch_features],
            }

            batches.append(batch)

        return batches

    def _decoder_bpe_index(self, sentences_spans: list):
        res = []
        tokens = [jj for ii in self.tasks for jj in ii["query_token"]]
        assert len(tokens) == len(
            sentences_spans
        ), f"tokens size: {len(tokens)}, sentences size: {len(sentences_spans)}"
        for sentence_idx, spans in enumerate(sentences_spans):
            token = tokens[sentence_idx]
            tmp = []
            for b, e in spans:
                nb = bisect.bisect_left(token, b)
                ne = bisect.bisect_left(token, e)
                tmp.append((nb, ne))
            res.append(tmp)
        return res


if __name__ == "__main__":
    pass
