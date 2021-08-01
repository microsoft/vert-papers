# coding=utf-8
# Refer to HuggingFace's transformers.

from __future__ import absolute_import, division, print_function

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler, TensorDataset

from seqeval.metrics import f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid: int, words: list, labels: list):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    def __init__(
        self, input_ids: list, input_mask: list, segment_ids: list, label_ids: list
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def evaluate_viterbi(
    args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", add_eval=False
):
    eval_dataset = load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id, mode=mode
    )

    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation with Viterbi Decoding %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    viterbi_decoder = ViterbiDecoder(labels, pad_token_label_id, args.device)
    out_label_ids = None
    pred_label_list = []

    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
                # XLM and RoBERTa don"t use segment_ids
                "labels": batch[3],
            }
            if add_eval:
                inputs["eval"] = True
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # decode with viterbi
        log_probs = torch.nn.functional.log_softmax(
            logits.detach(), dim=-1
        )  # batch_size x max_seq_len x n_labels

        pred_labels = viterbi_decoder.forward(log_probs, batch[1], batch[3])
        pred_label_list.extend(pred_labels)

        if out_label_ids is None:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps

    label_map = {i: label for i, label in enumerate(labels)}
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    if out_label_ids.shape[0] != len(pred_label_list):
        raise ValueError("Num of examples doesn't match!")

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
        if len(out_label_list[i]) != len(pred_label_list[i]):
            raise ValueError("Sequence length doesn't match!")

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, pred_label_list) * 100,
        "recall": recall_score(out_label_list, pred_label_list) * 100,
        "f1": f1_score(out_label_list, pred_label_list) * 100,
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info(f"  {key} = {results[key]}")

    return results, pred_label_list


def load_and_cache_examples(
    args, tokenizer, labels, pad_token_label_id, mode, data_dir=None
):
    if data_dir is None:
        data_dir = args.data_dir
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {data_dir}")
        examples = read_examples_from_file(data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["unicoder"]),
            # unicoder has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["unicoder"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["unicoder"]),
            # pad on the left for unicoder
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["unicoder"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        logger.info(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    return dataset


def non_en_dataset(
    args,
    total_num: int,
    tokenizer,
    labels: list,
    pad_token_label_id: int,
    tgt_langs: str,
):
    datasets = []
    for lang in tgt_langs.split():
        file = args.data_dir + lang
        dataset = load_and_cache_examples(
            args, tokenizer, labels, pad_token_label_id, "train", data_dir=file
        )
        dataset = get_shuffle_dataset(dataset, int(total_num / len(tgt_langs.split())))
        datasets.append(dataset)
    return ConcatDataset(datasets)


def get_shuffle_dataset(dataset, num):
    index = [i for i in range(0, dataset.__len__())]
    random.shuffle(index)

    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = [], [], [], []
    end = min(num, dataset.__len__())
    for i in range(0, end):
        all_input_ids.append(dataset.tensors[0][index[i]])
        all_input_mask.append(dataset.tensors[1][index[i]])
        all_segment_ids.append(dataset.tensors[2][index[i]])
        all_label_ids.append(dataset.tensors[3][index[i]])
    all_input_ids = torch.stack(all_input_ids)
    all_input_mask = torch.stack(all_input_mask)
    all_segment_ids = torch.stack(all_segment_ids)
    all_label_ids = torch.stack(all_label_ids)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info("  The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info(f"    Freeze: {n}")
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return outputs


def write_prediction(
    predictions: list, predictions_path: str, data_path: str, encoding: str = "utf-8"
):
    with open(predictions_path, "w", encoding=encoding) as writer:
        with open(data_path, "r", encoding=encoding) as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = "{} {} {}\n".format(
                        line.split()[0],
                        line.split()[-1].replace("\n", ""),
                        predictions[example_id].pop(0),
                    )
                    writer.write(output_line)
                else:
                    logger.warning(
                        f"Maximum sequence length exceeded: No prediction for '{line.split()[0]}'.",
                    )


def shuffle_word_embedding(seq: list, seq_end: int):
    index = [i for i in range(0, seq_end)]
    random.shuffle(index)
    seq_shuffle = [seq[ii] for ii in index]
    if len(seq_shuffle) < len(seq):
        seq_shuffle += [seq[ii] for ii in range(seq_end, len(seq))]
    return torch.stack(seq_shuffle)


class ViterbiDecoder:
    def __init__(self, labels: list, pad_token_label_id: int, device):
        self.n_labels = len(labels)
        self.pad_token_label_id = pad_token_label_id
        self.label_map = {i: label for i, label in enumerate(labels)}

        self.transitions = torch.zeros([self.n_labels, self.n_labels], device=device)
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                if labels[i][0] == "I" and labels[j][-3:] != labels[i][-3:]:
                    self.transitions[i, j] = -10000

    def forward(self, logprobs, attention_mask, label_ids):
        active_tokens = (attention_mask == 1) & (label_ids != self.pad_token_label_id)

        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        # scores = []
        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][
                active_tokens[idx]
            ]  # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0]  # n_labels

            for j in range(1, len(logprob_i)):  # for tag_feat in feat:
                next_label_var = forward_var + self.transitions  # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)  # n_labels

                logp_j = logprob_i[j]  # n_labels
                forward_var = viterbivars_t + logp_j  # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            # terminal_var = forward_var

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            # path_score = path_score.item()
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append([self.label_map[label_id] for label_id in best_path])
            # scores.append(path_score)

        return label_seqs  # , scores


def shuffle_input(seqs_id, mask, en, generator=False, device=None):
    # shuffle words of sentences
    seqs, labels = [], []
    for seq_index in range(seqs_id.size(0)):
        if random.choice([True, False]):
            seq_end = torch.sum(mask[seq_index])
            seq = shuffle_word_embedding(seqs_id[seq_index], seq_end)
            label = [0.0, 1.0]
        else:
            seq = seqs_id[seq_index]
            label = [1.0, 0.0]

        if generator:
            label = [label[1], label[0]]
        label = torch.Tensor([label] * seqs_id.size(1))
        seqs.append(seq)
        labels.append(label)
    seqs = torch.stack(seqs).to(device)
    labels = torch.stack(labels).to(device)

    return seqs, labels


def get_labels(path: str, encoding: str = "utf-8"):
    if path:
        with open(path, "r", encoding=encoding) as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        while "" in labels:
            labels.remove("")
        return labels
    else:
        return [
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]


def read_examples_from_file(data_dir: str, mode: str, encoding: str = "utf-8"):
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding=encoding) as f:
        words, labels = [], []
        for line in f:
            line = line.strip().replace("\t", " ")
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid=f"{mode}-{guid_index}",
                            words=words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words, labels = [], []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d" % (mode, guid_index), words=words, labels=labels
                )
            )
    return examples


def convert_examples_to_features(
    examples,
    label_list: list,
    max_seq_length: int,
    tokenizer,
    cls_token_at_end: bool = False,
    cls_token: str = "[CLS]",
    cls_token_segment_id: int = 1,
    sep_token: str = "[SEP]",
    sep_token_extra: bool = False,
    pad_on_left: bool = False,
    pad_token: int = 0,
    pad_token_segment_id: int = 0,
    pad_token_label_id: int = -1,
    sequence_a_segment_id: int = 0,
    mask_padding_with_zero: bool = True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (unicoder/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for unicoder)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(example)}")

        tokens, label_ids = [], []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
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

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def torch_device(gpu_id: int) -> torch.device:
    """ allocation torch device """
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    logger.info(f"torch device: {device}")
    return device


def mkdir(output_dir: str):
    """ mkdir file dir"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
