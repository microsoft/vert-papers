# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import pickle
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm, trange
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file, get_pij_en_valid, update_encoding

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer#, BertForTokenClassification
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from modeling import BertForTokenClassification_, ViterbiDecoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification_, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info("  The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info("    Freeze: %s", n)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]

    return outputs

def train(args, model, train_dataset, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    opt_params = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(opt_params, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total * args.warmup_ratio), t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  GPU IDs for training: %s", " ".join([str(id) for id in args.gpu_ids]))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_i in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0].to(args.device),
                      "attention_mask": batch[1].to(args.device),
                      "token_type_ids": batch[2].to(args.device) if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3].to(args.device)}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("===== evaluate_during_training =====")
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info("Epoch: {}\t global_step: {}\t eval_{}: {}".format(epoch_i, global_step, key, value))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Epoch: {}\t global_step: {}\t learning rate: {:.8}\t loss: {:.4f}".format(
                            epoch_i, global_step, scheduler.get_lr()[0],
                            (tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()

    return global_step, tr_loss / global_step

def train_KD(args, model, train_dataset, tokenizer, labels, pad_token_label_id): #src_probs, tokenizer,
    """ Train the model """
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    opt_params = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(opt_params, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(opt_params, lr=args.learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total * args.warmup_ratio), t_total=t_total)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  GPU IDs for training: %s", " ".join([str(id) for id in args.gpu_ids]))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_KD, logging_loss_KD = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_i in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0].to(args.device),
                      "attention_mask": batch[1].to(args.device),
                      "token_type_ids": batch[2].to(args.device) if args.model_type in ["bert", "xlnet"] else None,
                      "labels": batch[3].to(args.device),
                      "active_CE": batch[4].to(args.device) if len(batch) == 7 else None,
                      "pseudo_labels": batch[5] if len(batch) == 7 else None,
                      "src_probs": batch[-1],

                      "lambda_original_loss": args.lambda_original_loss,
                      "loss_with_crossEntropy": args.loss_with_crossEntropy,
                      "weight_crossEntropy": args.weight_crossEntropy
                      } # activate the KD loss

            outputs = model(**inputs)
            # loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss_KD, loss = outputs[:2]

            if loss_KD is None:
                logger.info("  ==> Step {}: loss is None.".format(global_step))
                continue

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss_KD = loss_KD.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss_KD = loss_KD / args.gradient_accumulation_steps

            # loss.backward()
            loss_KD.backward()

            tr_loss += loss.item()
            tr_loss_KD += loss_KD.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("===== evaluate_during_training =====")
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info("Epoch: {}\t global_step: {}\t eval_{}: {}".format(epoch_i, global_step, key, value))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar("loss_KD", (tr_loss_KD - logging_loss_KD) / args.logging_steps, global_step)
                    logger.info(
                        "Epoch: {}\t global_step: {}\t learning rate: {:.8}\t loss: {:.4f}\t loss_KD: {:.4f}".format(
                            epoch_i, global_step, scheduler.get_lr()[0],
                            (tr_loss - logging_loss) / args.logging_steps,
                            (tr_loss_KD - logging_loss_KD) / args.logging_steps))
                    logging_loss = tr_loss
                    logging_loss_KD = tr_loss_KD

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()

    return global_step, tr_loss_KD / global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def evaluate_viterbi(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # decode with viterbi
        log_probs = torch.nn.functional.log_softmax(logits.detach(), dim=-1) # batch_size x max_seq_len x n_labels

        pred_labels = viterbi_decoder.forward(log_probs,  batch[1], batch[3])
        pred_label_list.extend(pred_labels)

        if out_label_ids is None:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

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
        "f1": f1_score(out_label_list, pred_label_list) * 100
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, pred_label_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, data_dir=None):
    if data_dir is None:
        data_dir = args.data_dir
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, "cached_{}_{}_{}".format(mode, list(filter(None,
                                            args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = read_examples_from_file(data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

def prepare_dataset_token_level_combine(args, tokenizer, labels, pad_token_label_id, model_class):
    dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")

    src_probs = get_src_probs(args, dataset, model_class, args.src_model_path, src_dropout=False)
    pseudo_labels = torch.argmax(src_probs, dim=-1) # dataset_len x max_seq_len

    activate_CE = None
    if not isinstance(args.src_model_path_assist, list):
        raise ValueError("Invalid `src_model_path_assist`.")

    for path_tmp in args.src_model_path_assist:
        src_probs_tmp = get_src_probs(args, dataset, model_class, path_tmp, src_dropout=False)
        pseudo_labels_tmp = torch.argmax(src_probs_tmp, dim=-1)  # dataset_len x max_seq_len
        if activate_CE is None:
            activate_CE = pseudo_labels == pseudo_labels_tmp
        else:
            activate_CE &= pseudo_labels == pseudo_labels_tmp

    dataset.tensors += (activate_CE, pseudo_labels, src_probs)

    return dataset

def prepare_dataset_token_level_combine_ensemble(args, tokenizer, labels, pad_token_label_id, model_class):
    dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")

    src_probs = get_ensembled_src_probs(args, dataset, model_class, src_model_type="combine")
    pseudo_labels = torch.argmax(src_probs, dim=-1) # dataset_len x max_seq_len

    activate_CE = None
    for src_model_type in ["en", "trans"]:
        src_probs_tmp = get_ensembled_src_probs(args, dataset, model_class, src_model_type=src_model_type)
        pseudo_labels_tmp = torch.argmax(src_probs_tmp, dim=-1)
        if activate_CE is None:
            activate_CE = pseudo_labels == pseudo_labels_tmp
        else:
            activate_CE &= pseudo_labels == pseudo_labels_tmp

    dataset.tensors += (activate_CE, pseudo_labels, src_probs)

    return dataset

def get_ensembled_src_probs(args, dataset, model_class, src_model_type):
    if len(args.ensemble_seeds) <= 1:
        raise ValueError("Please check ensemble seeds!")
    tgt_lang = os.path.basename(args.data_dir)
    src_probs = None

    src_probs = None
    seeds_tmp = [22, 122, 283, 361, 649, 705, 854, 975]
    seeds_tmp.remove(args.seed)
    ensemble_seeds = np.random.choice(seeds_tmp, args.n_ensemble-1, replace=False).tolist()
    ensemble_seeds.append(args.seed)

    for s in args.ensemble_seeds:
        if src_model_type == "combine":
            if tgt_lang == 'de':
                src_model_path = 'result-20200106-C1-D1/result-{}/finetune-{}-{}-lr_5e-5'.format(s, s, tgt_lang)
            else:
                src_model_path = 'result-20200106-C1-D1/result-{}/finetune-trans-{}-lr_5e-5'.format(s, tgt_lang)
        elif src_model_type == "en":
            src_model_path = '../ts-enhanced-loss/conll-model-{}/mono-src-en'.format(s)
        elif src_model_type == "trans":
            src_model_path = 'result-20200118-finetune-5e-5/result-{}/ptrans-{}'.format(s, tgt_lang)
        else:
            raise ValueError("Invalid `src_modle_type`.")

        if src_probs is None:
            src_probs = get_src_probs(args, dataset, model_class, src_model_path, src_dropout=False)
        else:
            src_probs += get_src_probs(args, dataset, model_class, src_model_path, src_dropout=False)

    src_probs /= len(args.ensemble_seeds)

    return src_probs

def get_src_probs(args, dataset, model_class, src_model_path, src_dropout=False):#, src_lang):
    """ without softmax.
        preds: dataset_len x seq_len x label_len
    """
    # load src model
    src_model = model_class.from_pretrained(src_model_path)
    src_model.to(args.device)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info("***** Compute logits for [%s] dataset using the model [%s] *****", os.path.basename(args.data_dir), src_model_path)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    if src_dropout:
        logger.info("  Setting src_model.train().")
        src_model.train()
    else:
        logger.info("  Setting src_model.eval().")
        src_model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      "labels": None} #batch[3]}
            outputs = src_model(**inputs)
            logits = outputs[0]

        # nb_eval_steps += 1
        preds = logits.detach() if preds is None else torch.cat((preds, logits.detach()), dim=0) # dataset_len x max_seq_len x label_len

    preds = torch.nn.functional.softmax(preds, dim=-1)

    return preds #, eval_loss


def save_model(args, model, tokenizer):
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./data/ner/conll/debug", type=str,  # required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--output_dir", default="conll-model/test", type=str,  # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--src_model_path", default="", type=str,
                        help="path to load teacher models")

    parser.add_argument("--do_filter_token", action="store_true",
                        help="Whether to filter the cross entropy loss by M_src and M_trans.")
    # parser.add_argument("--src_model_path_assist", type=str, default="",
    parser.add_argument("--src_model_path_assist", type=str, nargs="+",
                        help="path to load the assisting teacher model. default: not use.")
    parser.add_argument("--alpha", default=-1.0, type=float,
                        help=" alpha * p_src + (1 - alpha) * p_trans")
    parser.add_argument("--ensemble_seeds", type=int, nargs="+",
                        help="random seeds for teacher ensemble.")

    parser.add_argument("--combine_dataset", action="store_true",
                        help="Whether to combine the translated data with the unlabeled target data.")
    parser.add_argument("--translation_dir", default="", type=str,
                        help="The translation data dir.")

    parser.add_argument("--lambda_original_loss", type=float, default=1.0,
                        help="")
    parser.add_argument("--loss_with_crossEntropy", action="store_true", #
                        help="Whether to add the cross entropy loss with the KD MSE loss.")
    parser.add_argument("--weight_crossEntropy", action="store_true", #
                        help="Whether to weight the Cross Entropy loss with the corresponding soft-label probability.")

    ## Other parameters
    parser.add_argument("--model_type", default='bert', type=str,  # required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--encoding", default='utf-8', type=str,  # required=True,
                        help="open encoding")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,  # required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--labels", default="./data/ner/conll/labels.txt", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--use_KD", action="store_true",
                        help="Whether to train with knowledge distillation.")
    parser.add_argument("--do_finetune", action="store_true",
                        help="Whether to finetune a trained NER model.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--use_viterbi", action="store_true",
                        help="Whether to decode with Viterbi.")
    parser.add_argument("--mode", type=str, default="test",
                        help="The mode (dataset type) used for prediction. Only work when do_predict is True.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--freeze_bottom_layer", default=3, type=int,
                        help="Freeze the bottom n layers of the model during fine-tuning.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument("--logging_steps", type=int, default=20,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=667,
                        help="random seed for initialization")
    parser.add_argument("--gpu_ids", type=int, nargs="+",
                        help="ids of the gpus to use")

    args = parser.parse_args()

    # Check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and (args.do_train or args.do_finetune) and os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")) and os.path.basename(args.output_dir) != "test":
        raise ValueError("Train: Output directory already exists and is not empty.")

    if os.path.exists(args.output_dir) and args.do_predict:
        is_done = False
        for name in os.listdir(args.output_dir): # result file: "test_results-TIME-LANGUAGE"
            if ("test_results{}-2020".format("-viterbi" if args.use_viterbi else "")) in name and (os.path.basename(args.data_dir) + ".txt") in name:
                is_done = True
                break
        # if is_done:
        #     raise ValueError("Predict: Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.n_gpu = len(args.gpu_ids)  # torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids[0])
    device = torch.device("cuda")
    args.device = device
    
    update_encoding(args.encoding)

    # Setup logging
    if args.do_train and not (os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir)
    args.log_dir = os.path.join(args.output_dir, "logs")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_name = "log-{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if args.do_train:
        log_name += "-train"
    if args.do_predict:
        log_name += "-predict"
    log_name += "-{}.txt".format(os.path.basename(args.data_dir))
    fh = logging.FileHandler(os.path.join(args.log_dir, log_name), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    args.model_type = args.model_type.lower()
    logger.info("Training/evaluation parameters %s", args)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index  # -100 here

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Training
    if args.do_train:
        # load target model
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
        # initial_model_path = ""
        # logger.info("Load initial model from [{}]...".format(initial_model_path))
        # model = model_class.from_pretrained(initial_model_path)

        model.to(args.device)

        if args.use_KD:
            logger.info("********** scheme: training with KD **********")

            if args.do_filter_token:
                if args.ensemble_seeds is not None and len(args.ensemble_seeds) >= 1:
                    train_dataset = prepare_dataset_token_level_combine_ensemble(args, tokenizer, labels, pad_token_label_id, model_class)
                else:
                    train_dataset = prepare_dataset_token_level_combine(args, tokenizer, labels, pad_token_label_id, model_class)
            else:
                train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
                src_probs = get_src_probs(args, train_dataset, model_class, args.src_model_path, src_dropout=False)
                train_dataset.tensors += (src_probs,)

            # Train!
            global_step, tr_loss_KD, tr_loss = train_KD(args, model, train_dataset, tokenizer, labels, pad_token_label_id)
            logger.info(" global_step = %s, average KD loss = %s, average loss = %s", global_step, tr_loss_KD, tr_loss)

        else:
            logger.info("********** scheme: training without KD **********")
            if args.combine_dataset: # combine src and translation
                dataset_src = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", data_dir=args.data_dir)
                dataset_trans = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", data_dir=args.translation_dir)
                train_dataset = torch.utils.data.ConcatDataset([dataset_src, dataset_trans])
            else:
                train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")

            global_step, tr_loss = train(args, model, train_dataset, tokenizer, labels, pad_token_label_id)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        save_model(args, model, tokenizer)


    if args.do_finetune:
        logger.info("********** scheme: fine-tune **********")
        logger.info("Loading tokenizer and model from {}...".format(args.src_model_path))

        tokenizer = tokenizer_class.from_pretrained(args.src_model_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.src_model_path)
        model.to(args.device)

        # prepare target training plain text
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")

        global_step, tr_loss = train(args, model, train_dataset, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        save_model(args, model, tokenizer)


    if args.do_predict:
        logger.info("********** scheme: prediction **********")
        predict_model_path = args.output_dir if args.src_model_path == "" else args.src_model_path
        logger.info("Loading tokenizer and model from {}...".format(predict_model_path))

        tokenizer = tokenizer_class.from_pretrained(predict_model_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(predict_model_path)
        model.to(args.device)
        if args.use_viterbi:
            result, predictions = evaluate_viterbi(args, model, tokenizer, labels, pad_token_label_id, mode=args.mode)
        else:
            result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode=args.mode)
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results{}-{}-{}-{}.txt".format(
            "-viterbi" if args.use_viterbi else "",
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            args.mode,
            os.path.basename(args.data_dir)))
        with open(output_test_results_file, "w", encoding=args.encoding) as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions{}-{}-{}-{}.txt".format(
            "-viterbi" if args.use_viterbi else "",
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            args.mode,
            os.path.basename(args.data_dir)))
        pickle.dump(predictions, open("test.pkl", "wb"))
        with open(output_test_predictions_file, "w", encoding=args.encoding) as writer:
            with open(os.path.join(args.data_dir, "{}.txt".format(args.mode)), "r", encoding=args.encoding) as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + line.split()[-1].replace("\n", "") + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


if __name__ == "__main__":
    main()
