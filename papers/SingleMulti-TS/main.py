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
import logging
import os
import random
import time

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from modeling import BertForTokenClassification_, BaseModel, DomainLearner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def train(args, model, task_dataset, src_probs, loss_ignore_index, src_predictions=None):
    """ Train the model using multi-task training and knowledge distillation """
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    if not src_probs is None:
        # check the size of the two relative datasets, expand task_dataset with src_probs_list
        assert len(task_dataset) == src_probs.size(0)

        # build hard labels if needed
        if not args.hard_label_usage == 'none':
            num_labels = src_probs.size(-1)
            confident_labels = None
            confident_labels_mask = None
            for src_prediction in src_predictions:
                tmp_src_labels = torch.argmax(src_prediction, dim=-1)
                if confident_labels is None:
                    confident_labels = tmp_src_labels
                    confident_labels_mask = torch.ones_like(confident_labels)
                else:
                    confident_labels_mask[confident_labels != tmp_src_labels] = 0
                    confident_labels[confident_labels != tmp_src_labels] = num_labels

            embedding_matrix = torch.cat([torch.eye(num_labels), torch.zeros(1, num_labels)], dim=0).to(args.device)
            hard_labels = torch.nn.functional.embedding(confident_labels, embedding_matrix).detach()
            confident_labels_mask = confident_labels_mask.detach()

            # s_label0 = torch.argmax(src_predictions[0], dim=-1)
            # s_label1 = torch.argmax(src_predictions[1], dim=-1)
            # s_label2 = torch.argmax(src_predictions[2], dim=-1)
            # for ki in range(src_probs.size(0)):
            #     for kj in range(src_probs.size(1)):
            #         if confident_labels_mask[ki, kj] == 1:
            #             if not (confident_labels[ki, kj] == s_label0[ki, kj] and confident_labels[ki, kj] == s_label1[
            #                 ki, kj] and confident_labels[ki, kj] == s_label2[ki, kj]):
            #                 raise ValueError("Error 0")
            #             if not (hard_labels[ki, kj, confident_labels[ki, kj]].cpu().item() == 1 and torch.sum(
            #                     hard_labels[ki, kj]).cpu().item() == 1):
            #                 raise ValueError("Error 2")
            #         else:
            #             if (confident_labels[ki, kj] == s_label0[ki, kj] and confident_labels[ki, kj] == s_label1[
            #                 ki, kj] and confident_labels[ki, kj] == s_label2[ki, kj]):
            #                 raise ValueError("Error 1")
            #             if not torch.sum(hard_labels[ki, kj]).cpu().item() == 0:
            #                 raise ValueError("Error 3")

            if args.hard_label_usage == 'replace':
                src_probs[confident_labels_mask == 1, :] = hard_labels[confident_labels_mask == 1, :]

                # for ki in range(src_probs.size(0)):
                #     for kj in range(src_probs.size(1)):
                #         if confident_labels_mask[ki, kj] == 1:
                #             if not torch.sum(torch.abs(src_probs[ki, kj] - hard_labels[ki, kj])).cpu().item() == 0:
                #                 raise ValueError("Error 0")
                #         else:
                #             if torch.sum(torch.abs(src_probs[ki, kj] - hard_labels[ki, kj])).cpu().item() == 0:
                #                 raise ValueError("Error 1")

        task_dataset.tensors += (src_probs,)
        if args.hard_label_usage == 'weight':
            task_dataset.tensors += (hard_labels,)
            task_dataset.tensors += (confident_labels_mask,)

    # parepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(task_dataset)
    dataloader = DataLoader(task_dataset, sampler=sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(dataloader) + 1
    else:
        t_total = len(dataloader) * args.num_train_epochs

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num task examples = %d", len(task_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  GPU IDs for training: %s", " ".join([str(id) for id in args.gpu_ids]))
    logger.info("  Total task optimization steps = %d", t_total)
    logger.info("  Total language identifier optimization steps = %d", t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    opt_params = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(opt_params, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total * args.warmup_ratio), t_total=t_total)

    if args.n_gpu > 1:
        base_model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    global_step = 0
    loss_accum, loss_KD_accum = 0.0, 0.0
    logging_loss, logging_loss_KD = 0.0, 0.0

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_i in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            model.zero_grad()

            inputs = {"input_ids": batch[0].to(args.device),
                      "attention_mask": batch[1].to(args.device),
                      "token_type_ids": batch[2].to(args.device),
                      "labels": batch[3].to(args.device),
                      "src_probs": batch[4] if not src_probs is None else None,
                      "loss_ignore_index": loss_ignore_index,
                      "hard_labels": batch[5] if args.hard_label_usage == 'weight' else None,
                      "hard_labels_mask": batch[6] if args.hard_label_usage == 'weight' else None,
                      "hard_label_loss_weight": args.hard_label_weight} # activate the KD loss

            outputs = model(**inputs)

            if src_probs is None:
                loss = outputs[0]
                loss_KD = loss
            else:
                loss_KD, loss = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss_KD = loss_KD.mean()

            # loss.backward()
            loss_KD.backward()

            loss_accum += loss.item()
            loss_KD_accum += loss_KD.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (loss_accum - logging_loss) / args.logging_steps, global_step)
                tb_writer.add_scalar("loss_KD", (loss_KD_accum - logging_loss_KD) / args.logging_steps, global_step)
                logger.info("Epoch: {}\t global_step: {}\t lr: {:.8}\tloss: {:.8f}\tloss_KD: {:.8f}".format(epoch_i,
                        global_step, scheduler.get_lr()[0], (loss_accum - logging_loss) / args.logging_steps,
                                                        (loss_KD_accum - logging_loss_KD) / args.logging_steps))

                logging_loss = loss_accum
                logging_loss_KD = loss_KD_accum

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # base model
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()

    return global_step, loss_KD_accum / global_step, loss_accum / global_step

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, args.tgt_lang, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation: %s *****", args.tgt_lang)
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
                      "token_type_ids": batch[2],
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


def weighted_voting(args, dataset, src_probs, labels, pad_token_label_id):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    out_label_ids = None
    for batch in eval_dataloader:
        out_label_ids = batch[3] if out_label_ids is None else np.append(out_label_ids, batch[3], axis=0)

    preds = np.argmax(src_probs.cpu().numpy(), axis=-1)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", os.path.basename(args.data_dir))
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, lang, mode):
    data_path = os.path.join(args.data_dir, lang)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_path, "cached_{}_{}_{}".format(mode,
                            list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s.", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_path)
        examples = read_examples_from_file(data_path, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=False,
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=False,
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
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


def get_src_probs(args, dataset, src_lang):
    """
    get label distribution from the `src_lang` teacher model.
    """
    # load src model
    src_model_path = os.path.join(args.src_model_dir, "{}{}".format(args.src_model_dir_prefix, src_lang))
    src_model = BertForTokenClassification_.from_pretrained(src_model_path)
    src_model.to(args.device)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info("***** Compute source probs for [%s] using model [%s] *****", args.tgt_lang, src_model_path)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    src_model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels": None} #batch[3]}
            outputs = src_model(**inputs)
            logits = outputs[0]

        preds = logits.detach() if preds is None else torch.cat((preds, logits.detach()), dim=0) # dataset_len x max_seq_len x label_len

    preds = torch.nn.functional.softmax(preds, dim=-1)

    return preds

def get_st_embeds(args, dataset, config, lang, base_model=None):
    logger.info("***** Compute sentence embeddings for [%s] plain text dataset using the [%s] base_model *****", lang, "pre-trained" if  base_model is None else "domain")
    if base_model is None:
        base_model = BaseModel.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
        base_model.to(args.device)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    st_embeds = None
    base_model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            outputs = base_model(**inputs)
            pooled_outputs = outputs[1]

        st_embeds = pooled_outputs.detach() if st_embeds is None else torch.cat((st_embeds, pooled_outputs.detach()), dim=0)  # dataset_len x hidden_size

    return st_embeds

def get_st_sims(args, model, f_dataset, src_idxs):
    # parepare dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(f_dataset)
    eval_dataloader = DataLoader(f_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Evaluate!
    logger.info("***** Running evaluation on %s *****", args.tgt_lang)
    logger.info("  Num of examples = %d", len(f_dataset))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_eval_batch_size)
    logger.info("  GPU IDs for training: %s", " ".join([str(id) for id in args.gpu_ids]))
    logger.info("  Batch size = %d", args.eval_batch_size)

    logits = None
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(batch[0], device=args.device) # loss, logits
            logits_batch = outputs[0] # batch_size x n_langs

        logits = logits_batch.detach() if logits is None else torch.cat((logits, logits_batch.detach()), dim=0)

    logits = logits[:, src_idxs]

    if args.tau_metric == "var":
        tau = torch.var(logits)
    elif args.tau_metric == "std":
        tau = torch.std(logits)
    else:
        assert False
    tau = torch.reciprocal(tau)
    logger.info(  "==> tau: {}".format(tau))

    logits *= tau

    st_sims = torch.nn.functional.softmax(logits, dim=-1)  # dataset_len x n_src_langs

    dm_sims = torch.mean(st_sims, dim=0)  # n_src_langs

    print_sim_info(st_sims, dm_sims)

    return st_sims, dm_sims


def get_unlearn_st_sims(args, tgt_dataset, src_domain_ebds):
    sims = []
    for i in range(len(src_domain_ebds)):
        if args.sim_type == 'cosine':
            sim = torch.nn.functional.cosine_similarity(tgt_dataset, src_domain_ebds[i], dim=1)
            sims.append(sim.unsqueeze(1))
        elif args.sim_type == 'l1':
            res = tgt_dataset - src_domain_ebds[i]
            sim = -torch.sum(torch.abs(res), dim=1, keepdim=True)
            sims.append(sim)
        elif args.sim_type == 'l2':
            res = tgt_dataset - src_domain_ebds[i]
            sim = -torch.sqrt(torch.sum(res.mul(res), dim=1, keepdim=True))
            sims.append(sim)
        else:
            raise ValueError('ERROR: unknown args.sim_type')
    logits = torch.cat(sims, dim=1)

    if args.tau_metric == "var":
        tau = torch.var(logits)
    elif args.tau_metric == "std":
        tau = torch.std(logits)
    else:
        assert False
    tau = torch.reciprocal(tau)
    logger.info(  "==> tau: {}".format(tau))

    logits *= tau

    st_sims = torch.nn.functional.softmax(logits, dim=-1)  # dataset_len x n_src_langs

    dm_sims = torch.mean(st_sims, dim=0)  # n_src_langs

    print_sim_info(st_sims, dm_sims)

    return st_sims, dm_sims


def print_sim_info(st_sims, dm_sims):
    logger.info("  Sentence similarities:")
    logger.info("  " + "\t".join(args.src_langs))
    for i in range(0, 20, 2):
        logger.info("  " + "\t".join([str(round(v.item(), 4)) for v in st_sims[i]]))

    logger.info("  Domain similarities:")
    logger.info("  " + "\t".join(args.src_langs))
    logger.info("  " + "\t".join([str(round(v.item(), 4)) for v in dm_sims]))


def get_src_weighted_probs(args, tgt_pt_dataset, config, mode="train", src_datasets=None):
    st_sims = None
    dm_sims = None
    if args.sim_dir != "":
        if args.sim_type == 'learn' or args.sim_type == 'learn_fixed_seed':
            if args.sim_with_tgt:
                sims_dir = os.path.join(args.sim_dir, "rank_{}-gamma_{}".format(args.low_rank_size, args.gamma_R))
            else:
                sims_dir = os.path.join(args.sim_dir, "{}{}{}-rank_{}-gamma_{}-seed_{}".format(args.tgt_lang,
                                                                                               '-balanced' if args.balance_classes else '',
                                                                                               '-domain_orth' if args.domain_orthogonal and args.gamma_R > 0 else '',
                                                                                               args.low_rank_size,
                                                                                               0 if args.gamma_R == 0 else args.gamma_R,
                                                                                               args.seed if not args.sim_type == 'learn_fixed_seed' else 42))
        else:
            sims_dir = args.sim_dir

        logger.info("sim_dir: {}".format(sims_dir))
        re_compute_sims = True
        sims_path = os.path.join(sims_dir, "sims-{}-{}-{}-{}-{}.bin".format(args.sim_type, args.tau_metric, "_".join(args.src_langs), args.tgt_lang,  mode))
        if os.path.exists(sims_path):
            logger.info("==> Loading similarities....")
            st_sims, dm_sims = torch.load(sims_path, map_location=args.device) # dataset_len x n_srcs
            if st_sims.shape[0] != len(tgt_pt_dataset):
                raise ValueError("==> Mismatch of target example numbers!")
            else:
                re_compute_sims = False
                print_sim_info(st_sims, dm_sims)

        if re_compute_sims: # sims = None
            logger.info("==> Computing similarities....")

            if args.sim_type == 'learn' or args.sim_type == 'learn_fixed_seed':
                dm_namespace = torch.load(os.path.join(sims_dir, "training_args.bin"))
                if args.sim_with_tgt:
                    langs = dm_namespace.src_langs + [dm_namespace.tgt_lang]
                else:
                    langs = dm_namespace.src_langs
                src_idxs = [langs.index(l) for l in args.src_langs]

                st_ebd = get_st_embeds(args, tgt_pt_dataset, config, args.tgt_lang)  # dataset_len x hidden_size
                dataset_st_ebd = TensorDataset(st_ebd)

                domain_model = DomainLearner(len(langs), config.hidden_size, args.low_rank_size)
                domain_model.load_state_dict(torch.load(os.path.join(sims_dir, "domain_model.bin"), map_location=args.device))
                domain_model.to(args.device)

                st_sims, dm_sims = get_st_sims(args, domain_model, dataset_st_ebd, src_idxs)
            else:
                st_ebd = get_st_embeds(args, tgt_pt_dataset, config, args.tgt_lang)  # dataset_len x hidden_size
                src_ebds = []
                for sri, src_dataset in enumerate(src_datasets):
                    src_st_ebd = get_st_embeds(args, src_dataset, config, args.src_langs[sri])
                    src_domain_st_ebd = torch.mean(src_st_ebd, dim=0, keepdim=True)
                    src_ebds.append(src_domain_st_ebd)
                st_sims, dm_sims = get_unlearn_st_sims(args, st_ebd, src_ebds)

            logger.info("==> Saving similarities....")
            torch.save((st_sims, dm_sims), sims_path)

    src_probs = None
    src_predictions = []
    if args.sim_level == "domain": # dm_sims: n_src_langs
        for i, lang in enumerate(args.src_langs):
            l_prob = get_src_probs(args, tgt_pt_dataset, src_lang=lang)  # dataset_len x max_seq_len x num_labels
            src_predictions.append(l_prob)
            if src_probs is None:
                src_probs = dm_sims[i] * l_prob if dm_sims is not None else (1.0 / len(args.src_langs)) * l_prob
            else:
                src_probs += dm_sims[i] * l_prob if dm_sims is not None else (1.0 / len(args.src_langs)) * l_prob
    elif args.sim_level == "sentence": # st_sims: dataset_len x n_src_langs
        for i, lang in enumerate(args.src_langs):
            l_prob = get_src_probs(args, tgt_pt_dataset, src_lang=lang)  # dataset_len x max_seq_len x num_labels
            src_predictions.append(l_prob)
            if src_probs is None:
                src_probs = st_sims[:, i].unsqueeze(dim=-1).unsqueeze(dim=-1) * l_prob if st_sims is not None else (1.0 / len(args.src_langs)) * l_prob
            else:
                src_probs += st_sims[:, i].unsqueeze(dim=-1).unsqueeze(dim=-1) * l_prob if st_sims is not None else (1.0 / len(args.src_langs)) * l_prob
    else:
        raise ValueError("==> Please clarify a similarity level!")

    return src_probs, src_predictions

def setup():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./data/ner/conll", type=str,  # required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--output_dir", default="conll-model/test", type=str,  # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--tgt_lang", default="debug", type=str,
                        help="target language to train the student model")
    parser.add_argument("--src_model_dir", default="conll-model", type=str,
                        help="path to load teacher models")
    parser.add_argument("--src_model_dir_prefix", default="mono-src-", type=str,
                        help="prefix of the teacher model dir (to indicate the model type)")
    parser.add_argument("--src_langs", type=str, nargs="+", default="",
                        help="source languages used for multi-teacher models")
    parser.add_argument("--sim_with_tgt", action="store_true",
                        help="whether the similarity matrix is trained with plain texts of target language.")

    parser.add_argument("--sim_dir", default="", type=str,
                        help="path to load trained sims")
    parser.add_argument("--sim_level", type=str, default="domain",
                        help="level to measure similarity: `domain` or `sentence`.")
    parser.add_argument("--low_rank_size", type=int, default=64,
                        help="size use for low rank approximation of the bilinear operation.")
    parser.add_argument("--gamma_R", type=float, default=0.01,
                        help="size use for low rank approximation of the bilinear operation.")
    parser.add_argument("--tau_metric", type=str, default="var",
                        help="whether to compute tau using target information.")

    ## Other parameters
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,  # required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--labels", default="./data/ner/conll/labels.txt", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training on the training set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_voting", action="store_true",
                        help="Whether to run predictions by voting.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
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
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument("--logging_steps", type=int, default=20,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gpu_ids", type=int, nargs="+",
                        help="ids of the gpus to use")
    parser.add_argument("--balance_classes", action="store_true",
                        help="To handle the class imbalance problem or not")
    parser.add_argument("--domain_orthogonal", action="store_true",
                        help="To constrain that the domain embeddings are orthogonal rather than the features are orthogonal.")
    parser.add_argument("--sim_type", default="learn", type=str,
                        help="Selection for calculating the similarities, should be in {'learn', 'cosine', 'l1', 'l2', 'learn_fixed_seed'}")
    parser.add_argument("--hard_label_usage", default="none", type=str,
                        help="About how to use ensured hard labels, should be in {'none', 'replace', 'weight'}")
    parser.add_argument("--hard_label_weight", default=0, type=float,
                        help="Weight for the loss w.r.t hard labels if --hard_label_usage == 'weight'.")
    parser.add_argument("--train_hard_label", action="store_true",
                        help="Use hard labels (0/1) to train the student network in a multi-source scenario.")

    args = parser.parse_args()
    if(args.tgt_lang in args.src_langs):
        args.src_langs.remove(args.tgt_lang)

    if not args.sim_type in ['learn', 'cosine', 'l1', 'l2', 'learn_fixed_seed']:
        raise ValueError("Error: sim_type NOT in provided candidate list")
    if not args.hard_label_usage in ['none', 'replace', 'weight']:
        raise ValueError("Error: hard_label_usage NOT in provided candidate list")


    # Check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")) and os.path.basename(args.output_dir) != "test":
        raise ValueError("Train: Output directory already exists and is not empty.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_voting and os.path.basename(args.output_dir) != "test":
        is_done = False
        for name in os.listdir(args.output_dir):  # result file: "test_results-TIME-LANGUAGE"
            if "test_results" in name:
                is_done = True
                break
        if is_done:
            raise ValueError("Voting: Output directory already exists and is not empty.")

    if os.path.exists(args.output_dir) and args.do_predict:
        is_done = False
        for name in os.listdir(args.output_dir): # result file: "test_results-TIME-LANGUAGE"
            if "test_results" in name and args.tgt_lang in name:
                is_done = True
                break
        if is_done:
            raise ValueError("Predict: Output directory already exists and is not empty.")

    # Setup CUDA, GPU & distributed training
    args.n_gpu = len(args.gpu_ids)  # torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids[0])
    device = torch.device("cuda")
    args.device = device

    # Setup logging
    if args.do_train and (not os.path.exists(args.output_dir)):
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
    log_name += "-{}".format("_".join(args.src_langs))
    log_name += "-{}.txt".format(args.tgt_lang)
    fh = logging.FileHandler(os.path.join(args.log_dir, log_name))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    logger.info("Training/evaluation parameters %s", args)

    return args

def main(args):
    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index  # -100 here

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)

    src_datasets = []
    for src in args.src_langs:
        src_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, src, mode="train")
        src_datasets.append(src_dataset)

    # Training
    if args.do_train:
        logger.info("********** scheme: training with KD **********")

        # compute sentence embeddings of target training examples
        dataset_pt = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, args.tgt_lang, mode="train")

        src_probs, src_predictions = get_src_weighted_probs(args, dataset_pt, config, mode="train", src_datasets=src_datasets)

        # load target model (pretrained BERT) and tokenizer
        model = BertForTokenClassification_.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
        model.to(args.device)

        task_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, args.tgt_lang, mode="train")

        if args.train_hard_label:
            # update the src_probs as hard labels
            voting_labels = torch.argmax(src_probs, dim=-1).cpu()
            voting_labels[task_dataset.tensors[3] == pad_token_label_id] = pad_token_label_id
            task_dataset = TensorDataset(task_dataset.tensors[0], task_dataset.tensors[1], task_dataset.tensors[2], voting_labels)
            src_probs = None

        # Train!
        global_step, loss_KD, loss = train(args, model, task_dataset, src_probs, pad_token_label_id, src_predictions)
        logger.info(" global_step = %s, average task KD loss = %s, average task loss = %s", global_step, loss_KD, loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_voting:
        logger.info("********** scheme: Voting with %s **********", "averaging" if args.sim_dir == "" else "{}/rank_{}-gamma_{}".format(args.sim_dir, args.low_rank_size, args.gamma_R))
        test_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, args.tgt_lang, mode="test")

        src_probs, _ = get_src_weighted_probs(args, test_dataset, config, mode="test", src_datasets=src_datasets)

        result, predictions = weighted_voting(args, test_dataset, src_probs, labels, pad_token_label_id)
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results-{}-{}.txt".format(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            os.path.basename(args.data_dir)))
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions-{}-{}.txt".format(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            os.path.basename(args.data_dir)))
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, args.tgt_lang, "test.txt"), "r") as f:
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

    if args.do_predict:
        logger.info("********** scheme: prediction **********")
        model = BertForTokenClassification_.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results-{}-{}.txt".format(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.tgt_lang))
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions-{}-{}.txt".format(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.tgt_lang))
        with open(output_test_predictions_file, "w", encoding='utf-8') as writer:
            with open(os.path.join(args.data_dir, args.tgt_lang, "test.txt"), "r", encoding='utf-8') as f:
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
    args = setup()
    main(args)
