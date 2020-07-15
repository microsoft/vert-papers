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
import torch, math
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

from transformers import AdamW, WarmupLinearSchedule
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from modeling import BaseModel, DomainLearner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def train(args, domain_model, pt_dataset):
    """ Train the model using multi-task training and knowledge distillation """
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    # parepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(pt_dataset)
    dataloader = DataLoader(pt_dataset, sampler=sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(dataloader) + 1
    else:
        t_total = len(dataloader) * args.num_train_epochs

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num of examples = %d", len(pt_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  GPU IDs for training: %s", " ".join([str(id) for id in args.gpu_ids]))
    logger.info("  Total task optimization steps = %d", t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(domain_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # scheduler_base = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total * args.warmup_ratio), t_total=t_total)

    if args.n_gpu > 1:
        domain_model = torch.nn.DataParallel(domain_model, device_ids=args.gpu_ids)

    global_step = 0
    loss_accum, logging_loss = 0.0, 0.0
    loss_f_accum, logging_loss_f = 0.0, 0.0
    loss_R_accum, logging_loss_R = 0.0, 0.0

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    domain_model.train()
    for epoch_i in range(args.num_train_epochs):
        for step, (features, labels) in enumerate(dataloader):
            domain_model.zero_grad()

            outputs = domain_model(features, labels, device=args.device)  # loss, logits
            loss, loss_f, loss_R = outputs[:3]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()

            loss_accum += loss.item()
            loss_f_accum += loss_f.item()
            loss_R_accum += loss_R.item()

            torch.nn.utils.clip_grad_norm_(domain_model.parameters(), args.max_grad_norm)

            scheduler.step()  # Update learning rate schedule
            optimizer.step()

            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

                tb_writer.add_scalar("loss", (loss_accum - logging_loss) / args.logging_steps, global_step)
                tb_writer.add_scalar("loss_f", (loss_f_accum - logging_loss_f) / args.logging_steps, global_step)
                tb_writer.add_scalar("loss_R", (loss_R_accum - logging_loss_R) / args.logging_steps, global_step)
                logger.info("Epoch: {}\t global_step: {}\t lr: {:.8}\tloss: {:.8f}".format(epoch_i,
                                                                                           global_step,
                                                                                           scheduler.get_lr()[0], (
                                                                                                       loss_accum - logging_loss) / args.logging_steps))
                logger.info(
                    "    loss_f: {:.8f}\tloss_R: {:.8f}".format((loss_f_accum - logging_loss_f) / args.logging_steps,
                                                                (loss_R_accum - logging_loss_R) / args.logging_steps))
                logging_loss = loss_accum
                logging_loss_f = loss_f_accum
                logging_loss_R = loss_R_accum

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(domain_model.state_dict(), os.path.join(output_dir, "domain_model.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()

    return global_step, loss / global_step


def evaluate(args, model, f_dataset, src_idxs):
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
            outputs = model(batch[0], device=args.device)  # loss, logits
            logits_batch = outputs[0]  # batch_size x n_langs

        logits = logits_batch.detach() if logits is None else torch.cat((logits, logits_batch.detach()), dim=0)

    logits = logits[:, src_idxs]

    if args.tau_metric == "var":
        tau = torch.var(logits)
    elif args.tau_metric == "std":
        tau = torch.std(logits)
    else:
        assert False
    tau = torch.reciprocal(tau)
    logger.info("==> tau: {}".format(tau))

    logits *= tau

    sims = torch.nn.functional.softmax(logits, dim=-1)  # dataset_len x n_src_langs

    dm_sims = torch.mean(sims, dim=0)  # n_src_langs

    logger.info("  Domain similarities:")
    logger.info("  " + "\t".join(args.src_langs))
    logger.info("  " + "\t".join([str(round(v.item(), 4)) for v in dm_sims]))

    return sims, dm_sims


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, lang, mode, plain_text=False):
    data_path = os.path.join(args.data_dir, lang)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_path, "cached_{}_{}_{}".format(mode,
                                                                            list(filter(None,
                                                                                        args.model_name_or_path.split(
                                                                                            "/"))).pop(),
                                                                            str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s. Plain text: %s", cached_features_file, plain_text)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s. Plain text: %s", data_path, plain_text)
        examples = read_examples_from_file(data_path, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=False,
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=False,
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

    if not plain_text:
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
    else:
        # assert lang in args.src_langs and lang != args.tgt_lang
        language_id = args.src_langs.index(lang) if lang in args.src_langs else len(args.src_langs)
        all_language_id = torch.tensor([language_id] * len(features), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_language_id)
        return dataset


def get_init_domain_embed(args, dataset, lang):
    config = BertConfig.from_pretrained(args.model_name_or_path)
    base_model = BaseModel.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                           config=config)
    base_model.to(args.device)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info("***** Compute logits for [%s] dataset using the base_model *****", lang)
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

        st_embeds = pooled_outputs.detach() if st_embeds is None else torch.cat((st_embeds, pooled_outputs.detach()),
                                                                                dim=0)  # dataset_len x hidden_size

    return st_embeds


def setup():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./data/ner/conll", type=str,  # required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--output_dir", default="domain_model/test", type=str,  # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--tgt_lang", default="debug", type=str,
                        help="target language to train the student model")
    parser.add_argument("--src_langs", type=str, nargs="+", default="",
                        help="source languages used for multi-teacher models")
    parser.add_argument("--low_rank_size", type=int, default=128,
                        help="size use for low rank approximation of the bilinear operation.")
    parser.add_argument("--gamma_R", type=float, default=0.01,
                        help="size use for low rank approximation of the bilinear operation.")
    parser.add_argument("--tau_metric", type=str, default="",
                        help="metric to implement normalization: var, std")

    ## Other parameters
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,  # required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--labels", default="./data/ner/conll/labels.txt ", type=str,
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
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument("--logging_steps", type=int, default=20,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,  # required=True,
                        help="random seed for initialization")
    parser.add_argument("--gpu_ids", type=int, nargs="+",  # required=True,
                        help="ids of the gpus to use")
    parser.add_argument("--balance_classes", action="store_true",
                        help="To handle the class imbalance problem or not")
    parser.add_argument("--domain_orthogonal", action="store_true",
                        help="To constrain that the domain embeddings are orthogonal rather than the features are orthogonal.")
    args = parser.parse_args()

    # args.src_langs = ["en", "es", "nl"]
    # args.tgt_lang = "de"
    if(args.tgt_lang in args.src_langs):
        args.src_langs.remove(args.tgt_lang)

    # Check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train \
            and not args.overwrite_output_dir and os.path.basename(args.output_dir) != "test":
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.n_gpu = len(args.gpu_ids)  # torch.cuda.device_count()
    device = torch.device("cpu") if args.n_gpu == 0 else torch.device("cuda:{}".format(args.gpu_ids[0]))
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

    # langs = [l for l in args.src_langs] + [args.tgt_lang]
    langs = args.src_langs
    num_langs = len(langs)

    pad_token_label_id = CrossEntropyLoss().ignore_index  # -100 here

    # load target model (pretrained BERT) and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path)

    # Training
    if args.do_train:
        logger.info("********** scheme: train domain learner **********")
        # prepare plain text datasets & compute sentence embeddings( the order of the args.src_langs matters!!! )
        f_datasets = []
        domain_embeds = []
        cnt_datasets = []
        for i, lang in enumerate(langs):
            pt_dts = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, lang, mode="train",
                                             plain_text=True)
            st_ebd = get_init_domain_embed(args, pt_dts, lang)  # dataset_size x hidden_size
            domain_embeds.append(torch.mean(st_ebd, dim=0))
            lang_id = torch.tensor([i] * st_ebd.size(0), dtype=torch.long).to(args.device)  # dataset_size
            f_datasets.append(TensorDataset(st_ebd, lang_id))
            cnt_datasets.append(st_ebd.size(0))

        f_datasets = torch.utils.data.ConcatDataset(f_datasets)
        domain_embeds = torch.stack(domain_embeds)  # (n_langs + 1) x hidden_size, device

        class_weight = None
        if args.balance_classes:
            class_weight = torch.from_numpy(1.0 / np.array(cnt_datasets, dtype=np.float32))
            class_weight = class_weight / torch.sum(class_weight)
            class_weight.requires_grad = False
            class_weight = class_weight.to(args.device)

        domain_model = DomainLearner(num_langs, config.hidden_size, args.low_rank_size, weights_init=domain_embeds,
                                     gamma=args.gamma_R, class_weight=class_weight, domain_orthogonal=args.domain_orthogonal)
        domain_model.to(args.device)

        # Train!
        global_step, loss = train(args, domain_model, f_datasets)
        logger.info(" global_step = %s, loss = %s", global_step, loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(domain_model.state_dict(), os.path.join(args.output_dir, "domain_model.bin"))
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_predict:
        # save domain similarity
        logger.info("********** scheme: prediction - compute domain similarity **********")
        sims_dir = os.path.join(args.output_dir, "{}{}{}-rank_{}-gamma_{}".format(args.tgt_lang, '' if not args.balance_classes else '-balanced',
                                                                                  '' if not args.domain_orthogonal else '-domain_orth', args.low_rank_size, args.gamma_R))

        dm_namespace = torch.load(os.path.join(sims_dir, "training_args.bin"))
        # langs = [l for l in dm_namespace.src_langs] + [dm_namespace.tgt_lang]
        langs = dm_namespace.src_langs
        src_idxs = [langs.index(l) for l in args.src_langs]

        pt_dts = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, args.tgt_lang, mode="train",
                                         plain_text=True)
        st_ebd = get_init_domain_embed(args, pt_dts, args.tgt_lang)  # dataset_size x hidden_size
        dataset_st_ebd = TensorDataset(st_ebd)

        domain_model = DomainLearner(len(langs), config.hidden_size, args.low_rank_size)
        domain_model.load_state_dict(torch.load(os.path.join(sims_dir, "domain_model.bin"), map_location=args.device))
        domain_model.to(args.device)

        st_sims, dm_sims = evaluate(args, domain_model, dataset_st_ebd, src_idxs)

        torch.save(st_sims, os.path.join(sims_dir, "sims-{}-{}-{}-{}.bin".format(args.tau_metric, "_".join(args.src_langs), args.tgt_lang, "train")))


if __name__ == "__main__":
    args = setup()
    main(args)
