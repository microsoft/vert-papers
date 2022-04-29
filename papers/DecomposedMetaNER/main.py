# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

import joblib
from learner import Learner
from modeling import ViterbiDecoder
from preprocessor import Corpus, EntityTypes
from utils import set_seed


def get_label_list(args):
    # prepare dataset
    if args.tagging_scheme == "BIOES":
        label_list = ["O", "B", "I", "E", "S"]
    elif args.tagging_scheme == "BIO":
        label_list = ["O", "B", "I"]
    else:
        label_list = ["O", "I"]
    return label_list


def get_data_path(agrs, train_mode: str):
    assert args.dataset in [
        "FewNERD",
        "Domain",
        "Domain2",
    ], f"Dataset: {args.dataset} Not Support."
    if args.dataset == "FewNERD":
        return os.path.join(
            args.data_path,
            args.mode,
            "{}_{}_{}.jsonl".format(train_mode, args.N, args.K),
        )
    elif args.dataset == "Domain":
        if train_mode == "dev":
            train_mode = "valid"
        text = "_shot_5" if args.K == 5 else ""
        replace_text = "-" if args.K == 5 else "_"
        return os.path.join(
            "ACL2020data",
            "xval_ner{}".format(text),
            "ner_{}_{}{}.json".format(train_mode, args.N, text).replace(
                "_", replace_text
            ),
        )
    elif args.dataset == "Domain2":
        if train_mode == "train":
            return os.path.join("domain2", "{}_10_5.json".format(train_mode))
        return os.path.join(
            "domain2", "{}_{}_{}.json".format(train_mode, args.mode, args.K)
        )


def replace_type_embedding(learner, args):
    logger.info("********** Replace trained type embedding **********")
    entity_types = joblib.load(os.path.join(args.result_dir, "type_embedding.pkl"))
    N, H = entity_types.types_embedding.weight.data.shape
    for ii in range(N):
        learner.models.embeddings.word_embeddings.weight.data[
            ii + 1
        ] = entity_types.types_embedding.weight.data[ii]


def train_meta(args):
    logger.info("********** Scheme: Meta Learning **********")
    label_list = get_label_list(args)

    valid_data_path = get_data_path(args, "dev")
    valid_corpus = Corpus(
        logger,
        valid_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        viterbi=args.viterbi,
        tagging=args.tagging_scheme,
        device=args.device,
        concat_types=args.concat_types,
        dataset=args.dataset,
    )

    if args.debug:
        test_corpus = valid_corpus
        train_corpus = valid_corpus
    else:
        train_data_path = get_data_path(args, "train")
        train_corpus = Corpus(
            logger,
            train_data_path,
            args.bert_model,
            args.max_seq_len,
            label_list,
            args.entity_types,
            do_lower_case=True,
            shuffle=True,
            tagging=args.tagging_scheme,
            device=args.device,
            concat_types=args.concat_types,
            dataset=args.dataset,
        )

        if not args.ignore_eval_test:
            test_data_path = get_data_path(args, "test")
            test_corpus = Corpus(
                logger,
                test_data_path,
                args.bert_model,
                args.max_seq_len,
                label_list,
                args.entity_types,
                do_lower_case=True,
                shuffle=False,
                viterbi=args.viterbi,
                tagging=args.tagging_scheme,
                device=args.device,
                concat_types=args.concat_types,
                dataset=args.dataset,
            )

    learner = Learner(
        args.bert_model,
        label_list,
        args.freeze_layer,
        logger,
        args.lr_meta,
        args.lr_inner,
        args.warmup_prop_meta,
        args.warmup_prop_inner,
        args.max_meta_steps,
        py_alias=args.py_alias,
        args=args,
    )

    if "embedding" in args.concat_types:
        replace_type_embedding(learner, args)

    t = time.time()
    F1_valid_best = {ii: -1.0 for ii in ["all", "type", "span"]}
    F1_test = -1.0
    best_step, protect_step = -1.0, 100 if args.train_mode != "type" else 50

    for step in range(args.max_meta_steps):
        progress = 1.0 * step / args.max_meta_steps

        batch_query, batch_support = train_corpus.get_batch_meta(
            batch_size=args.inner_size
        )  # (batch_size=32)
        if args.use_supervise:
            span_loss, type_loss = learner.forward_supervise(
                batch_query,
                batch_support,
                progress=progress,
                inner_steps=args.inner_steps,
            )
        else:
            span_loss, type_loss = learner.forward_meta(
                batch_query,
                batch_support,
                progress=progress,
                inner_steps=args.inner_steps,
            )

        if step % 20 == 0:
            logger.info(
                "Step: {}/{}, span loss = {:.6f}, type loss = {:.6f}, time = {:.2f}s.".format(
                    step, args.max_meta_steps, span_loss, type_loss, time.time() - t
                )
            )

        if step % args.eval_every_meta_steps == 0 and step > protect_step:
            logger.info("********** Scheme: evaluate - [valid] **********")
            result_valid, predictions_valid = test(args, learner, valid_corpus, "valid")

            F1_valid = result_valid["f1"]
            is_best = False
            if F1_valid > F1_valid_best["all"]:
                logger.info("===> Best Valid F1: {}".format(F1_valid))
                logger.info("  Saving model...")
                learner.save_model(args.result_dir, "en", args.max_seq_len, "all")
                F1_valid_best["all"] = F1_valid
                best_step = step
                is_best = True

            if (
                result_valid["span_f1"] > F1_valid_best["span"]
                and args.train_mode != "type"
            ):
                F1_valid_best["span"] = result_valid["span_f1"]
                learner.save_model(args.result_dir, "en", args.max_seq_len, "span")
                logger.info("Best Span Store {}".format(step))
                is_best = True
            if (
                result_valid["type_f1"] > F1_valid_best["type"]
                and args.train_mode != "span"
            ):
                F1_valid_best["type"] = result_valid["type_f1"]
                learner.save_model(args.result_dir, "en", args.max_seq_len, "type")
                logger.info("Best Type Store {}".format(step))
                is_best = True

            if is_best and not args.ignore_eval_test:
                logger.info("********** Scheme: evaluate - [test] **********")
                result_test, predictions_test = test(args, learner, test_corpus, "test")

                F1_test = result_test["f1"]
                logger.info(
                    "Best Valid F1: {}, Step: {}".format(F1_valid_best, best_step)
                )
                logger.info("Test F1: {}".format(F1_test))


def test(args, learner, corpus, types: str):
    if corpus.viterbi != "none":
        id2label = corpus.id2label
        transition_matrix = corpus.transition_matrix
        if args.viterbi == "soft":
            label_list = get_label_list(args)
            train_data_path = get_data_path(args, "train")
            train_corpus = Corpus(
                logger,
                train_data_path,
                args.bert_model,
                args.max_seq_len,
                label_list,
                args.entity_types,
                do_lower_case=True,
                shuffle=True,
                tagging=args.tagging_scheme,
                viterbi="soft",
                device=args.device,
                concat_types=args.concat_types,
                dataset=args.dataset,
            )
            id2label = train_corpus.id2label
            transition_matrix = train_corpus.transition_matrix

        viterbi_decoder = ViterbiDecoder(id2label, transition_matrix)
    else:
        viterbi_decoder = None
    result_test, predictions = learner.evaluate_meta_(
        corpus,
        logger,
        lr=args.lr_finetune,
        steps=args.max_ft_steps,
        mode=args.mode,
        set_type=types,
        type_steps=args.max_type_ft_steps,
        viterbi_decoder=viterbi_decoder,
    )
    return result_test, predictions


def evaluate(args):
    logger.info("********** Scheme: Meta Test **********")
    label_list = get_label_list(args)

    valid_data_path = get_data_path(args, "dev")
    valid_corpus = Corpus(
        logger,
        valid_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        viterbi=args.viterbi,
        concat_types=args.concat_types,
        dataset=args.dataset,
    )

    test_data_path = get_data_path(args, "test")
    test_corpus = Corpus(
        logger,
        test_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        viterbi=args.viterbi,
        concat_types=args.concat_types,
        dataset=args.dataset,
    )

    learner = Learner(
        args.bert_model,
        label_list,
        args.freeze_layer,
        logger,
        args.lr_meta,
        args.lr_inner,
        args.warmup_prop_meta,
        args.warmup_prop_inner,
        args.max_meta_steps,
        model_dir=args.model_dir,
        py_alias=args.py_alias,
        args=args,
    )

    logger.info("********** Scheme: evaluate - [valid] **********")
    test(args, learner, valid_corpus, "valid")

    logger.info("********** Scheme: evaluate - [test] **********")
    test(args, learner, test_corpus, "test")


def convert_bpe(args):
    def convert_base(train_mode: str):
        data_path = get_data_path(args, train_mode)
        corpus = Corpus(
            logger,
            data_path,
            args.bert_model,
            args.max_seq_len,
            label_list,
            args.entity_types,
            do_lower_case=True,
            shuffle=False,
            tagging=args.tagging_scheme,
            viterbi=args.viterbi,
            concat_types=args.concat_types,
            dataset=args.dataset,
            device=args.device,
        )

        for seed in [171, 354, 550, 667, 985]:
            path = os.path.join(
                args.model_dir,
                f"all_{train_mode if train_mode == 'test' else 'valid'}_preds.pkl",
            ).replace("171", str(seed))
            data = joblib.load(path)
            if len(data) == 3:
                spans = data[-1]
            else:
                spans = [[jj[:-2] for jj in ii] for ii in data[-1]]
            target = [[jj[:-1] for jj in ii] for ii in data[0]]

            res = corpus._decoder_bpe_index(spans)
            target = corpus._decoder_bpe_index(target)
            with open(
                f"preds/{args.mode}-{args.N}way{args.K}shot-seed{seed}-{train_mode}.jsonl",
                "w",
            ) as f:
                json.dump(res, f)
            if seed != 171:
                continue
            with open(
                f"preds/{args.mode}-{args.N}way{args.K}shot-seed{seed}-{train_mode}_golden.jsonl",
                "w",
            ) as f:
                json.dump(target, f)

    logger.info("********** Scheme: Convert BPE **********")
    os.makedirs("preds", exist_ok=True)
    label_list = get_label_list(args)
    convert_base("dev")
    convert_base("test")


if __name__ == "__main__":

    def my_bool(s):
        return s != "False"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="FewNERD", help="FewNERD or Domain"
    )
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--mode", type=str, default="intra")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="if true, will load the trained model and run test only",
    )
    parser.add_argument(
        "--convert_bpe",
        action="store_true",
        help="if true, will convert the bpe encode result to word level.",
    )
    parser.add_argument("--tagging_scheme", type=str, default="BIO", help="BIO or IO")
    # dataset settings
    parser.add_argument("--data_path", type=str, default="episode-data")
    parser.add_argument(
        "--result_dir", type=str, help="where to save the result.", default="test"
    )
    parser.add_argument(
        "--model_dir", type=str, help="dir name of a trained model", default=""
    )

    # meta-test setting
    parser.add_argument(
        "--lr_finetune",
        type=float,
        help="finetune learning rate, used in [test_meta]. and [k_shot setting]",
        default=3e-5,
    )
    parser.add_argument(
        "--max_ft_steps", type=int, help="maximal steps token for fine-tune.", default=3
    )
    parser.add_argument(
        "--max_type_ft_steps",
        type=int,
        help="maximal steps token for entity type fine-tune.",
        default=0,
    )

    # meta-train setting
    parser.add_argument(
        "--inner_steps",
        type=int,
        help="every ** inner update for one meta-update",
        default=2,
    )  # ===>
    parser.add_argument(
        "--inner_size",
        type=int,
        help="[number of tasks] for one meta-update",
        default=32,
    )
    parser.add_argument(
        "--lr_inner", type=float, help="inner loop learning rate", default=3e-5
    )
    parser.add_argument(
        "--lr_meta", type=float, help="meta learning rate", default=3e-5
    )
    parser.add_argument(
        "--max_meta_steps",
        type=int,
        help="maximal steps token for meta training.",
        default=5001,
    )
    parser.add_argument("--eval_every_meta_steps", type=int, default=500)
    parser.add_argument(
        "--warmup_prop_inner",
        type=int,
        help="warm up proportion for inner update",
        default=0.1,
    )
    parser.add_argument(
        "--warmup_prop_meta",
        type=int,
        help="warm up proportion for meta update",
        default=0.1,
    )

    # permanent params
    parser.add_argument(
        "--freeze_layer", type=int, help="the layer of mBERT to be frozen", default=0
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
        default="",
    )
    parser.add_argument(
        "--viterbi", type=str, default="hard", help="hard or soft or None"
    )
    parser.add_argument(
        "--concat_types", type=str, default="past", help="past or before or None"
    )
    # expt setting
    parser.add_argument(
        "--seed", type=int, help="random seed to reproduce the result.", default=667
    )
    parser.add_argument("--gpu_device", type=int, help="GPU device num", default=0)
    parser.add_argument("--py_alias", type=str, help="python alias", default="python")
    parser.add_argument(
        "--types_path",
        type=str,
        help="the path of entities types",
        default="data/entity_types.json",
    )
    parser.add_argument(
        "--negative_types_number",
        type=int,
        help="the number of negative types in each batch",
        default=4,
    )
    parser.add_argument(
        "--negative_mode", type=str, help="the mode of negative types", default="batch"
    )
    parser.add_argument(
        "--types_mode", type=str, help="the embedding mode of type span", default="cls"
    )
    parser.add_argument("--name", type=str, help="the name of experiment", default="")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument(
        "--init_type_embedding_from_bert",
        action="store_true",
        help="initialization type embedding from BERT",
    )

    parser.add_argument(
        "--use_classify",
        action="store_true",
        help="use classifier after entity embedding",
    )
    parser.add_argument(
        "--distance_mode", type=str, help="embedding distance mode", default="cos"
    )
    parser.add_argument("--similar_k", type=float, help="cosine similar k", default=10)
    parser.add_argument("--shared_bert", default=True, type=my_bool, help="shared BERT")
    parser.add_argument("--train_mode", default="add", type=str, help="add, span, type")
    parser.add_argument("--eval_mode", default="add", type=str, help="add, two-stage")
    parser.add_argument(
        "--type_threshold", default=2.5, type=float, help="typing decoder threshold"
    )
    parser.add_argument(
        "--lambda_max_loss", default=2.0, type=float, help="span max loss lambda"
    )
    parser.add_argument(
        "--inner_lambda_max_loss", default=2.0, type=float, help="span max loss lambda"
    )
    parser.add_argument(
        "--inner_similar_k", type=float, help="cosine similar k", default=10
    )
    parser.add_argument(
        "--ignore_eval_test", help="if/not eval in test", action="store_true"
    )
    parser.add_argument(
        "--nouse_inner_ft",
        action="store_true",
        help="if true, will convert the bpe encode result to word level.",
    )
    parser.add_argument(
        "--use_supervise",
        action="store_true",
        help="if true, will convert the bpe encode result to word level.",
    )

    args = parser.parse_args()
    args.negative_types_number = args.N - 1
    if "Domain" in args.dataset:
        args.types_path = "data/entity_types_domain.json"

    # setup random seed
    set_seed(args.seed, args.gpu_device)

    # set up GPU device
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device)

    # setup logger settings
    if args.test_only:
        top_dir = "models-{}-{}-{}".format(args.N, args.K, args.mode)
        args.model_dir = "{}-innerSteps_{}-innerSize_{}-lrInner_{}-lrMeta_{}-maxSteps_{}-seed_{}{}".format(
            args.bert_model,
            args.inner_steps,
            args.inner_size,
            args.lr_inner,
            args.lr_meta,
            args.max_meta_steps,
            args.seed,
            "-name_{}".format(args.name) if args.name else "",
        )
        args.model_dir = os.path.join(top_dir, args.model_dir)
        if not os.path.exists(args.model_dir):
            if args.convert_bpe:
                os.makedirs(args.model_dir)
            else:
                raise ValueError("Model directory does not exist!")
        fh = logging.FileHandler(
            "{}/log-test-ftLr_{}-ftSteps_{}.txt".format(
                args.model_dir, args.lr_finetune, args.max_ft_steps
            )
        )

    else:
        top_dir = "models-{}-{}-{}".format(args.N, args.K, args.mode)
        args.result_dir = "{}-innerSteps_{}-innerSize_{}-lrInner_{}-lrMeta_{}-maxSteps_{}-seed_{}{}".format(
            args.bert_model,
            args.inner_steps,
            args.inner_size,
            args.lr_inner,
            args.lr_meta,
            args.max_meta_steps,
            args.seed,
            "-name_{}".format(args.name) if args.name else "",
        )
        os.makedirs(top_dir, exist_ok=True)
        if not os.path.exists("{}/{}".format(top_dir, args.result_dir)):
            os.mkdir("{}/{}".format(top_dir, args.result_dir))
        elif args.result_dir != "test":
            pass

        args.result_dir = "{}/{}".format(top_dir, args.result_dir)
        fh = logging.FileHandler("{}/log-training.txt".format(args.result_dir))

        # dump args
        with Path("{}/args-train.json".format(args.result_dir)).open(
            "w", encoding="utf-8"
        ) as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

    if args.debug:
        os.makedirs("debug", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    args.device = device
    logger.info(f"Using Device {device}")
    args.entity_types = EntityTypes(
        args.types_path, args.negative_types_number, args.negative_mode
    )
    args.entity_types.build_types_embedding(
        args.bert_model,
        True,
        args.device,
        args.types_mode,
        args.init_type_embedding_from_bert,
    )

    if args.convert_bpe:
        convert_bpe(args)
    elif args.test_only:
        if args.model_dir == "":
            raise ValueError("NULL model directory!")
        evaluate(args)
    else:
        if args.model_dir != "":
            raise ValueError("Model directory should be NULL!")
        train_meta(args)
