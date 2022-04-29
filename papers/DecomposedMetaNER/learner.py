# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import shutil
import time
from copy import deepcopy

import numpy
import numpy as np
import torch
from torch import nn

import joblib
from modeling import BertForTokenClassification_
from transformers import CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME
from transformers import AdamW as BertAdam
from transformers import get_linear_schedule_with_warmup


logger = logging.getLogger(__file__)


class Learner(nn.Module):
    ignore_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    pad_token_label_id = -1

    def __init__(
        self,
        bert_model,
        label_list,
        freeze_layer,
        logger,
        lr_meta,
        lr_inner,
        warmup_prop_meta,
        warmup_prop_inner,
        max_meta_steps,
        model_dir="",
        cache_dir="",
        gpu_no=0,
        py_alias="python",
        args=None,
    ):

        super(Learner, self).__init__()

        self.lr_meta = lr_meta
        self.lr_inner = lr_inner
        self.warmup_prop_meta = warmup_prop_meta
        self.warmup_prop_inner = warmup_prop_inner
        self.max_meta_steps = max_meta_steps

        self.bert_model = bert_model
        self.label_list = label_list
        self.py_alias = py_alias
        self.entity_types = args.entity_types
        self.is_debug = args.debug
        self.train_mode = args.train_mode
        self.eval_mode = args.eval_mode
        self.model_dir = model_dir
        self.args = args
        self.freeze_layer = freeze_layer

        num_labels = len(label_list)

        # load model
        if model_dir != "":
            if self.eval_mode != "two-stage":
                self.load_model(self.eval_mode)
        else:
            logger.info("********** Loading pre-trained model **********")
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            self.model = BertForTokenClassification_.from_pretrained(
                bert_model,
                cache_dir=cache_dir,
                num_labels=num_labels,
                output_hidden_states=True,
            ).to(args.device)

        if self.eval_mode != "two-stage":
            self.model.set_config(
                args.use_classify,
                args.distance_mode,
                args.similar_k,
                args.shared_bert,
                self.train_mode,
            )
            self.model.to(args.device)
            self.layer_set()

    def layer_set(self):
        # layer freezing
        no_grad_param_names = ["embeddings", "pooler"] + [
            "layer.{}.".format(i) for i in range(self.freeze_layer)
        ]
        logger.info("The frozen parameters are:")
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info("  {}".format(name))

        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_meta)
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(self.max_meta_steps * self.warmup_prop_meta),
            num_training_steps=self.max_meta_steps,
        )

    def get_optimizer_grouped_parameters(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params

    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)

    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data)  # accumulate

    def get_learning_rate(self, lr, progress, warmup, schedule="linear"):
        if schedule == "linear":
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress - 1.0) / (warmup - 1.0), 0.0)
        return lr

    def inner_update(self, data_support, lr_curr, inner_steps, no_grad: bool = False):
        inner_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_inner)
        self.model.train()

        for i in range(inner_steps):
            inner_opt.param_groups[0]["lr"] = lr_curr
            inner_opt.param_groups[1]["lr"] = lr_curr

            inner_opt.zero_grad()
            _, _, loss, type_loss = self.model.forward_wuqh(
                input_ids=data_support["input_ids"],
                attention_mask=data_support["input_mask"],
                token_type_ids=data_support["segment_ids"],
                labels=data_support["label_ids"],
                e_mask=data_support["e_mask"],
                e_type_ids=data_support["e_type_ids"],
                e_type_mask=data_support["e_type_mask"],
                entity_types=self.entity_types,
                is_update_type_embedding=True,
                lambda_max_loss=self.args.inner_lambda_max_loss,
                sim_k=self.args.inner_similar_k,
            )
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss
            if no_grad:
                continue
            loss.backward()
            inner_opt.step()

        return loss.item()

    def forward_supervise(self, batch_query, batch_support, progress, inner_steps):
        span_losses, type_losses = [], []
        task_num = len(batch_query)

        for task_id in range(task_num):
            _, _, loss, type_loss = self.model.forward_wuqh(
                input_ids=batch_query[task_id]["input_ids"],
                attention_mask=batch_query[task_id]["input_mask"],
                token_type_ids=batch_query[task_id]["segment_ids"],
                labels=batch_query[task_id]["label_ids"],
                e_mask=batch_query[task_id]["e_mask"],
                e_type_ids=batch_query[task_id]["e_type_ids"],
                e_type_mask=batch_query[task_id]["e_type_mask"],
                entity_types=self.entity_types,
                lambda_max_loss=self.args.lambda_max_loss,
            )
            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss

            loss.backward()
            self.opt.step()
            self.scheduler.step()
            self.model.zero_grad()

        for task_id in range(task_num):
            _, _, loss, type_loss = self.model.forward_wuqh(
                input_ids=batch_support[task_id]["input_ids"],
                attention_mask=batch_support[task_id]["input_mask"],
                token_type_ids=batch_support[task_id]["segment_ids"],
                labels=batch_support[task_id]["label_ids"],
                e_mask=batch_support[task_id]["e_mask"],
                e_type_ids=batch_support[task_id]["e_type_ids"],
                e_type_mask=batch_support[task_id]["e_type_mask"],
                entity_types=self.entity_types,
                lambda_max_loss=self.args.lambda_max_loss,
            )
            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss

            loss.backward()
            self.opt.step()
            self.scheduler.step()
            self.model.zero_grad()

        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
        )

    def forward_meta(self, batch_query, batch_support, progress, inner_steps):
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        meta_grad = []
        span_losses, type_losses = [], []

        task_num = len(batch_query)
        lr_inner = self.get_learning_rate(
            self.lr_inner, progress, self.warmup_prop_inner
        )

        # compute meta_grad of each task
        for task_id in range(task_num):
            self.inner_update(batch_support[task_id], lr_inner, inner_steps=inner_steps)
            _, _, loss, type_loss = self.model.forward_wuqh(
                input_ids=batch_query[task_id]["input_ids"],
                attention_mask=batch_query[task_id]["input_mask"],
                token_type_ids=batch_query[task_id]["segment_ids"],
                labels=batch_query[task_id]["label_ids"],
                e_mask=batch_query[task_id]["e_mask"],
                e_type_ids=batch_query[task_id]["e_type_ids"],
                e_type_mask=batch_query[task_id]["e_type_mask"],
                entity_types=self.entity_types,
                lambda_max_loss=self.args.lambda_max_loss,
            )
            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss
            grad = torch.autograd.grad(loss, params)
            meta_grad.append(grad)

            self.load_weights(names, weights)

        # accumulate grads of all tasks to param.grad
        self.opt.zero_grad()

        # similar to backward()
        for g in meta_grad:
            self.load_gradients(names, g)
        self.opt.step()
        self.scheduler.step()

        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
        )

    # ---------------------------------------- Evaluation -------------------------------------- #
    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, "w", encoding="utf-8") as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write("{} {} {}\n".format(words[i][j], word, y_pred[i][j]))
            fw.write("\n")

    def batch_test(self, data):
        N = data["input_ids"].shape[0]
        B = 16
        BATCH_KEY = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "e_mask",
            "e_type_ids",
            "e_type_mask",
        ]

        logits, e_logits, loss, type_loss = [], [], 0, 0
        for i in range((N - 1) // B + 1):
            tmp = {
                ii: jj if ii not in BATCH_KEY else jj[i * B : (i + 1) * B]
                for ii, jj in data.items()
            }
            tmp_l, tmp_el, tmp_loss, tmp_eval_type_loss = self.model.forward_wuqh(**tmp)
            if tmp_l is not None:
                logits.extend(tmp_l.detach().cpu().numpy())
            if tmp_el is not None:
                e_logits.extend(tmp_el.detach().cpu().numpy())
            if tmp_loss is not None:
                loss += tmp_loss
            if tmp_eval_type_loss is not None:
                type_loss += tmp_eval_type_loss
        return logits, e_logits, loss, type_loss

    def evaluate_meta_(
        self,
        corpus,
        logger,
        lr,
        steps,
        mode,
        set_type,
        type_steps: int = None,
        viterbi_decoder=None,
    ):
        if not type_steps:
            type_steps = steps
        if self.is_debug:
            self.save_model(self.args.result_dir, "begin", self.args.max_seq_len, "all")

        logger.info("Begin first Stage.")
        if self.eval_mode == "two-stage":
            self.load_model("span")
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        t_tmp = time.time()
        targets, predes, spans, lss, type_preds, type_g = [], [], [], [], [], []
        for item_id in range(corpus.n_total):
            eval_query, eval_support = corpus.get_batch_meta(
                batch_size=1, shuffle=False
            )

            # train on support examples
            if not self.args.nouse_inner_ft:
                self.inner_update(eval_support[0], lr_curr=lr, inner_steps=steps)

            # eval on pseudo query examples (test example)
            self.model.eval()
            with torch.no_grad():
                logits, e_ls, tmp_eval_loss, _ = self.batch_test(
                    {
                        "input_ids": eval_query[0]["input_ids"],
                        "attention_mask": eval_query[0]["input_mask"],
                        "token_type_ids": eval_query[0]["segment_ids"],
                        "labels": eval_query[0]["label_ids"],
                        "e_mask": eval_query[0]["e_mask"],
                        "e_type_ids": eval_query[0]["e_type_ids"],
                        "e_type_mask": eval_query[0]["e_type_mask"],
                        "entity_types": self.entity_types,
                    }
                )
                lss.append(logits)
                if self.model.train_mode != "type":
                    eval_loss += tmp_eval_loss
                if self.model.train_mode != "span":
                    type_pred, type_ground = self.eval_typing(
                        e_ls, eval_query[0]["e_type_mask"]
                    )
                    type_preds.append(type_pred)
                    type_g.append(type_ground)
                else:
                    e_mask, e_type_ids, e_type_mask, result, types = self.decode_span(
                        logits,
                        eval_query[0]["label_ids"],
                        eval_query[0]["types"],
                        eval_query[0]["input_mask"],
                        viterbi_decoder,
                    )
                    targets.extend(eval_query[0]["entities"])
                    spans.extend(result)

            nb_eval_steps += 1

            self.load_weights(names, weights)
            if item_id % 200 == 0:
                logger.info(
                    "  To sentence {}/{}. Time: {}sec".format(
                        item_id, corpus.n_total, time.time() - t_tmp
                    )
                )

        logger.info("Begin second Stage.")
        if self.eval_mode == "two-stage":
            self.load_model("type")
            names = self.get_names()
            params = self.get_params()
            weights = deepcopy(params)

        if self.train_mode == "add":
            for item_id in range(corpus.n_total):
                eval_query, eval_support = corpus.get_batch_meta(
                    batch_size=1, shuffle=False
                )
                logits = lss[item_id]

                # train on support examples
                self.inner_update(eval_support[0], lr_curr=lr, inner_steps=type_steps)

                # eval on pseudo query examples (test example)
                self.model.eval()
                with torch.no_grad():
                    e_mask, e_type_ids, e_type_mask, result, types = self.decode_span(
                        logits,
                        eval_query[0]["label_ids"],
                        eval_query[0]["types"],
                        eval_query[0]["input_mask"],
                        viterbi_decoder,
                    )

                    _, e_logits, _, tmp_eval_type_loss = self.batch_test(
                        {
                            "input_ids": eval_query[0]["input_ids"],
                            "attention_mask": eval_query[0]["input_mask"],
                            "token_type_ids": eval_query[0]["segment_ids"],
                            "labels": eval_query[0]["label_ids"],
                            "e_mask": e_mask,
                            "e_type_ids": e_type_ids,
                            "e_type_mask": e_type_mask,
                            "entity_types": self.entity_types,
                        }
                    )

                    eval_loss += tmp_eval_type_loss

                    if self.eval_mode == "two-stage":
                        logits, e_ls, tmp_eval_loss, _ = self.batch_test(
                            {
                                "input_ids": eval_query[0]["input_ids"],
                                "attention_mask": eval_query[0]["input_mask"],
                                "token_type_ids": eval_query[0]["segment_ids"],
                                "labels": eval_query[0]["label_ids"],
                                "e_mask": eval_query[0]["e_mask"],
                                "e_type_ids": eval_query[0]["e_type_ids"],
                                "e_type_mask": eval_query[0]["e_type_mask"],
                                "entity_types": self.entity_types,
                            }
                        )

                        type_pred, type_ground = self.eval_typing(
                            e_ls, eval_query[0]["e_type_mask"]
                        )
                        type_preds.append(type_pred)
                        type_g.append(type_ground)
                taregt, p = self.decode_entity(
                    e_logits, result, types, eval_query[0]["entities"]
                )
                predes.extend(p)

                self.load_weights(names, weights)
                if item_id % 200 == 0:
                    logger.info(
                        "  To sentence {}/{}. Time: {}sec".format(
                            item_id, corpus.n_total, time.time() - t_tmp
                        )
                    )

        eval_loss = eval_loss / nb_eval_steps
        if self.is_debug:
            joblib.dump([targets, predes, spans], "debug/f1.pkl")
        store_dir = self.args.model_dir if self.args.model_dir else self.args.result_dir
        joblib.dump(
            [targets, predes, spans],
            "{}/{}_{}_preds.pkl".format(store_dir, "all", set_type),
        )
        joblib.dump(
            [type_g, type_preds],
            "{}/{}_{}_preds.pkl".format(store_dir, "typing", set_type),
        )
        pred = [[jj[:-1] for jj in ii] for ii in predes]
        p, r, f1 = self.cacl_f1(targets, pred)
        pred = [
            [jj[:-1] for jj in ii if jj[-1] > self.args.type_threshold] for ii in predes
        ]
        p_t, r_t, f1_t = self.cacl_f1(targets, pred)

        span_p, span_r, span_f1 = self.cacl_f1(
            [[(jj[0], jj[1]) for jj in ii] for ii in targets], spans
        )
        type_p, type_r, type_f1 = self.cacl_f1(type_g, type_preds)

        results = {
            "loss": eval_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
            "span_p": span_p,
            "span_r": span_r,
            "span_f1": span_f1,
            "type_p": type_p,
            "type_r": type_r,
            "type_f1": type_f1,
            "precision_threshold": p_t,
            "recall_threshold": r_t,
            "f1_threshold": f1_t,
        }

        logger.info("***** Eval results %s-%s *****", mode, set_type)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info(
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
            results["precision"] * 100,
            results["recall"] * 100,
            results["f1"] * 100,
            results["span_p"] * 100,
            results["span_r"] * 100,
            results["span_f1"] * 100,
            results["type_p"] * 100,
            results["type_r"] * 100,
            results["type_f1"] * 100,
            results["precision_threshold"] * 100,
            results["recall_threshold"] * 100,
            results["f1_threshold"] * 100,
        )

        return results, preds

    def save_model(self, result_dir, fn_prefix, max_seq_len, mode: str = "all"):
        # Save a trained model and the associated configuration
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        output_model_file = os.path.join(
            result_dir, "{}_{}_{}".format(fn_prefix, mode, WEIGHTS_NAME)
        )
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(result_dir, CONFIG_NAME)
        with open(output_config_file, "w", encoding="utf-8") as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        model_config = {
            "bert_model": self.bert_model,
            "do_lower": False,
            "max_seq_length": max_seq_len,
            "num_labels": len(self.label_list) + 1,
            "label_map": label_map,
        }
        json.dump(
            model_config,
            open(
                os.path.join(result_dir, f"{mode}-model_config.json"),
                "w",
                encoding="utf-8",
            ),
        )
        if mode == "type":
            joblib.dump(
                self.entity_types, os.path.join(result_dir, "type_embedding.pkl")
            )

    def save_best_model(self, result_dir: str, mode: str):
        output_model_file = os.path.join(result_dir, "en_tmp_{}".format(WEIGHTS_NAME))
        config_name = os.path.join(result_dir, "tmp-model_config.json")
        shutil.copy(output_model_file, output_model_file.replace("tmp", mode))
        shutil.copy(config_name, config_name.replace("tmp", mode))

    def load_model(self, mode: str = "all"):
        if not self.model_dir:
            return
        model_dir = self.model_dir
        logger.info(f"********** Loading saved {mode} model **********")
        output_model_file = os.path.join(
            model_dir, "en_{}_{}".format(mode, WEIGHTS_NAME)
        )
        self.model = BertForTokenClassification_.from_pretrained(
            self.bert_model, num_labels=len(self.label_list), output_hidden_states=True
        )
        self.model.set_config(
            self.args.use_classify,
            self.args.distance_mode,
            self.args.similar_k,
            self.args.shared_bert,
            mode,
        )
        self.model.to(self.args.device)
        self.model.load_state_dict(torch.load(output_model_file, map_location="cuda"))
        self.layer_set()

    def decode_span(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        types,
        mask: torch.Tensor,
        viterbi_decoder=None,
    ):
        if self.is_debug:
            joblib.dump([logits, target, self.label_list], "debug/span.pkl")
        device = target.device
        K = max([len(ii) for ii in types])
        if viterbi_decoder:
            N = target.shape[0]
            B = 16
            result = []
            for i in range((N - 1) // B + 1):
                tmp_logits = torch.tensor(logits[i * B : (i + 1) * B]).to(target.device)
                if len(tmp_logits.shape) == 2:
                    tmp_logits = tmp_logits.unsqueeze(0)
                tmp_target = target[i * B : (i + 1) * B]
                log_probs = nn.functional.log_softmax(
                    tmp_logits.detach(), dim=-1
                )  # batch_size x max_seq_len x n_labels
                pred_labels = viterbi_decoder.forward(
                    log_probs, mask[i * B : (i + 1) * B], tmp_target
                )

                for ii, jj in zip(pred_labels, tmp_target.detach().cpu().numpy()):
                    left, right, tmp = 0, 0, []
                    while right < len(jj) and jj[right] == self.ignore_token_label_id:
                        tmp.append(-1)
                        right += 1
                    while left < len(ii):
                        tmp.append(ii[left])
                        left += 1
                        right += 1
                        while (
                            right < len(jj) and jj[right] == self.ignore_token_label_id
                        ):
                            tmp.append(-1)
                            right += 1
                    result.append(tmp)
        target = target.detach().cpu().numpy()
        B, T = target.shape
        if not viterbi_decoder:
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, -1)

        if self.label_list == ["O", "B", "I"]:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == self.pad_token_label_id
                ):
                    max_pad -= 1
                while idx < max_pad:
                    if target[ii][idx] == self.ignore_token_label_id or (
                        result[ii][idx] != 1
                    ):
                        idx += 1
                        continue
                    e = idx
                    while e < max_pad - 1 and (
                        target[ii][e + 1] == self.ignore_token_label_id
                        or result[ii][e + 1] in [self.ignore_token_label_id, 2]
                    ):
                        e += 1
                    tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)
        elif self.label_list == ["O", "B", "I", "E", "S"]:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == self.pad_token_label_id
                ):
                    max_pad -= 1
                while idx < max_pad:
                    if target[ii][idx] == self.ignore_token_label_id or (
                        result[ii][idx] not in [1, 4]
                    ):
                        idx += 1
                        continue
                    e = idx
                    while (
                        e < max_pad - 1
                        and result[ii][e] not in [3, 4]
                        and (
                            target[ii][e + 1] == self.ignore_token_label_id
                            or result[ii][e + 1] in [self.ignore_token_label_id, 2, 3]
                        )
                    ):
                        e += 1
                    if e < max_pad and result[ii][e] in [3, 4]:
                        while (
                            e < max_pad - 1
                            and target[ii][e + 1] == self.ignore_token_label_id
                        ):
                            e += 1
                        tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)
        M = max([len(ii) for ii in res])
        e_mask = np.zeros((B, M, T), np.int8)
        e_type_mask = np.zeros((B, M, K), np.int8)
        e_type_ids = np.zeros((B, M, K), np.int)
        for ii in range(B):
            for idx, (s, e) in enumerate(res[ii]):
                e_mask[ii][idx][s : e + 1] = 1
            types_set = types[ii]
            if len(res[ii]):
                e_type_ids[ii, : len(res[ii]), : len(types_set)] = [types_set] * len(
                    res[ii]
                )
            e_type_mask[ii, : len(res[ii]), : len(types_set)] = np.ones(
                (len(res[ii]), len(types_set))
            )
        return (
            torch.tensor(e_mask).to(device),
            torch.tensor(e_type_ids, dtype=torch.long).to(device),
            torch.tensor(e_type_mask).to(device),
            res,
            types,
        )

    def decode_entity(self, e_logits, result, types, entities):
        if self.is_debug:
            joblib.dump([e_logits, result, types, entities], "debug/e.pkl")
        target, preds = entities, []
        B = len(e_logits)
        logits = e_logits

        for ii in range(B):
            tmp = []
            tmp_res = result[ii]
            tmp_types = types[ii]
            for jj in range(len(tmp_res)):
                lg = logits[ii][jj, : len(tmp_types)]
                tmp.append((*tmp_res[jj], tmp_types[np.argmax(lg)], lg[np.argmax(lg)]))
            preds.append(tmp)
        return target, preds

    def cacl_f1(self, targets: list, predes: list):
        tp, fp, fn = 0, 0, 0
        for ii, jj in zip(targets, predes):
            ii, jj = set(ii), set(jj)
            same = ii - (ii - jj)
            tp += len(same)
            fn += len(ii - jj)
            fp += len(jj - ii)
        p = tp / (fp + tp + 1e-10)
        r = tp / (fn + tp + 1e-10)
        return p, r, 2 * p * r / (p + r + 1e-10)

    def eval_typing(self, e_logits, e_type_mask):
        e_logits = e_logits
        e_type_mask = e_type_mask.detach().cpu().numpy()
        if self.is_debug:
            joblib.dump([e_logits, e_type_mask], "debug/typing.pkl")

        N = len(e_logits)
        B_S = 16
        res = []
        for i in range((N - 1) // B_S + 1):
            tmp_e_logits = np.argmax(e_logits[i * B_S : (i + 1) * B_S], -1)
            B, M = tmp_e_logits.shape
            tmp_e_type_mask = e_type_mask[i * B_S : (i + 1) * B_S][:, :M, 0]
            res.extend(tmp_e_logits[tmp_e_type_mask == 1])
        ground = [0] * len(res)
        return enumerate(res), enumerate(ground)
