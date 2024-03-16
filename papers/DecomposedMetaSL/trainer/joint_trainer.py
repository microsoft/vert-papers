import json
import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from util.log_utils import write_ep_ment_log_json, write_ment_log, eval_ment_log
from util.log_utils import write_ent_log, write_ent_pred_json, eval_ent_log, cal_prf

from tqdm import tqdm

class JointTrainer:
    def __init__(self):
        return

    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def get_learning_rate(self, lr, progress, warmup, schedule="linear"):
        if schedule == "linear":
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress - 1.0) / (warmup - 1.0), 0.0)
        return lr

    def check_input_ment(self, query):
        pred_cnt, gold_cnt, hit_cnt = 0, 0, 0
        for i in range(len(query['spans'])):
            pred_ments = query['span_indices'][i][query['span_mask'][i].eq(1)].detach().cpu().tolist()
            gold_ments = [x[1:] for x in query['spans'][i]]
            pred_cnt += len(pred_ments)
            gold_cnt += len(gold_ments)
            for x in gold_ments:
                if x in pred_ments:
                    hit_cnt += 1
        return pred_cnt, gold_cnt, hit_cnt

    def eval(self, model, device, dataloader, load_ckpt=None, ment_update_iter=0, type_update_iter=0, learning_rate=3e-5, eval_iter=-1, eval_mode="test-twostage", overlap=False, threshold=-1):
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('[ERROR] Ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
        model.eval()
        eval_loss = 0
        tot_seq_cnt = 0
        tot_ment_logs = {}
        tot_type_logs = {}
        eval_batchs = iter(dataloader)
        tot_ment_metric_logs = defaultdict(int)
        tot_type_metric_logs = defaultdict(int)
        eval_loss = 0
        if eval_iter <= 0:
            eval_iter = len(dataloader.dataset.sampler)
        tot_seq_cnt = 0
        print("[eval] update {} steps | total {} episode".format(ment_update_iter, eval_iter))
        for batch_id in tqdm(range(eval_iter)):
            batch = next(eval_batchs)
            batch['support'] = dataloader.dataset.batch_to_device(batch['support'], device)
            batch['query'] = dataloader.dataset.batch_to_device(batch['query'], device)

            res = model.forward_joint_meta(batch, ment_update_iter, learning_rate, eval_mode)
            eval_loss += res['loss']

            if eval_mode == 'test-twostage':
                batch["query"]["span_indices"] = res['pred_spans']
                batch["query"]["span_mask"] = res['pred_masks']
            ment_metric_logs, ment_logs = model.seqment_eval(res["ment_preds"], batch["query"], model.ment_idx2label, model.schema)
            type_metric_logs, type_logs = model.greedy_eval(res['type_logits'], batch['query'], overlap=overlap, threshold=threshold)

            tot_seq_cnt += batch['query']['word'].size(0)
            for k, v in ment_logs.items():
                if k not in tot_ment_logs:
                    tot_ment_logs[k] = []
                tot_ment_logs[k] += v
            for k, v in type_logs.items():
                if k not in tot_type_logs:
                    tot_type_logs[k] = []
                tot_type_logs[k] += v

            for k, v in ment_metric_logs.items():
                tot_ment_metric_logs[k] += v
            for k, v in type_metric_logs.items():
                tot_type_metric_logs[k] += v

        ment_p, ment_r, ment_f1 = cal_prf(tot_ment_metric_logs["ment_hit_cnt"], tot_ment_metric_logs["ment_pred_cnt"], tot_ment_metric_logs["ment_gold_cnt"])
        print("seq num:", tot_seq_cnt, "hit cnt:", tot_ment_metric_logs["ment_hit_cnt"], "pred cnt:", tot_ment_metric_logs["ment_pred_cnt"], "gold cnt:", tot_ment_metric_logs["ment_gold_cnt"])
        print("avg hit:", tot_ment_metric_logs["ment_hit_cnt"] / tot_seq_cnt, "avg pred:", tot_ment_metric_logs["ment_pred_cnt"] / tot_seq_cnt, "avg gold:", tot_ment_metric_logs["ment_gold_cnt"] / tot_seq_cnt)
        ent_p, ent_r, ent_f1 = cal_prf(tot_type_metric_logs["ent_hit_cnt"], tot_type_metric_logs["ent_pred_cnt"], tot_type_metric_logs["ent_gold_cnt"])
        model.train()
        return eval_loss / eval_iter, ment_p, ment_r, ment_f1, ent_p, ent_r, ent_f1, tot_ment_logs, tot_type_logs


    def train(self, model, training_args, device, trainloader, devloader, load_ckpt=None, ignore_log=True, dev_pred_fn=None, dev_log_fn=None):
        if load_ckpt is not None:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('[ERROR] Ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
            print("load ckpt from {}".format(load_ckpt))
        # Init optimizer
        print('Use bert optim!')
        parameters_to_optimize = list(filter(lambda x: x[1].requires_grad, model.named_parameters()))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        parameters_groups = [
            {'params': [p for n, p in parameters_to_optimize if ("bert." in n) and (not any(nd in n for nd in no_decay))],
             'lr': training_args.bert_learning_rate, 'weight_decay': training_args.bert_weight_decay},
            {'params': [p for n, p in parameters_to_optimize if ("bert." in n) and any(nd in n for nd in no_decay)],
             'lr': training_args.bert_learning_rate, 'weight_decay': 0},
            {'params': [p for n, p in parameters_to_optimize if "bert." not in n],
             'lr': training_args.learning_rate, 'weight_decay': training_args.weight_decay}
        ]

        optimizer = torch.optim.AdamW(parameters_groups)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_step,
                                                    num_training_steps=training_args.train_iter)
        model.train()
        model.zero_grad()

        best_f1 = 0.0
        train_loss = 0.0
        train_ment_loss = 0.0
        train_type_loss = 0.0
        train_type_acc = 0
        train_ment_acc = 0
        iter_sample = 0
        it = 0
        train_batchs = iter(trainloader)
        for _ in range(training_args.train_iter):
            it += 1
            torch.cuda.empty_cache()
            model.train()
            batch = next(train_batchs)
            batch['support'] = trainloader.dataset.batch_to_device(batch['support'], device)
            batch['query'] = trainloader.dataset.batch_to_device(batch['query'], device)

            if training_args.use_maml:
                progress = 1.0 * (it - 1) / training_args.train_iter
                lr_inner = self.get_learning_rate(
                    training_args.train_inner_lr, progress, training_args.warmup_prop_inner
                )
                res = model.forward_joint_meta(batch, training_args.train_inner_steps, lr_inner, "train")

                for g in res['grads']:
                    model.load_gradients(res['names'], g)
                train_loss += res['loss']
                train_ment_loss += res['ment_loss']
                train_type_loss += res['type_loss']
            else:
                raise ValueError
            if training_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            type_pred = torch.cat(res['type_preds'], dim=0).detach().cpu().numpy()
            type_gold = torch.cat(res['type_golds'], dim=0).detach().cpu().numpy()
            ment_pred = torch.cat(res['ment_preds'], dim=0).detach().cpu().numpy()
            ment_gold = torch.cat(res['ment_golds'], dim=0).detach().cpu().numpy()
            type_acc = model.span_accuracy(type_pred, type_gold)
            ment_acc = model.span_accuracy(ment_pred, ment_gold)
            train_type_acc += type_acc
            train_ment_acc += ment_acc

            iter_sample += 1
            if not ignore_log:
                raise ValueError

            if it % 100 == 0 or it % training_args.log_steps == 0:
                if not ignore_log:
                    raise ValueError
                else:
                    print('step: {0:4} | loss: {1:2.6f} | ment loss: {2:2.6f} | type loss: {3:2.6f} | ment acc {4:.5f} | type acc {5:.5f}'
                        .format(it, train_loss / iter_sample, train_ment_loss / iter_sample, train_type_loss / iter_sample, train_ment_acc / iter_sample, train_type_acc / iter_sample) + '\r')                 
                train_loss = 0
                train_ment_loss = 0
                train_type_loss = 0
                train_ment_acc = 0
                train_type_acc = 0
                iter_sample = 0

            if it % training_args.val_steps == 0:
                eval_loss, eval_ment_p, eval_ment_r, eval_ment_f1, eval_p, eval_r, eval_f1, eval_ment_logs, eval_logs = self.eval(model, device, devloader, ment_update_iter=training_args.eval_ment_inner_steps, type_update_iter=training_args.eval_type_inner_steps, learning_rate=training_args.eval_inner_lr, eval_iter=training_args.dev_iter, eval_mode="test-twostage")
                print('[EVAL] step: {0:4} | loss: {1:2.6f} | [MENTION] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f} [ENTITY] precision: {5:3.4f}, recall: {6:3.4f}, f1: {7:3.4f}'\
                    .format(it, eval_loss, eval_ment_p, eval_ment_r, eval_ment_f1, eval_p, eval_r, eval_f1) + '\r')
                if eval_f1 > best_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()},
                               os.path.join(training_args.output_dir, "model.pth.tar"))
                    best_f1 = eval_f1
                    if dev_pred_fn is not None:
                        with open(dev_pred_fn, mode="w", encoding="utf-8") as fp:
                            json.dump({"ment_p": eval_ment_p, "ment_r": eval_ment_r, "ment_f1": eval_ment_f1, "precision": eval_p, "recall": eval_r, "f1": eval_f1}, fp)

                    eval_ent_log(devloader.dataset.samples, eval_logs)
            if training_args.train_iter == it:
                break
        print("\n####################\n")
        print("Finish training ")
        return