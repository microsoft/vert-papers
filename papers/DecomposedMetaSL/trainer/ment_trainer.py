import json
import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from util.log_utils import write_ep_ment_log_json, write_ment_log, eval_ment_log, cal_prf
from tqdm import tqdm

class MentTrainer:
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

    def eval(self, model, device, dataloader, load_ckpt=None, update_iter=0, learning_rate=3e-5, eval_iter=-1):
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('[ERROR] Ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
        model.eval()
        tot_metric_logs = defaultdict(int)
        eval_loss = 0
        tot_seq_cnt = 0
        tot_logs = {}
        eval_batchs = iter(dataloader)
        tot_metric_logs = defaultdict(int)
        eval_loss = 0
        if eval_iter <= 0:
            eval_iter = len(dataloader.dataset.sampler)
        tot_seq_cnt = 0
        tot_logs = {}
        print("[eval] update {} steps | total {} episode".format(update_iter, eval_iter))
        for batch_id in range(eval_iter):
            batch = next(eval_batchs)
            batch['support'] = dataloader.dataset.batch_to_device(batch['support'], device)
            batch['query'] = dataloader.dataset.batch_to_device(batch['query'], device)
            if update_iter > 0:
                res = model.forward_meta(batch, update_iter, learning_rate, "test")
                eval_loss += res['loss']
            else:
                res = model.forward_sup(batch, "test")
                eval_loss += res['loss'].item()
            metric_logs, logs = model.seqment_eval(res["preds"], batch["query"], model.idx2label, model.schema)
            tot_seq_cnt += batch['query']['word'].size(0)
            for k, v in logs.items():
                if k not in tot_logs:
                    tot_logs[k] = []
                tot_logs[k] += v
            for k, v in metric_logs.items():
                tot_metric_logs[k] += v

        ment_p, ment_r, ment_f1 = cal_prf(tot_metric_logs["ment_hit_cnt"], tot_metric_logs["ment_pred_cnt"], tot_metric_logs["ment_gold_cnt"])
        print("seq num:", tot_seq_cnt, "hit cnt:", tot_metric_logs["ment_hit_cnt"], "pred cnt:", tot_metric_logs["ment_pred_cnt"], "gold cnt:", tot_metric_logs["ment_gold_cnt"])
        print("avg hit:", tot_metric_logs["ment_hit_cnt"] / tot_seq_cnt, "avg pred:", tot_metric_logs["ment_pred_cnt"] / tot_seq_cnt, "avg gold:", tot_metric_logs["ment_gold_cnt"] / tot_seq_cnt)
        ent_p, ent_r, ent_f1 = 0, 0, 0
        model.train()
        return eval_loss / eval_iter, ment_p, ment_r, ment_f1, ent_p, ent_r, ent_f1, tot_logs




    def train(self, model, training_args, device, trainloader, devloader, eval_log_fn=None, load_ckpt=None, ignore_log=True):
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
        parameters_to_optimize = list(model.named_parameters())
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
        train_acc = 0
        iter_sample = 0
        tot_metric_logs = defaultdict(int)
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
                res = model.forward_meta(batch, training_args.train_inner_steps, lr_inner, "train")
                for g in res['grads']:
                    model.load_gradients(res['names'], g)
                train_loss += res['loss']
            else:
                res = model.forward_sup(batch, "train")
                loss = res['loss']
                loss.backward()
                train_loss += loss.item()
            if training_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pred = torch.cat(res['preds'], dim=0).detach().cpu().numpy()
            gold = torch.cat(res['golds'], dim=0).detach().cpu().numpy()

            acc = model.span_accuracy(pred, gold)
            train_acc += acc

            iter_sample += 1
            if not ignore_log:
                metric_logs, logs = model.seqment_eval(res["preds"], batch["query"], model.idx2label, model.schema)

                for k, v in metric_logs.items():
                    tot_metric_logs[k] += v


            if it % 100 == 0 or it % training_args.log_steps == 0:
                if not ignore_log:
                    precision, recall, f1 = cal_prf(tot_metric_logs["ment_hit_cnt"], tot_metric_logs["ment_pred_cnt"],
                                                        tot_metric_logs["ment_gold_cnt"])

                    print('step: {0:4} | loss: {1:2.6f} | span acc {2:.5f} [ENTITY] precision: {3:3.4f}, recall: {4:3.4f}, f1: {5:3.4f}'\
                        .format(it, train_loss / iter_sample, train_acc / iter_sample, precision, recall, f1) + '\r')
                else:
                    print('step: {0:4} | loss: {1:2.6f} | span acc {2:.5f}'
                        .format(it, train_loss / iter_sample, train_acc / iter_sample) + '\r')                 
                train_loss = 0
                train_acc = 0
                iter_sample = 0
                tot_metric_logs = defaultdict(int)

            if it % training_args.val_steps == 0:
                eval_loss, eval_ment_p, eval_ment_r, eval_ment_f1, eval_p, eval_r, eval_f1, eval_logs = self.eval(model, device, devloader, update_iter=training_args.eval_inner_steps, learning_rate=training_args.eval_inner_lr, eval_iter=training_args.dev_iter)
                print('[EVAL] step: {0:4} | loss: {1:2.6f} | [MENTION] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                    .format(it, eval_loss, eval_ment_p, eval_ment_r, eval_ment_f1) + '\r')
                if eval_ment_f1 > best_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()},
                                os.path.join(training_args.output_dir, "model.pth.tar"))
                    best_f1 = eval_ment_f1
                    if eval_log_fn is not None:
                        write_ment_log(devloader.dataset.samples, eval_logs, eval_log_fn)
            if training_args.train_iter == it:
                break
        print("\n####################\n")
        print("Finish training ")
        return