import json
import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from util.log_utils import write_pos_pred_json

class POSTrainer:
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

    def eval(self, model, device, dataloader, load_ckpt=None, update_iter=0, learning_rate=3e-5, eval_iter=-1, eval_mode="test"):
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('[ERROR] Ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
        model.eval()
        eval_batchs = iter(dataloader)
        tot_metric_logs = defaultdict(int)
        eval_loss = 0
        if eval_iter <= 0:
            eval_iter = len(dataloader.dataset.sampler)
        tot_seq_cnt = 0
        tot_logs = {}
        for batch_id in tqdm(range(eval_iter)):
            batch = next(eval_batchs)
            batch['support'] = dataloader.dataset.batch_to_device(batch['support'], device)
            batch['query'] = dataloader.dataset.batch_to_device(batch['query'], device)
            if update_iter > 0:
                res = model.forward_meta(batch, update_iter, learning_rate, eval_mode)
                eval_loss += res['loss']
            else:
                res = model.forward_proto(batch, eval_mode)
                eval_loss += res['loss'].item()
            metric_logs, logs = model.seq_eval(res['preds'], batch['query'])
            tot_seq_cnt += batch["query"]["seq_len"].size(0)
            logs["support_index"] = batch["support"]["index"]
            logs["support_sentence_num"] = batch["support"]["sentence_num"]
            logs["support_subsentence_num"] = batch["support"]["subsentence_num"]

            for k, v in logs.items():
                if k not in tot_logs:
                    tot_logs[k] = []
                tot_logs[k] += v
            for k, v in metric_logs.items():
                tot_metric_logs[k] += v
        token_acc = tot_metric_logs["hit_cnt"] / tot_metric_logs["gold_cnt"]
        print("seq num:", tot_seq_cnt, "hit cnt:", tot_metric_logs["hit_cnt"], "gold cnt:", tot_metric_logs["gold_cnt"], "acc:", token_acc)
        model.train()
        return eval_loss / eval_iter, token_acc, tot_logs

    def train(self, model, training_args, device, trainloader, devloader, load_ckpt=None, dev_pred_fn=None, dev_log_fn=None, ignore_log=True):
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

        best_f1 = -1
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
                    model.load_gradients(res['names'], g) # loss backward
                train_loss += res['loss']
            else:
                res = model.forward_proto(batch, "train")
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
                metric_logs, logs = model.seq_eval(res["preds"], batch["query"], model.schema)
                for k, v in metric_logs.items():
                    tot_metric_logs[k] += v

            if it % 100 == 0 or it % training_args.log_steps == 0:
                if not ignore_log:
                    acc = tot_metric_logs["hit_cnt"] / tot_metric_logs["gold_cnt"]
                    print('step: {0:4} | loss: {1:2.6f} | span acc {2:.5f} [Token] acc: {3:3.4f}'\
                        .format(it, train_loss / iter_sample, train_acc / iter_sample, acc) + '\r')
                else:
                    print('step: {0:4} | loss: {1:2.6f} | span acc {2:.5f}'\
                        .format(it, train_loss / iter_sample, train_acc / iter_sample) + '\r')  
                train_loss = 0
                train_acc = 0
                iter_sample = 0
                tot_metric_logs = defaultdict(int)

            if it % training_args.val_steps == 0:
                eval_loss, eval_acc, eval_logs = self.eval(model, device, devloader, eval_iter=training_args.dev_iter,
                        update_iter=training_args.eval_inner_steps, learning_rate=training_args.eval_inner_lr, eval_mode=training_args.train_eval_mode)
                print('[EVAL] step: {0:4} | loss: {1:2.6f} | F1: {2:3.4f}'\
                    .format(it, eval_loss, eval_acc) + '\r')

                if eval_acc > best_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()},
                               os.path.join(training_args.output_dir, "model.pth.tar"))
                    best_f1 = eval_acc
                    if dev_pred_fn is not None:
                        write_pos_pred_json(devloader.dataset.samples, eval_logs, dev_pred_fn)
                    print("Best acc: {}".format(best_f1))

            if it > 1003:
                break
        print("\n####################\n")
        print("Finish training ")
        return