# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path
import json
from seqeval.metrics import precision_score, recall_score
from seqeval.scheme import IOB2

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from models.seqtagger import NERTagger
from utils.utils_ner import *

class NERTrainer():
    def __init__(self, args, processor=None):
        super(NERTrainer, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.lower_case)
        if processor is None:
            self.processor = DataUtils(self.tokenizer, args.max_seq_length)
        else:
            self.processor = processor

        self.args.label2idx = self.processor.label2idx
        print(self.args.label2idx)
        self.args.num_tag = len(self.args.label2idx)
        # ner encoder
        self.ner_encoder = AutoModel.from_pretrained(args.model_name_or_path)
        self.args.hidden_size = self.ner_encoder.config.hidden_size
        # ner tagger
        self.tagger = NERTagger(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        return

    def reset(self):
        self.ner_encoder = AutoModel.from_pretrained(self.args.model_name_or_path)
        print("===== reinit from {} ======".format(self.args.model_name_or_path))
        self.tagger = NERTagger(self.args)
        return

    def load_partial_model(self, ckpt, part_name):
        if part_name == "tagger":
            # allow without crf
            self.tagger.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
        elif part_name == "ner_encoder":
            self.ner_encoder.load_state_dict(torch.load(ckpt, map_location=self.device))
        else:
            raise ValueError
        print(f"load {part_name} from {ckpt} !!")
        return     

    def load_model(self, ckpt_dir):
        load_names =  ["tagger", "ner_encoder"]
        for pname in load_names:
            self.load_partial_model(os.path.join(ckpt_dir, pname + ".pth"), pname)
        return

    def save_partial_model(self, pmodel, ckpt):
        model_to_save = pmodel.module if hasattr(pmodel, "module") else pmodel
        torch.save(model_to_save.state_dict(), ckpt)
        return

    def save_model(self, ckpt_dir):
        self.save_partial_model(self.tagger, os.path.join(ckpt_dir, "tagger.pth"))
        self.save_partial_model(self.ner_encoder, os.path.join(ckpt_dir, "ner_encoder.pth"))
        return

    def prepare_model(self, device, n_gpu):
        self.ner_encoder = self.ner_encoder.to(device)
        self.tagger = self.tagger.to(device)
        if n_gpu > 1 and (not hasattr(self.ner_encoder, "module")):
            self.ner_encoder = nn.DataParallel(self.ner_encoder)
            self.tagger = nn.DataParallel(self.tagger)
        return

    def set_zero_grad(self):
        self.tagger.zero_grad()
        self.ner_encoder.zero_grad()
        return

    def set_mode(self, mode):
        assert mode in ["train", "eval"]
        if mode == "train":
            self.ner_encoder.train()
            self.tagger.train()
        else:
            self.ner_encoder.eval()
            self.tagger.eval()
        return

    def prepare_ner_train(self, train_data_size):
        num_train_steps = train_data_size // self.args.train_batch_size * self.args.train_ner_epochs
        warmup_steps = int(self.args.warmup_proportion * num_train_steps)
        ner_optimizer = self.get_ner_optimizer(self.args)
        ner_scheduler =  get_linear_schedule_with_warmup(ner_optimizer, warmup_steps, num_train_steps)
        self.set_mode("train")
        self.set_zero_grad()
        return ner_optimizer, ner_scheduler

    def change_grad(self, module, keep_grad):
        for n, p in module.named_parameters():
            p.requires_grad = keep_grad
        return

    def get_group_params(self, module, no_decay):
        zero_decay_params = []
        decay_params = []
        for n, p in module.named_parameters():
            if p.requires_grad:
                if any(nd in n for nd in no_decay):
                    zero_decay_params.append(p)
                else:
                    decay_params.append(p)
        return zero_decay_params, decay_params

    def fix_partial_emb(self, module, no_grad):
        for n, p in module.named_parameters():
            if any(nd in n for nd in no_grad):
                p.requires_grad = False
                print("froze {}".format(n))
        return

    def get_ner_optimizer(self, train_args):
        self.change_grad(self.tagger, True)
        self.change_grad(self.ner_encoder, True)
        # we don't learn the crf layer, just apply it for viterbi decoding
        self.fix_partial_emb(self.tagger, ["crf_layer"])
        # freeze first three layers and embedding layer
        if train_args.frz_bert_layers >= 0:
            no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(train_args.frz_bert_layers)]
            self.fix_partial_emb(self.ner_encoder, no_grad)
        no_decay = ["bias", "LayerNorm.weight"]
        zero_decay_params, decay_params = self.get_group_params(self.ner_encoder, no_decay)
        zero_tagger, decay_tagger = self.get_group_params(self.tagger, no_decay)
        zero_decay_params += zero_tagger
        decay_params += decay_tagger
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": train_args.weight_decay, "lr": train_args.ner_lr},
            {"params": zero_decay_params, "weight_decay": 0.0, "lr": train_args.ner_lr},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.999))
        return optimizer
    
    
    def ner_step(self, input_ids=None,
                    input_embs=None,
                    attention_mask=None,
                    word_to_piece_inds=None,
                    word_to_piece_ends=None,
                    sent_lens=None,
                    word_labels=None,
                    weights=None):
                    
        if input_ids is not None:
            token_outputs = self.ner_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            token_outputs = self.ner_encoder(inputs_embeds=input_embs, attention_mask=attention_mask)
        token_hidden = token_outputs[0]
        word_logits, decodeIdx, word_hidden, ner_loss = self.tagger(token_hidden, word_to_piece_inds, word_to_piece_ends, sent_lens, 
                                                                    labels=word_labels, 
                                                                    weights=weights)
        if self.n_gpu > 1:
            ner_loss = ner_loss.mean()
        return ner_loss, decodeIdx, word_logits, token_hidden, word_hidden


    def eval_ner(self, dataset, out_file=None, res_file=None, ori_sents=None, return_hidden=False):
        rng_state = torch.get_rng_state()
        dataloader = self.processor.get_loader(dataset, batch_size=self.args.eval_batch_size, shuffle=False)
        self.set_mode("eval")
        self.prepare_model(self.device, self.n_gpu)
        ori_pred_list = []
        ori_logit_list = []
        word_hidden_list = []
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        for _, inputs in enumerate(dataloader):
            ner_batch = {
                         "input_ids": inputs[1].to(self.device),
                         "attention_mask": inputs[2].to(self.device),
                         "word_to_piece_inds": inputs[3].to(self.device),
                         "word_to_piece_ends": inputs[4].to(self.device),
                         "sent_lens": inputs[5].to(self.device),
                         }
            _, decodeIdx, logits, _, word_hidden = self.ner_step(**ner_batch)
            tmp = decodeIdx.detach().masked_fill(inputs[6].to(self.device).eq(-100), -100)
            ner_loss = loss_fct(logits.view(-1, logits.size(-1)), tmp.view(-1))
            ori_pred_list.append(decodeIdx.detach().cpu())
            ori_logit_list.append(logits.detach().cpu())
            word_hidden_list.append(word_hidden.detach().cpu())

        ori_pred_labels = torch.cat(ori_pred_list, dim=0)
        ori_pred_labels = ori_pred_labels.masked_fill(dataset.data["labels"].eq(-100), -100)
        ori_pred_logits = torch.cat(ori_logit_list, dim=0)
        word_hidden_list = torch.cat(word_hidden_list, dim=0)
        
        ori_y_pred, y_true, ori_f1 = self.processor.performance_report(ori_pred_labels, dataset.data["labels"], print_log=True)
        ori_prec = precision_score(y_true, ori_y_pred, scheme=IOB2)
        ori_recall = recall_score(y_true, ori_y_pred, scheme=IOB2)

        if out_file is not None:
            if ori_sents is not None:
                with open(out_file, mode="w", encoding="utf-8") as fp:
                    lines = []
                    for words, preds in zip(ori_sents, ori_y_pred):
                        for w, t in zip(words, preds):
                            lines.append("{}\t{}\n".format(w, t))
                        lines.append("\n")
                    fp.writelines(lines)   
            else:
                with open(out_file, mode="w", encoding="utf-8") as fp:
                    lines = []
                    for preds in ori_y_pred:
                        lines.append(" ".join(preds) + "\n")
                    fp.writelines(lines)
        if res_file is not None:
            with open(res_file, mode="w", encoding="utf-8") as fp:
                json.dump({"precision": ori_prec, "recall": ori_recall, "f1": ori_f1}, fp)
        print("{:.4f}\t{:.4f}\t{:.4f}\n".format(ori_prec, ori_recall, ori_f1))
        outputs = (ori_prec, ori_recall, ori_f1, ori_pred_labels, ori_y_pred, ori_pred_logits)
        if return_hidden:
            outputs = outputs + (word_hidden_list,)

        torch.set_rng_state(rng_state)
        return outputs
    
    def train_ner(self, train_dataset, src_dev_dataset, eval_dataset, output_dir):
        print("\n\n******* NER-training *******\n\n")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.prepare_model(self.device, self.n_gpu)

        ner_optimizer, ner_scheduler = self.prepare_ner_train(train_data_size=len(train_dataset))
        train_dataloader = self.processor.get_loader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        use_weight = "weights" in train_dataset.data

        epoch_ner_loss = AverageMeter()
        i = 0
        best_f1 = -100
        self.set_zero_grad()

        val_steps = self.args.val_steps if self.args.val_steps > 0 else len(train_dataloader)
        for epoch in range(self.args.train_ner_epochs):
            print("Epoch {}, step {}".format(epoch, len(train_dataloader)))
            for step, ner_inputs in enumerate(train_dataloader):
                max_src_len = int(torch.sum(ner_inputs[2], dim=1).max().item())
                self.set_mode("train")
                src_ner_batch = {   
                             "input_ids": ner_inputs[1][:, :max_src_len].to(self.device),                     
                             "attention_mask": ner_inputs[2][:, :max_src_len].to(self.device),
                             "word_to_piece_inds": ner_inputs[3][:, :max_src_len].to(self.device),
                             "word_to_piece_ends": ner_inputs[4][:, :max_src_len].to(self.device),
                             "sent_lens": ner_inputs[5].to(self.device),
                             "word_labels": ner_inputs[6][:, :max_src_len].to(self.device),
                }
                src_ner_batch["input_ids"] = ner_inputs[1][:, :max_src_len].to(self.device)

                weights = None
                if use_weight:
                    weights = ner_inputs[7][:, :max_src_len].to(self.device)
                src_ner_batch["weights"] = weights

                ner_loss, _, word_logits, _, _ = self.ner_step(**src_ner_batch)
                epoch_ner_loss.update(ner_loss.item())
                ner_loss.backward()
                nn.utils.clip_grad_norm_(self.tagger.parameters(), self.args.ner_max_grad_norm)
                nn.utils.clip_grad_norm_(self.ner_encoder.parameters(), self.args.ner_max_grad_norm)
                ner_optimizer.step()
                ner_scheduler.step()
                self.set_zero_grad()

                if (i % self.args.logging_steps) == 0:
                    print("Epoch {} Step {}, avg src ner loss {}".format(epoch, step, epoch_ner_loss.avg))
                    if self.args.save_log:
                        wandb.log({"ner/train_loss": epoch_ner_loss.avg})
                    epoch_ner_loss.reset()
                i += 1
                
                if (i % val_steps) == 0:
                    print("============eval on src dev file============")
                    dev_res = self.eval_ner(src_dev_dataset,
                        out_file=os.path.join(output_dir, "src_dev_epoch_{}.txt".format(epoch)),
                        res_file=os.path.join(output_dir, "src_dev_epoch_{}_metrics.json".format(epoch)))
                    src_dev_f1 = dev_res[2]

                    print("============eval on tgt test file============")
                    res = self.eval_ner(eval_dataset,
                        out_file=os.path.join(output_dir, "test_epoch_{}.txt".format(epoch)),
                        res_file=os.path.join(output_dir, "test_epoch_{}_metrics.json".format(epoch)))
                    test_f1 = res[2]
                    
                    if src_dev_f1 > best_f1:
                        best_f1 = src_dev_f1
                        print("better model!!!!!!!!!!!!!!!")
                        if self.args.select_ckpt and self.args.save_ckpt:
                            self.save_model(output_dir)
                        with open(os.path.join(output_dir, "best_test.json"), mode="w") as fp:
                            json.dump({"p": res[0], "r": res[1], "f1": res[2],
                                       "devp": dev_res[0], "devr": dev_res[1], "devf1": dev_res[2]}, fp)

        if not self.args.select_ckpt and self.args.save_ckpt:
            self.save_model(output_dir)
        return
    