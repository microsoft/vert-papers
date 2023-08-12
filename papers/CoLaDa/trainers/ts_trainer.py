# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path
import json
import joblib
from models.knn import WordKNN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from utils.utils_ner import *
from .base_trainer import NERTrainer

def sharpening(soft_labels, temp):
    soft_labels = soft_labels.pow(temp)
    return soft_labels / soft_labels.abs().sum(1, keepdim=True)

def tempereture_softmax(logits, tau):
    return (logits/tau).softmax(-1)

class TSTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.lower_case)
        self.processor = DataUtils(self.tokenizer, args.max_seq_length)
        print("loading dataset...")
        self.src_ner_bert_dataset = self.processor.get_ner_dataset(self.args.train_file)
        self.unlabel_bert_dataset = self.processor.get_ner_dataset(self.args.unlabel_file)
        if self.args.trans_file:
            self.trans_ner_bert_dataset = self.processor.get_ner_dataset(self.args.trans_file) 
        self.src_dev_dataset = self.processor.get_ner_dataset(self.args.src_dev_file)
        
        if args.do_eval: # do test
            self.eval_dataset = self.processor.get_ner_dataset(self.args.test_file)
        print("load dataset done.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.src_trainer = NERTrainer(args, self.processor)
        self.trans_trainer = NERTrainer(args, self.processor)
        self.stu_trainer = NERTrainer(args, self.processor)
        return

    def get_word_repr(self, trainer, dataset, pooling='avg', last_n_layer=1):
        rng_state = torch.get_rng_state()
        dataloader = self.processor.get_loader(dataset, batch_size=self.args.eval_batch_size, shuffle=False)
        trainer.set_mode("eval")
        trainer.prepare_model(self.device, trainer.n_gpu)
        word_hidden_list = []
        for _, inputs in enumerate(dataloader):
            ner_batch = {
                         "input_ids": inputs[1].to(self.device),
                         "attention_mask": inputs[2].to(self.device),
                         "word_to_piece_inds": inputs[3].to(self.device),
                         "word_to_piece_ends": inputs[4].to(self.device),
                         "sent_lens": inputs[5].to(self.device),
                         }
            with torch.no_grad():
                layer_emb = trainer.ner_encoder(input_ids=ner_batch["input_ids"], attention_mask=ner_batch["attention_mask"], output_hidden_states=True)[2][-last_n_layer:]
                token_emb = torch.stack(layer_emb, dim=0).mean(0)
                trans_word_hidden = trainer.tagger.combine(token_emb, ner_batch["word_to_piece_inds"], ner_batch["word_to_piece_ends"], pooling=pooling)
                word_hidden_list.append(trans_word_hidden.detach().cpu())
        word_hidden_list = torch.cat(word_hidden_list, dim=0)
        torch.set_rng_state(rng_state)
        return word_hidden_list
    
    def load_knn(self, trainer, dataset=None, knn_file=None):
        print("reweight translation data with knn confidence!")
        if dataset is None:
            dataset = self.trans_ner_bert_dataset
        trans_word_hidden = self.get_word_repr(trainer, dataset, pooling=self.args.knn_pooling, last_n_layer=self.args.knn_lid)
        
        init_labels = dataset.data["labels"]
        trans_mask = init_labels > -1
        flat_word_reprs = trans_word_hidden[trans_mask]
        flat_word_lbls = init_labels[trans_mask]
        N, L = trans_word_hidden.size(0), trans_word_hidden.size(1)
    
        knner = WordKNN(flat_word_reprs, flat_word_lbls, normalize=True)
        knn_dists, knn_ids, knn_lbls = knner.search_knn(flat_word_reprs, k=self.args.K + 1) #do not contain self...
        word_knn_dists = torch.zeros((N, L, self.args.K), dtype=torch.float32)
        word_knn_golds = torch.zeros((N, L, self.args.K), dtype=torch.long)
        # remove self
        word_knn_dists[trans_mask] = knn_dists[:, 1:]
        word_knn_golds[trans_mask] = knn_lbls[:, 1:]
        
        # change class knn to type level (do not diff B-X and I-X)
        add_idx_list = torch.zeros(self.args.num_tag, dtype=torch.long)
        for cname, idx  in self.processor.label2idx.items():
            if cname == "O":
                add_idx_list[idx] = -10000000
            elif cname[:2] == "B-":
                add_idx_list[idx] = self.processor.label2idx["I-{}".format(cname[2:])]
            else:
                add_idx_list[idx] = self.processor.label2idx["B-{}".format(cname[2:])]
        add_idx_labels = add_idx_list[init_labels.masked_fill(~trans_mask, self.processor.label2idx["O"])]
        align_cnt = word_knn_golds.eq(dataset.data["labels"].unsqueeze(-1)).sum(-1)
        align_cnt += word_knn_golds.eq(add_idx_labels.unsqueeze(-1)).sum(-1)

        max_cnt = torch.zeros(self.args.num_tag)
        for lbl, idx in self.processor.label2idx.items():
            cnt = torch.max(align_cnt[init_labels.eq(idx)]).item()
            max_cnt[idx] = cnt
        # label consistency score
        tmp_weights = (align_cnt  + self.args.smK) / (max_cnt[init_labels.masked_fill(~trans_mask, 0)] + self.args.smK)
        # sigmoid function
        alpha = torch.zeros(self.args.num_tag, dtype=torch.float64)
        for k, cname in self.processor.idx2label.items():
            mask_pos = init_labels.eq(k)
            alpha[k] = tmp_weights[mask_pos].mean() 
            tmp_weights[mask_pos] = tmp_weights[mask_pos] - alpha[k] * self.args.emu
            tmp_weights[mask_pos] = 1 / (1 + torch.exp(- tmp_weights[mask_pos] * self.args.ealpha))
            tmp_weights[mask_pos] = tmp_weights[mask_pos] / torch.max(tmp_weights[mask_pos]).item()
        weights = tmp_weights

        if knn_file is not None:
            joblib.dump({"knn_weight": weights, "knn_cnt": align_cnt,
                        "ori_labels": init_labels}, knn_file)
        return weights

    
    def train_filter_trans(self):
        # load M_tgt to make annotation
        self.src_trainer.load_model(self.args.ckpt_dir)
        res = self.src_trainer.eval_ner(self.trans_ner_bert_dataset)
        p, r, f1, pred_labels = res[:4]
        pred_logits = res[-1]
        pred_probs = torch.softmax(pred_logits, dim=-1)
        src_weights = None
        trans_labels = self.trans_ner_bert_dataset.data["labels"]
        valid_weights = (trans_labels > -1).type(torch.float32)
        tmp = trans_labels.masked_fill(trans_labels < 0, 0).type(torch.long)
        hard_labels = torch.zeros_like(pred_probs, dtype=torch.float32)
        hard_labels = hard_labels.scatter(index=tmp.unsqueeze(-1), value=1.0, dim=-1)
        if self.args.filter_trans == 'reweight':    
            src_weights = None
            self.stu_trainer.load_model(self.args.ckpt_dir)
            max_src_pred_labels = torch.argmax(pred_probs, dim=-1).masked_fill(trans_labels < 0, -100)
            src_trans_dataset = NERDataset(
                {
                    "idx": self.trans_ner_bert_dataset.data["idx"],
                    "words": self.trans_ner_bert_dataset.data["words"], # this is vocab token id, not bert's token id
                    "word_masks": self.trans_ner_bert_dataset.data["word_masks"],
                    "word_to_piece_inds": self.trans_ner_bert_dataset.data["word_to_piece_inds"],
                    "word_to_piece_ends": self.trans_ner_bert_dataset.data["word_to_piece_ends"],
                    "sent_lens": self.trans_ner_bert_dataset.data["sent_lens"],
                    "labels": max_src_pred_labels,
                }
            )
            src_weights = self.load_knn(self.stu_trainer, dataset=src_trans_dataset, knn_file=os.path.join(self.args.output_dir, "srcknn.pkl"))
            soft_labels = self.args.trans_lam * hard_labels + (1 - self.args.trans_lam) * src_weights.unsqueeze(-1) * pred_probs
        else:
            soft_labels = (1 - self.args.trans_lam) * pred_probs + self.args.trans_lam * hard_labels
        final_labels = soft_labels * valid_weights.unsqueeze(-1)
        filter_trans_dataset = NERDataset(
            {
                "idx": self.trans_ner_bert_dataset.data["idx"],
                "words": self.trans_ner_bert_dataset.data["words"], # this is vocab token id, not bert's token id
                "word_masks": self.trans_ner_bert_dataset.data["word_masks"],
                "word_to_piece_inds": self.trans_ner_bert_dataset.data["word_to_piece_inds"],
                "word_to_piece_ends": self.trans_ner_bert_dataset.data["word_to_piece_ends"],
                "sent_lens": self.trans_ner_bert_dataset.data["sent_lens"],
                "labels": final_labels,
                "weights": valid_weights,
            }
        )
            
        self.trans_trainer.reset()
        # finetune from step2's model of last iteration (the M_tgt annotation model)
        self.trans_trainer.load_model(self.args.ckpt_dir)
        self.trans_trainer.train_ner(filter_trans_dataset, self.src_dev_dataset, self.eval_dataset, self.args.output_dir)
        return

    def kd_train(self):
        # load ckpt to make prediction on D_tgt
        self.trans_trainer.load_model(self.args.ckpt_dir)
        res = self.trans_trainer.eval_ner(self.unlabel_bert_dataset)
        p, r, f1, trans_labels = res[:4]
        trans_logits = res[-1]
        valid_weights = (trans_labels > -1).type(torch.float32)
        if self.args.filter_tgt == 'reweight':
            tea_dataset = NERDataset(
            {
                "idx": self.unlabel_bert_dataset.data["idx"],
                "words": self.unlabel_bert_dataset.data["words"], # this is vocab token id, not bert's token id
                "word_masks": self.unlabel_bert_dataset.data["word_masks"],
                "word_to_piece_inds": self.unlabel_bert_dataset.data["word_to_piece_inds"],
                "word_to_piece_ends": self.unlabel_bert_dataset.data["word_to_piece_ends"],
                "sent_lens": self.unlabel_bert_dataset.data["sent_lens"],
                "labels": trans_labels,
                "soft_logits": trans_logits,
            }
            )
            self.stu_trainer.reset()
            # load the model to produce knn representation
            self.stu_trainer.load_model(self.args.ckpt_dir)
            weights = self.load_knn(self.stu_trainer, dataset=tea_dataset, knn_file=os.path.join(self.args.output_dir, "knn.pkl"))
            valid_weights = valid_weights * weights
            
        trans_probs = torch.softmax(trans_logits, dim=-1)
        # the temperature for KD
        if self.args.T > 0:
            ptu = trans_probs ** (1 / self.args.T)
            trans_probs = ptu / ptu.sum(dim=-1, keepdim=True) 
            kd_dataset = NERDataset(
                {
                    "idx": self.unlabel_bert_dataset.data["idx"],
                    "words": self.unlabel_bert_dataset.data["words"],
                    "word_masks": self.unlabel_bert_dataset.data["word_masks"],
                    "word_to_piece_inds": self.unlabel_bert_dataset.data["word_to_piece_inds"],
                    "word_to_piece_ends": self.unlabel_bert_dataset.data["word_to_piece_ends"],
                    "sent_lens": self.unlabel_bert_dataset.data["sent_lens"],
                    "labels": trans_probs,
                    "weights": valid_weights,
                    "valid_masks": self.unlabel_bert_dataset.data["labels"] > -1,
                }
            )
        else:
            kd_dataset = NERDataset(
                {
                    "idx": self.unlabel_bert_dataset.data["idx"],
                    "words": self.unlabel_bert_dataset.data["words"],
                    "word_masks": self.unlabel_bert_dataset.data["word_masks"],
                    "word_to_piece_inds": self.unlabel_bert_dataset.data["word_to_piece_inds"],
                    "word_to_piece_ends": self.unlabel_bert_dataset.data["word_to_piece_ends"],
                    "sent_lens": self.unlabel_bert_dataset.data["sent_lens"],
                    "labels": trans_labels,
                    "weights": valid_weights,
                    "valid_masks": self.unlabel_bert_dataset.data["labels"] > -1,
                }
            )
        self.stu_trainer.reset()
        self.stu_trainer.train_ner(kd_dataset, self.src_dev_dataset, self.eval_dataset, self.args.output_dir)
        return