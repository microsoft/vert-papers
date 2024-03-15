import torch
from torch import nn
import torch.nn.functional as F
from .span_fewshotmodel import FewShotSpanModel
from copy import deepcopy
import numpy as np
from .crf import LinearCRF
from .loss_model import MaxLoss

class MentSeqtagger(FewShotSpanModel):
    def __init__(self, span_encoder, num_tag, label2idx, schema, use_crf, max_loss):
        super(MentSeqtagger, self).__init__(span_encoder)
        self.num_tag = num_tag
        self.use_crf = use_crf
        self.label2idx = label2idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.schema = schema
        self.cls = nn.Linear(span_encoder.word_dim, self.num_tag)
        if self.use_crf:
            self.crf_layer = LinearCRF(self.num_tag, schema=schema, add_constraint=True, label2idx=label2idx)
        self.ment_loss_fct = MaxLoss(gamma=max_loss)
        self.init_weights()
        return

    def init_weights(self):
        self.cls.weight.data.normal_(mean=0.0, std=0.02)
        if self.cls.bias is not None:
            self.cls.bias.data.zero_()
        if self.use_crf:
            self.crf_layer.init_params()
        return

    def forward_step(self, batch, crf_mode=True):
        word_emb = self.word_encoder(batch['word'], batch['word_mask'],
                                        batch['word_to_piece_ind'],
                                        batch['word_to_piece_end'])
        logits = self.cls(word_emb)
        gold = batch['ment_labels']
        tot_loss = self.ment_loss_fct(logits, gold)
        if self.use_crf and crf_mode:
            crf_sp_logits = torch.zeros((logits.size(0), logits.size(1), 3), device=logits.device)
            crf_sp_logits = torch.cat([logits, crf_sp_logits], dim=2)
            _, pred = self.crf_layer.decode(crf_sp_logits, batch['seq_len']) 
        else:
            pred = torch.argmax(logits, dim=-1)
        pred = pred.masked_fill(gold.eq(-1), -1)
        return logits, pred, gold, tot_loss


    def inner_update(self, train_data, inner_steps, lr_inner):
        parameters_to_optimize = list(self.named_parameters())
        decay_params = []
        nodecay_params = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        for n, p in parameters_to_optimize:
            if p.requires_grad:
                if ("bert." in n) and (not any(nd in n for nd in no_decay)):
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)
        parameters_groups = [
            {'params': decay_params,
             'lr': lr_inner, 'weight_decay': 1e-3},
            {'params': nodecay_params,
             'lr': lr_inner, 'weight_decay': 0},
        ]
        inner_opt = torch.optim.AdamW(parameters_groups, lr=lr_inner)
        self.train()
        for _ in range(inner_steps):
            inner_opt.zero_grad()
            _, _, _, loss = self.forward_step(train_data, crf_mode=False)
            loss.backward()
            inner_opt.step()
        return

    def forward_sup(self, batch, mode):
        support, query = batch["support"], batch["query"]
        query_logits = []
        query_preds = []
        query_golds = []
        all_loss = 0
        task_num = 0
        current_support_num = 0
        current_query_num = 0
        if mode == "train":
            crf_mode = False
        else:
            assert mode == "test"
            crf_mode = True
            self.eval()
            print('eval mode')
        data_keys = ['word', 'word_mask', 'word_to_piece_ind', 'word_to_piece_end', 'seq_len', 'ment_labels']
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            one_support = {
                k: support[k][current_support_num:current_support_num + sent_support_num] for k in data_keys
            }
            one_query = {
                k: query[k][current_query_num:current_query_num + sent_query_num] for k in data_keys
            }
            _, _, _, sp_loss = self.forward_step(one_support, False)
            all_loss += sp_loss
            qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query, crf_mode)
            query_preds.append(qy_pred)
            query_golds.append(qy_gold)
            query_logits.append(qy_logits)
            all_loss += qy_loss
            task_num += 2
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        return {'loss': all_loss / task_num, 'preds': query_preds, 'golds': query_golds, 'logits': query_logits}

    def forward_meta(self, batch, inner_steps, lr_inner, mode):
        names, params = self.get_named_params()
        weights = deepcopy(params)
        meta_grad = []
        episode_losses = []
        query_preds = []
        query_golds = []
        query_logits = []
        support, query = batch["support"], batch["query"]
        current_support_num = 0
        current_query_num = 0
        data_keys = ['word', 'word_mask', 'word_to_piece_ind', 'word_to_piece_end', 'seq_len', 'ment_labels']
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            one_support = {
                k: support[k][current_support_num:current_support_num + sent_support_num] for k in data_keys
            }
            one_query = {
                k: query[k][current_query_num:current_query_num + sent_query_num] for k in data_keys
            }
            self.zero_grad()
            self.inner_update(one_support, inner_steps, lr_inner)
            if mode == "train":
                qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query, crf_mode=False)
                grad = torch.autograd.grad(qy_loss, params)
                meta_grad.append(grad)
            else:
                self.eval()
                with torch.no_grad():
                    qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query)
            episode_losses.append(qy_loss.item())
            query_preds.append(qy_pred)
            query_golds.append(qy_gold)
            query_logits.append(qy_logits)
            self.load_weights(names, weights)

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        self.zero_grad()
        return {'loss': np.mean(episode_losses), 'names': names, 'grads': meta_grad, 'preds': query_preds, 'golds': query_golds, 'logits': query_logits}

    def get_named_params(self):
        names = [n for n, p in self.named_parameters() if p.requires_grad]
        params = [p for n, p in self.named_parameters() if p.requires_grad]
        return names, params

    def load_weights(self, names, params):
        model_params = self.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)
        return

    def load_gradients(self, names, grads):
        model_params = self.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data)  # accumulate
        return