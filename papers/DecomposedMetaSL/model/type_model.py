import torch
from torch import nn
import torch.nn.functional as F
from .span_fewshotmodel import FewShotSpanModel
from copy import deepcopy
from .loss_model import FocalLoss
import numpy as np

class SpanProtoCls(FewShotSpanModel):
    def __init__(self, span_encoder, use_oproto, dot=False, normalize="none", 
                 temperature=None, use_focal=False):
        super(SpanProtoCls, self).__init__(span_encoder)
        self.dot = dot
        self.normalize = normalize
        self.temperature = temperature
        self.use_focal = use_focal
        self.use_oproto = use_oproto
        self.proto = None
        if use_focal:
            self.base_loss_fct = FocalLoss(gamma=1.0)
            print("use focal loss")
        else:
            self.base_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
            print("use cross entropy loss")
        print("use dot : {} use normalizatioin: {} use temperature: {}".format(self.dot, self.normalize,
                                                                               self.temperature if self.temperature else "none"))
        self.cached_o_proto = torch.zeros(span_encoder.span_dim, requires_grad=False)
        return

    def loss_fct(self, logits, targets, inst_weights=None):
        if inst_weights is None:
            loss = self.base_loss_fct(logits, targets)
            loss = loss.mean()
        else:
            targets = torch.clamp(targets, min=0)
            one_hot_targets = torch.zeros(logits.size(), device=logits.device).scatter_(1, targets.unsqueeze(1), 1)
            soft_labels = inst_weights.unsqueeze(1) * one_hot_targets + (1 - one_hot_targets) * (1 - inst_weights).unsqueeze(1) / (logits.size(1) - 1)
            logp = F.log_softmax(logits, dim=-1)
            loss = - (logp * soft_labels).sum(1)
            loss = loss.mean()
        return loss


    def __dist__(self, x, y, dim):
        if self.normalize == 'l2':
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        if self.dot:
            sim = (x * y).sum(dim) 
        else:
            sim = -(torch.pow(x - y, 2)).sum(dim)
        if self.temperature:
            sim = sim / self.temperature
        return sim

    def __batch_dist__(self, S_emb, Q_emb, Q_mask):
        Q_emb = Q_emb[Q_mask.eq(1), :].view(-1, Q_emb.size(-1))
        dist = self.__dist__(S_emb.unsqueeze(0), Q_emb.unsqueeze(1), 2)
        return dist

    def __get_proto__(self, S_emb, S_tag, S_mask):
        proto = []
        embedding = S_emb[S_mask.eq(1), :].view(-1, S_emb.size(-1))
        S_tag = S_tag[S_mask.eq(1)]
        if self.use_oproto:
            st_idx = 0
        else:
            st_idx = 1
            proto = [self.cached_o_proto]
        for label in range(st_idx, torch.max(S_tag) + 1):
            proto.append(torch.mean(embedding[S_tag.eq(label), :], 0))
        proto = torch.stack(proto, dim=0)
        return proto

    def __get_proto_dist__(self, Q_emb, Q_mask):
        dist = self.__batch_dist__(self.proto, Q_emb, Q_mask)
        if not self.use_oproto:
            dist[:, 0] = -1000000
        return dist

    def forward_step(self, query):
        if query['span_mask'].sum().item() == 0: # there is no query mentions
            print("no query mentions")
            empty_tensor = torch.tensor([], device=query['word'].device)
            zero_tensor = torch.tensor([0], device=query['word'].device)
            return empty_tensor, empty_tensor, empty_tensor, zero_tensor
        query_span_emb = self.word_encoder(query['word'], query['word_mask'], word_to_piece_inds=query['word_to_piece_ind'],
                                            word_to_piece_ends=query['word_to_piece_end'], span_indices=query['span_indices'])

        logits = self.__get_proto_dist__(query_span_emb, query['span_mask'])
        golds = query["span_tag"][query["span_mask"].eq(1)].view(-1)
        query_span_weights = query["span_weights"][query["span_mask"].eq(1)].view(-1)
        if self.use_oproto:
            loss = self.loss_fct(logits, golds, inst_weights=query_span_weights)
        else:
            loss = self.loss_fct(logits[:, 1:], golds - 1, inst_weights=query_span_weights)
        _, preds = torch.max(logits, dim=-1)
        return logits, preds, golds, loss

    def init_proto(self, support_data):
        self.eval()
        with torch.no_grad():
            support_span_emb = self.word_encoder(support_data['word'], support_data['word_mask'], word_to_piece_inds=support_data['word_to_piece_ind'],
                                                        word_to_piece_ends=support_data['word_to_piece_end'], span_indices=support_data['span_indices'])
            self.cached_o_proto = self.cached_o_proto.to(support_span_emb.device)
            proto = self.__get_proto__(support_span_emb, support_data['span_tag'], support_data['span_mask'])
        self.proto = nn.Parameter(proto.data, requires_grad=True)
        return

    def inner_update(self, support_data, inner_steps, lr_inner):
        self.init_proto(support_data)
        parameters_to_optimize = list(self.named_parameters())
        decay_params = []
        nodecay_params = []
        decay_names = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        for n, p in parameters_to_optimize:
            if p.requires_grad:
                if ("bert." in n) and (not any(nd in n for nd in no_decay)):
                    decay_params.append(p)
                    decay_names.append(n)
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
            _, _, _, loss = self.forward_step(support_data)
            if loss.requires_grad:
                loss.backward()
                inner_opt.step()
        return

    def forward_meta(self, batch, inner_steps, lr_inner, mode):
        names, params = self.get_named_params(no_grads=["proto"])
        weights = deepcopy(params)

        meta_grad = []
        episode_losses = []
        query_logits = []
        query_preds = []
        query_golds = []
        current_support_num = 0
        current_query_num = 0
        support, query = batch["support"], batch["query"]
        data_keys = ['word', 'word_mask', 'word_to_piece_ind', 'word_to_piece_end', 'span_indices', 'span_mask', 'span_tag', 'span_weights']

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            one_support = {
                k: support[k][current_support_num:current_support_num + sent_support_num] for k in data_keys if k in support
            }
            one_query = {
                k: query[k][current_query_num:current_query_num + sent_query_num] for k in data_keys if k in query
            }
            self.zero_grad()
            self.inner_update(one_support, inner_steps, lr_inner) 
            if mode == "train":
                qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query) 
                if one_query['span_mask'].sum().item() == 0:
                    pass
                else:
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

    def get_named_params(self, no_grads=[]):
        names = [n for n, p in self.named_parameters() if p.requires_grad and (n not in no_grads)]
        params = [p for n, p in self.named_parameters() if p.requires_grad and (n not in no_grads)]
        return names, params

    def load_weights(self, names, params):
        model_params = self.state_dict()
        for n, p in zip(names, params):
            assert n in model_params
            model_params[n].data.copy_(p.data)
        return

    def load_gradients(self, names, grads):
        model_params = self.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data)  # accumulate