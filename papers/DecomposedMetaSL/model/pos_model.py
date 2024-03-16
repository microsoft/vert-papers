import torch
from torch import nn
import torch.nn.functional as F
from .pos_fewshotmodel import FewShotTokenModel
from copy import deepcopy
import numpy as np
from .loss_model import MaxLoss

class SeqProtoCls(FewShotTokenModel):
    def __init__(self, span_encoder, max_loss, dot=False, normalize="none", temperature=None, use_focal=False):
        super(SeqProtoCls, self).__init__(span_encoder)
        self.dot = dot
        self.normalize = normalize
        self.temperature = temperature
        self.use_focal = use_focal
        self.loss_fct = MaxLoss(gamma=max_loss)
        self.proto = None
        print("use dot : {} use normalizatioin: {} use temperature: {}".format(self.dot, self.normalize,
                                                                               self.temperature if self.temperature else "none"))
        return

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
        if Q_mask is None:
            Q_emb = Q_emb.view(-1, Q_emb.size(-1))
        else:
            Q_emb = Q_emb[Q_mask.eq(1), :].view(-1, Q_emb.size(-1))
        dist = self.__dist__(S_emb.unsqueeze(0), Q_emb.unsqueeze(1), 2)
        return dist

    def __get_proto__(self, S_emb, S_tag, S_mask, max_tag):
        proto = []
        embedding = S_emb[S_mask.eq(1), :].view(-1, S_emb.size(-1))
        S_tag = S_tag[S_mask.eq(1)]
        for label in range(max_tag + 1):
            if S_tag.eq(label).sum().item() == 0:
                proto.append(torch.zeros(embedding.size(-1), device=embedding.device))
            else:
                proto.append(torch.mean(embedding[S_tag.eq(label), :], 0))
        proto = torch.stack(proto, dim=0)
        return proto

    def __get_proto_dist__(self, Q_emb, Q_mask):
        dist = self.__batch_dist__(self.proto, Q_emb, Q_mask)
        return dist

    def forward_step(self, query):
        query_word_emb = self.word_encoder(query['word'], query['word_mask'], word_to_piece_inds=query['word_to_piece_ind'],
                                            word_to_piece_ends=query['word_to_piece_end'])
        logits = self.__get_proto_dist__(query_word_emb, None)
        logits = logits.view(query_word_emb.size(0), query_word_emb.size(1), -1)
        gold = query['word_labels']
        tot_loss = self.loss_fct(logits, gold)
        pred = torch.argmax(logits, dim=-1)
        pred = pred.masked_fill(query['word_labels'] < 0, -1)
        return logits, pred, gold, tot_loss

    def init_proto(self, support):
        self.eval()
        with torch.no_grad():
            support_word_emb = self.word_encoder(support['word'], support['word_mask'], word_to_piece_inds=support['word_to_piece_ind'],
                                                    word_to_piece_ends=support['word_to_piece_end'])

            proto = self.__get_proto__(support_word_emb, support['word_labels'], support['word_labels'] > -1, len(support['label2idx']) - 1)
        self.proto = nn.Parameter(proto.data, requires_grad=True)
        return


    def inner_update(self, support_data, inner_steps, lr_inner):
        self.init_proto(support_data)
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
            _, _, _, loss = self.forward_step(support_data)
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
        data_keys = ['word', 'word_mask', 'word_to_piece_ind', 'word_to_piece_end', 'seq_len', 'word_labels']

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            label2tag = query['label2tag'][i]
            one_support = {
                k: support[k][current_support_num:current_support_num + sent_support_num] for k in data_keys if k in support
            }
            one_query = {
                k: query[k][current_query_num:current_query_num + sent_query_num] for k in data_keys if k in query
            }
            one_support['label2idx'] = one_query['label2idx'] = {label:idx for idx, label in label2tag.items()}
            self.zero_grad()
            self.inner_update(one_support, inner_steps, lr_inner) # inner update parameters on support data
            if mode == "train":
                qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query) # evaluate on query data
                grad = torch.autograd.grad(qy_loss, params) # meta-update
                meta_grad.append(grad)
            elif mode == "test":
                self.eval()
                with torch.no_grad():
                    qy_logits, qy_pred, qy_gold, qy_loss = self.forward_step(one_query)
            else:
                raise ValueError
                
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
        return