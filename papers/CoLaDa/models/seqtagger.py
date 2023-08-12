# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.crf import LinearCRF

import logging
logger = logging.getLogger()

class Loss(nn.Module):
    def __init__(self, nclass):
        super(Loss, self).__init__()
        self.nclass = nclass
        return

    def get_weighted_loss(self, log_probs, labels, weights=None):
        log_probs = -log_probs
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels < 0
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        if weights is not None:
            weights = weights.unsqueeze(-1)
            nll_loss = nll_loss * weights
        nll_loss.masked_fill_(padding_mask, 0.0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        return nll_loss


    def cal_loss(self, type_logits, gold, weights=None):
        # use CE loss
        if gold.dim() == 2:
            log_probs = nn.functional.log_softmax(type_logits, dim=-1)
            loss = self.get_weighted_loss(log_probs, gold, weights)
        else:
            loss = -(gold * nn.functional.log_softmax(type_logits, dim=-1)).sum(-1)
            loss = (loss * weights).sum() / (weights > 0).sum()
        return loss


class NERTagger(nn.Module):
    def __init__(self, params):
        super(NERTagger, self).__init__()
        self.dropout = nn.Dropout(params.ner_drop)
        self.num_tag = params.num_tag
        self.hidden_size = params.hidden_size
        self.cls = nn.Linear(params.hidden_size, self.num_tag)
        self.use_crf = params.use_crf
        self.word_pooling = params.word_pooling
        if self.use_crf:
            self.crf_layer = LinearCRF(self.num_tag, schema=params.schema, add_constraint=True, label2idx=params.label2idx)
        self.loss_fct = Loss(self.num_tag)
        self.init_params()
        return

    def init_params(self):
        self.cls.weight.data.normal_(mean=0.0, std=0.02)
        if self.cls.bias is not None:
            self.cls.bias.data.zero_()
        if self.use_crf:
            self.crf_layer.init_params()
        return

    def combine(self, seq_hidden, start_inds, end_inds, pooling='avg'):
        batch_size, max_seq_len, hidden_dim = seq_hidden.size()
        max_word_num = start_inds.size(1)
        if pooling == 'first':
            embeddings = torch.gather(seq_hidden, 1,
                                      start_inds.unsqueeze(-1).expand(batch_size, max_word_num, hidden_dim))
        elif pooling == 'avg':
            device = seq_hidden.device
            span_len = end_inds - start_inds + 1
            max_span_len = torch.max(span_len).item()
            subtoken_offset = torch.arange(0, max_span_len).to(device).view(1, 1, max_span_len).expand(batch_size,
                                                                                                       max_word_num,
                                                                                                       max_span_len)
            subtoken_pos = subtoken_offset + start_inds.unsqueeze(-1).expand(batch_size, max_word_num, max_span_len)
            subtoken_mask = subtoken_pos.le(end_inds.view(batch_size, max_word_num, 1))
            subtoken_pos = subtoken_pos.masked_fill(~subtoken_mask, 0).view(batch_size, max_word_num * max_span_len,
                                                                            1).expand(batch_size,
                                                                                      max_word_num * max_span_len,
                                                                                      hidden_dim)
            embeddings = torch.gather(seq_hidden, 1, subtoken_pos).view(batch_size, max_word_num, max_span_len,
                                                                        hidden_dim)
            embeddings = embeddings * subtoken_mask.unsqueeze(-1)
            embeddings = torch.div(embeddings.sum(2), span_len.unsqueeze(-1).float())
        else:
            raise ValueError('encode choice not in first / avg')
        return embeddings

    def forward(self, token_emb, word_to_piece_inds, word_to_piece_ends, sent_lens, labels=None, weights=None):
        word_emb = self.combine(token_emb, word_to_piece_inds, word_to_piece_ends, pooling=self.word_pooling)
        word_emb = self.dropout(word_emb)
        type_logits = self.cls(word_emb)
        if labels is not None:
            loss = self.loss_fct.cal_loss(type_logits, labels, weights=weights)
        else:
            loss = None
        if self.use_crf:
            crf_sp_logits = torch.zeros((type_logits.size(0), type_logits.size(1), type_logits.size(2) + 3), device=type_logits.device)
            crf_sp_logits[:, :, :type_logits.size(2)] = type_logits
            _, decodeIdx = self.crf_layer.decode(crf_sp_logits, sent_lens)
        else:
            decodeIdx = torch.argmax(type_logits, dim=-1)
        return type_logits, decodeIdx, word_emb, loss