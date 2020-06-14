import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class AttentionComponent(nn.Module):
    def __init__(self, dec_units, input_dim, score_type='bahdanau', use_bias=True, reset_para=False):
        super(AttentionComponent, self).__init__()
        self.score_type = score_type
        self.dec_units = dec_units
        self.input_dim = input_dim

        if self.score_type == 'bahdanau':
            self.W1 = nn.Linear(self.input_dim, self.dec_units, bias=use_bias)
            self.W2 = nn.Linear(self.input_dim, self.dec_units, bias=use_bias)
            self.V = nn.Linear(self.dec_units, 1, bias=use_bias)
            if reset_para:
                util.init_linear_(self.W1)
                util.init_linear_(self.W2)
                util.init_linear_(self.V)

        elif self.score_type == 'luong':
            self.W = nn.Linear(self.input_dim, self.input_dim)
            if reset_para:
                util.init_linear_(self.W)
            # self.W = nn.Linear(self.input_dim,self.input_dim)
        # elif self.score_type == 'dot':

    '''
        entities => (n_batch, 1 or n_tokens, n_features)
        contexts => (n_batch, n_tokens, n_features)
    '''

    def forward(self, entities, contexts):
        batch_n, token_n, feature_n = entities.size()
        att_weights = self.get_att_weights(entities, contexts)
        att_vectors = self.get_att_vectors(att_weights, contexts)
        return att_vectors, att_weights

    def get_att_vectors(self, att_weights, contexts):
        rets = torch.mul(contexts, att_weights)
        att_vectors = torch.sum(rets, 1)
        return att_vectors

    def get_att_weights(self, entities, contexts):
        batch_n, token_n, feature_n = entities.size()
        batch_n_c, token_n_c, feature_n_c = contexts.size()
        assert (batch_n == batch_n_c and feature_n == feature_n_c)
        assert (token_n == 1 or token_n == token_n_c)
        assert (feature_n == self.input_dim)
        self_att = token_n == token_n_c

        scores = self.score_func(entities, contexts)  # (bn * tn) * dec_units
        att_weights = F.softmax(scores, 1)
        return att_weights

    def score_func(self, ht, hs):
        if self.score_type == 'bahdanau':
            batch_n_c, token_n_c, feature_n_c = hs.size()
            rc = self.W1(hs)
            re = self.W2(ht)
            if ht.size(1) == 1 or ht.size(1) == hs.size(1):
                score = torch.tanh(rc + re)
            else:
                raise Exception("score func else")
                # batch_n = hs.size(0)
                # scores = []
                # for i in range(batch_n):
                #     score = torch.tanh(rc[i] + re[i])
                #     scores.append(score)
                # score = torch.cat(scores)
            score = self.V(score).view(batch_n_c, token_n_c, 1)
            # score = self.V(score)
        elif self.score_type == 'luong':
            # Todo:fix
            score = self.W(hs)
            score = torch.matmul(score, ht.transpose(1, 2))
            # score = torch.mul(ht.view(1, -1), self.W1(hs))
        elif self.score_type == 'dot':
            score = torch.matmul(hs, ht.transpose(1, 2))
        else:
            raise Exception('not support score function type')

        return score
