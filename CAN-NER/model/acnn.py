import torch
import torch.nn as nn

import util
from attention_layer import AttentionComponent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ACNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, window_size=5, use_cnn=True, use_bias=True,
                 cat_self=False, pos_info=False, att_type='bahdanau', reset_para=False):
        super(ACNN, self).__init__()

        self.window_size = window_size
        assert window_size % 2 == 1
        self.padding = self.window_size // 2
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_cnn = use_cnn
        self.cat_self = cat_self
        self.pos_info = pos_info

        if pos_info:
            new_feature_dim = self.feature_dim + self.window_size
            self.attc = AttentionComponent(self.hidden_dim, new_feature_dim, att_type, use_bias, reset_para)
            if use_cnn:
                self.cnn = nn.Linear(new_feature_dim * self.window_size, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.cnn)
            else:
                self.linear = nn.Linear(new_feature_dim, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.linear)

            if self.cat_self:
                if self.use_cnn:
                    int_features = new_feature_dim + self.hidden_dim
                else:
                    int_features = new_feature_dim * 2
                self.linear = nn.Linear(int_features, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.linear)
        else:
            self.attc = AttentionComponent(self.hidden_dim, self.feature_dim)
            if use_cnn:
                self.cnn = nn.Linear(self.feature_dim * self.window_size, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.cnn)
            else:
                self.linear = nn.Linear(self.feature_dim, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.cnn)
            if self.cat_self:
                if self.use_cnn:
                    int_features = self.feature_dim + self.hidden_dim
                else:
                    int_features = self.feature_dim * 2
                self.linear = nn.Linear(int_features, self.hidden_dim, bias=use_bias)
                if reset_para:
                    util.init_linear_(self.cnn)

    def forward(self, embeds_output):
        batch_n, seq_len, feature_dim = embeds_output.size()

        conv_embeds = self.getConvEmbeds(embeds_output[:]).transpose(3, 2)

        # batch, seq_len, features, windows
        context_cnn_out = []
        for i in range(seq_len):
            contexts = conv_embeds[:, i, :, :]
            entity = contexts[:, self.padding, :].unsqueeze(1)
            if self.use_cnn:
                att_weigths = self.attc.get_att_weights(entity, contexts)
                att_context = torch.mul(att_weigths, contexts).reshape(batch_n, -1)
                # att_context = torch.mul(att_weigths, contexts).view(batch_n, -1)

                # .view(batch_n, self.feature_dim * self.window_size)
                cnn_out = self.cnn(att_context)
                context_cnn_out.append(cnn_out)
            else:
                att_context, _ = self.attc.forward(entity, contexts)
                context_cnn_out.append(att_context)
        context_cnn_out = torch.stack(context_cnn_out, 1)  # batch_n, seq_len,

        if self.cat_self:
            entities = embeds_output.contiguous().view(batch_n, seq_len, feature_dim)
            context_cnn_out = torch.cat((context_cnn_out, entities), 2)

        if self.use_cnn and not self.cat_self:
            att_out = context_cnn_out
        else:
            att_out = self.linear(context_cnn_out)  # batch * seq_len, hidden_dim

        att_out = att_out.view(batch_n, seq_len, self.hidden_dim)
        return att_out

    def getConvEmbeds(self, embeds_output):
        batch_n, seq_len, feature_dim = embeds_output.size()
        for i in range(self.padding):
            pad = self.padding_vector(self.feature_dim)
            pad = pad.view(1, 1, -1)
            pad = pad.expand(batch_n, 1, self.feature_dim)
            embeds_output = torch.cat([pad, embeds_output, pad], 1)

        conv_out = []
        for i in range(self.window_size):
            ed = i + seq_len
            out = embeds_output[:, i: ed, :]
            if self.pos_info:
                pos = torch.zeros(seq_len, self.window_size)
                pos[:, i] = 1
                # for i in range(self.window_size):
                #     pos[i][i] = 1
                pos = pos.expand(batch_n, seq_len, self.window_size)
                pos.to(device)
                out = torch.cat((out, pos), 2)

            conv_out.append(out)
        conv_out = torch.stack(conv_out, 3)
        return conv_out

    def padding_vector(self, dim):
        v = torch.zeros(dim)
        v.to(device)
        return v

