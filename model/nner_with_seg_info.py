import torch
import crfb
import acnn
import util
import torch.nn as nn
from self_attention_layer import SelfAttentionComponent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CANNERModel(nn.Module):
    def __init__(self, config, tag2id, dropout, seg_dim = 300, embed_dim = 300, pretrain_embed=None):
        super().__init__()

        # make sure this tag2id do not contains start and stop
        self.tag_to_ix = tag2id
        self.target_size = len(tag2id) + 2

        self.reset_para = False
        self.embed_dim = embed_dim
        self.seg_dim = seg_dim

        self.model_name = config.model_name
        self.hidden_dim = config.hidden_dim
        self.window_size = config.window_size


        self.dropout = torch.nn.Dropout(p=dropout)

        self.features_embeds = []
        embeds = torch.nn.Embedding.from_pretrained(torch.from_numpy(pretrain_embed), freeze=False)
        soft_seg = torch.nn.Embedding(5, self.seg_dim)  # OBMES 01234
        if self.reset_para:
            util.init_embedding_(soft_seg)
        self.add_module('feature_embeds_{}'.format(0), embeds)
        self.add_module('feature_embeds_{}'.format(1), soft_seg)
        self.features_embeds.append(embeds)
        self.features_embeds.append(soft_seg)


        self.features_embeds = [attr.to(device) for attr in self.features_embeds]
        self.cnn = acnn.ACNN(self.embed_dim + self.seg_dim, self.hidden_dim, self.window_size)

        self.rnn = torch.nn.GRU(self.hidden_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
                                batch_first=True)

        self.conv_activation = torch.nn.LeakyReLU(0.01)
        self.attc = SelfAttentionComponent(self.hidden_dim, self.hidden_dim)
        self.hidden2tag = torch.nn.Linear(self.hidden_dim * 2, self.target_size)

        self.crf = crfb.CRF(self.target_size - 2, self.model_name)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.target_size)

        if self.reset_para:
            util.init_lstm_(self.rnn)
            util.init_linear_(self.hidden2tag)


    def forward(self, batch_x, mask):
        batch_n, seq_len, feature_num = batch_x.size()
        for i in range(feature_num):
            feature_embed_data = batch_x[:, :, i]
            feature_embeds = getattr(self, 'feature_embeds_{}'.format(i))
            if i == 0:
                embeds_output = feature_embeds(feature_embed_data)
            else:
                embeds_output = torch.cat((embeds_output, feature_embeds(feature_embed_data)), 2)

        # batch, seq_len, hidden_dim
        cnn_out = self.cnn(embeds_output)
        cnn_out = self.conv_activation(cnn_out)
        cnn_out = self.dropout(cnn_out)

        att_contexts = self.attc(cnn_out)
        cat_out = torch.cat((cnn_out, att_contexts), 2)
        cat_out = self.dropout(cat_out)

        out = self.hidden2tag(cat_out)

        return out