import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import string
from transformers import AutoModel, AutoTokenizer
from .adapter_layer import AdapterStack

class BERTSpanEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length, last_n_layer=-4, word_encode_choice='first', span_encode_choice=None,
                 use_width=False, width_dim=20, use_case=False, case_dim=20, drop_p=0.1, use_att=False, att_hidden_dim=100,
                 use_adapter=False, adapter_size=64, adapter_layer_ids=None):
        nn.Module.__init__(self)
        self.use_adapter = use_adapter
        self.adapter_layer_ids = adapter_layer_ids
        self.ada_layer_num = 12
        if self.adapter_layer_ids is not None:
            self.ada_layer_num = len(self.adapter_layer_ids)
        if self.use_adapter:
            from .bert_adapter import BertModel
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.ment_adapters = AdapterStack(adapter_size, num_hidden_layers=self.ada_layer_num)
            self.type_adapters = AdapterStack(adapter_size, num_hidden_layers=self.ada_layer_num)
            print("use task specific adapter !!!!!!!!!")
        else:
            self.bert = AutoModel.from_pretrained(pretrain_path)
        for n, p in self.bert.named_parameters():
            if "pooler" in n:
                p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.last_n_layer = last_n_layer
        self.word_encode_choice = word_encode_choice
        self.span_encode_choice = span_encode_choice
        self.drop = nn.Dropout(drop_p)
        self.word_dim = self.bert.config.hidden_size

        self.use_att = use_att
        self.att_hidden_dim = att_hidden_dim
        if use_att:
            self.att_layer = nn.Sequential(
                nn.Linear(self.word_dim, self.att_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.att_hidden_dim, 1)
            )
        if span_encode_choice is None:
            span_dim = self.word_dim * 2
            print("span representation is [head; tail]")
        else:
            span_dim = self.word_dim
            print("span representation is ", self.span_encode_choice)

        self.use_width = use_width
        if self.use_width:
            self.width_dim = width_dim
            self.width_mat = nn.Embedding(50, width_dim)
            span_dim = span_dim + width_dim
            print("use width embedding")
        self.use_case = use_case
        if self.use_case:
            self.case_dim = case_dim
            self.case_mat = nn.Embedding(10, case_dim)
            span_dim = span_dim + case_dim
            print("use case embedding")
        self.span_dim = span_dim
        print("word dim is {}, span dim is {}".format(self.word_dim, self.span_dim))
        return


    def combine(self, seq_hidden, start_inds, end_inds, pooling='avg'):
        batch_size, max_seq_len, hidden_dim = seq_hidden.size()
        max_span_num = start_inds.size(1)
        if pooling == 'first':
            embeddings = torch.gather(seq_hidden, 1,
                                      start_inds.unsqueeze(-1).expand(batch_size, max_span_num, hidden_dim))
        elif pooling == 'avg':
            device = seq_hidden.device
            span_len = end_inds - start_inds + 1
            max_span_len = torch.max(span_len).item()
            subtoken_offset = torch.arange(0, max_span_len).to(device).view(1, 1, max_span_len).expand(batch_size,
                                                                                                       max_span_num,
                                                                                                       max_span_len)
            subtoken_pos = subtoken_offset + start_inds.unsqueeze(-1).expand(batch_size, max_span_num, max_span_len)
            subtoken_mask = subtoken_pos.le(end_inds.view(batch_size, max_span_num, 1))
            subtoken_pos = subtoken_pos.masked_fill(~subtoken_mask, 0).view(batch_size, max_span_num * max_span_len,
                                                                            1).expand(batch_size,
                                                                                      max_span_num * max_span_len,
                                                                                      hidden_dim)
            embeddings = torch.gather(seq_hidden, 1, subtoken_pos).view(batch_size, max_span_num, max_span_len,
                                                                        hidden_dim)
            embeddings = embeddings * subtoken_mask.unsqueeze(-1)
            embeddings = torch.div(embeddings.sum(2), span_len.unsqueeze(-1).float())
        elif pooling == 'attavg':
            global_weights = self.att_layer(seq_hidden.view(-1, hidden_dim)).view(batch_size, max_seq_len, 1)
            seq_hidden_w = torch.cat([seq_hidden, global_weights], dim=2)

            device = seq_hidden.device
            span_len = end_inds - start_inds + 1
            max_span_len = torch.max(span_len).item()
            subtoken_offset = torch.arange(0, max_span_len).to(device).view(1, 1, max_span_len).expand(batch_size,
                                                                                                       max_span_num,
                                                                                                       max_span_len)
            subtoken_pos = subtoken_offset + start_inds.unsqueeze(-1).expand(batch_size, max_span_num, max_span_len)
            subtoken_mask = subtoken_pos.le(end_inds.view(batch_size, max_span_num, 1))
            subtoken_pos = subtoken_pos.masked_fill(~subtoken_mask, 0).view(batch_size, max_span_num * max_span_len,
                                                                            1).expand(batch_size,
                                                                                      max_span_num * max_span_len,
                                                                                      hidden_dim + 1)
            span_w_embeddings = torch.gather(seq_hidden_w, 1, subtoken_pos).view(batch_size, max_span_num, max_span_len,
                                                                                 hidden_dim + 1)
            span_w_embeddings = span_w_embeddings * subtoken_mask.unsqueeze(-1)
            word_embeddings = span_w_embeddings[:, :, :, :-1]
            word_weights = span_w_embeddings[:, :, :, -1].masked_fill(~subtoken_mask, -1e8)
            word_weights = F.softmax(word_weights, dim=2).unsqueeze(3)
            embeddings = (word_weights * word_embeddings).sum(2)
        elif pooling == 'max':
            device = seq_hidden.device
            span_len = end_inds - start_inds + 1
            max_span_len = torch.max(span_len).item()
            subtoken_offset = torch.arange(0, max_span_len).to(device).view(1, 1, max_span_len).expand(batch_size,
                                                                                                       max_span_num,
                                                                                                       max_span_len)
            subtoken_pos = subtoken_offset + start_inds.unsqueeze(-1).expand(batch_size, max_span_num, max_span_len)
            subtoken_mask = subtoken_pos.le(
                end_inds.view(batch_size, max_span_num, 1))  # batch_size, max_span_num, max_span_len
            subtoken_pos = subtoken_pos.masked_fill(~subtoken_mask, 0).view(batch_size, max_span_num * max_span_len,
                                                                            1).expand(batch_size,
                                                                                      max_span_num * max_span_len,
                                                                                      hidden_dim)
            embeddings = torch.gather(seq_hidden, 1, subtoken_pos).view(batch_size, max_span_num, max_span_len,
                                                                        hidden_dim)
            embeddings = embeddings.masked_fill(
                (~subtoken_mask).unsqueeze(-1).expand(batch_size, max_span_num, max_span_len, hidden_dim), -1e8).max(
                dim=2)[0]
        else:
            raise ValueError('encode choice not in first / avg / max')
        return embeddings


    def forward(self, words, word_masks, word_to_piece_inds=None, word_to_piece_ends=None, span_indices=None, word_shape_ids=None, mode=None, bottom_hiddens=None):
        assert word_to_piece_inds.size(0) == words.size(0)
        assert word_to_piece_inds.size(1) <= words.size(1)
        assert word_to_piece_ends.size() == word_to_piece_inds.size()
        output_bottom_hiddens = None
        if (mode is not None) and self.use_adapter:
            if mode == 'ment':
                outputs = self.bert(words, attention_mask=word_masks, output_hidden_states=True, return_dict=True, adapters=self.ment_adapters, adapter_layer_ids=self.adapter_layer_ids, input_bottom_hiddens=bottom_hiddens)
            else:
                outputs = self.bert(words, attention_mask=word_masks, output_hidden_states=True, return_dict=True, adapters=self.type_adapters, adapter_layer_ids=self.adapter_layer_ids, input_bottom_hiddens=bottom_hiddens)
            if mode == 'ment':
                output_bottom_hiddens = outputs['hidden_states'][self.adapter_layer_ids[0]]
        else:
            outputs = self.bert(words, attention_mask=word_masks, output_hidden_states=True, return_dict=True)
            
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat(
            [hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][self.last_n_layer:]], 0)
        del outputs
        piece_embeddings = torch.sum(last_four_hidden_states, 0)  # [num_sent, number_of_tokens, 768]
        _, piece_len, hidden_dim = piece_embeddings.size()

        word_embeddings = self.combine(piece_embeddings, word_to_piece_inds, word_to_piece_ends,
                                       self.word_encode_choice)
        if span_indices is None:
            word_embeddings = self.drop(word_embeddings)
            if mode == 'ment':
                return word_embeddings, output_bottom_hiddens
            return word_embeddings

        if word_shape_ids is not None:
            assert word_shape_ids.size() == word_to_piece_inds.size()
            assert torch.sum(span_indices[:, :, 1].lt(span_indices[:, :, 0])).item() == 0

        embeds = []
        if self.span_encode_choice is None:
            start_word_embeddings = self.combine(word_embeddings, span_indices[:, :, 0], None, 'first')
            start_word_embeddings = self.drop(start_word_embeddings)
            embeds.append(start_word_embeddings)
            end_word_embeddings = self.combine(word_embeddings, span_indices[:, :, 1], None, 'first')
            end_word_embeddings = self.drop(end_word_embeddings)
            embeds.append(end_word_embeddings)
        else:
            pool_embeddings = self.combine(word_embeddings, span_indices[:, :, 0], span_indices[:, :, 1],
                                           self.span_encode_choice)
            pool_embeddings = self.drop(pool_embeddings)
            embeds.append(pool_embeddings)

        if self.use_width:
            width_embeddings = self.width_mat(span_indices[:, :, 1] - span_indices[:, :, 0])
            width_embeddings = self.drop(width_embeddings)
            embeds.append(width_embeddings)

        if self.use_case:
            case_wemb = self.case_mat(word_shape_ids)
            case_embeddings = self.combine(case_wemb, span_indices[:, :, 0], span_indices[:, :, 1], 'avg')
            case_embeddings = self.drop(case_embeddings)
            embeds.append(case_embeddings)

        span_embeddings = torch.cat(embeds, dim=-1)
        span_embeddings = span_embeddings.view(span_indices.size(0), span_indices.size(1), self.span_dim)
        return span_embeddings

    def get_word_case(self, token):
        if token.isdigit():
            tfeat = 0
        elif token in string.punctuation:
            tfeat = 1
        elif token.isupper():
            tfeat = 2
        elif token[0].isupper():
            tfeat = 3
        elif token.islower():
            tfeat = 4
        else:
            tfeat = 5
        return tfeat

    def tokenize_label(self, label_name):
        token_ids = self.tokenizer.encode(label_name, add_special_tokens=True, max_length=self.max_length, truncation=True)
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(token_ids)] = 1
        # padding
        while len(token_ids) < self.max_length:
            token_ids.append(0)
        return token_ids, mask

    def tokenize(self, input_tokens, true_token_flags=None):
        cur_tokens = ['[CLS]']
        cur_word_to_piece_ind = []
        cur_word_to_piece_end = []
        cur_word_shape_ind = []
        raw_tokens_list = []
        word_mask_list = []
        indexed_tokens_list = []
        word_to_piece_ind_list = []
        word_to_piece_end_list = []
        word_shape_ind_list = []
        seq_len = []
        word_flag = True
        for i in range(len(input_tokens)):
            word = input_tokens[i]
            if true_token_flags is None:
                word_flag = True
            else:
                word_flag = true_token_flags[i]
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ['[UNK]']
            if len(cur_tokens) + len(word_tokens) + 2 > self.max_length:
                raw_tokens_list.append(cur_tokens + ['[SEP]'])
                word_to_piece_ind_list.append(cur_word_to_piece_ind)
                word_to_piece_end_list.append(cur_word_to_piece_end)
                word_shape_ind_list.append(cur_word_shape_ind)
                seq_len.append(len(cur_word_to_piece_ind))
                cur_tokens = ['[CLS]'] + word_tokens
                if word_flag:
                    cur_word_to_piece_ind = [1]
                    cur_word_to_piece_end = [len(cur_tokens) - 1]
                    cur_word_shape_ind = [self.get_word_case(word)]
                else:
                    cur_word_to_piece_ind = []
                    cur_word_to_piece_end = []
                    cur_word_shape_ind = []
            else:
                if word_flag:
                    cur_word_to_piece_ind.append(len(cur_tokens))
                    cur_tokens.extend(word_tokens)
                    cur_word_to_piece_end.append(len(cur_tokens) - 1)
                    cur_word_shape_ind.append(self.get_word_case(word))
                else:
                    cur_tokens.extend(word_tokens)
        if len(cur_tokens):
            assert len(cur_tokens) < self.max_length
            raw_tokens_list.append(cur_tokens + ['[SEP]'])
            word_to_piece_ind_list.append(cur_word_to_piece_ind)
            word_to_piece_end_list.append(cur_word_to_piece_end)
            word_shape_ind_list.append(cur_word_shape_ind)
            seq_len.append(len(cur_word_to_piece_ind))
        assert seq_len == [len(x) for x in word_to_piece_ind_list]
        if true_token_flags is None:
            assert sum(seq_len) == len(input_tokens)
        assert len(raw_tokens_list) == len(word_to_piece_ind_list)
        assert len(raw_tokens_list) == len(word_to_piece_end_list)

        for raw_tokens in raw_tokens_list:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(raw_tokens)
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)
            if len(indexed_tokens) != self.max_length:
                print(input_tokens)
                print(raw_tokens)
                raise ValueError
            assert len(indexed_tokens) == self.max_length
            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(raw_tokens)] = 1
            word_mask_list.append(mask)
        sent_num = len(indexed_tokens_list)
        assert sent_num == len(word_mask_list)
        assert sent_num == len(word_to_piece_ind_list)
        assert sent_num == len(seq_len)
        return indexed_tokens_list, word_mask_list, word_to_piece_ind_list, word_to_piece_end_list, word_shape_ind_list, seq_len