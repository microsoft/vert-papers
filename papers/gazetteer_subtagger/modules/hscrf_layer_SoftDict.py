from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import allennlp.nn.util as util
import numpy as np
import torch.nn.functional as F



class HSCRF(nn.Module):
    def __init__(self, ix_to_tag, word_rep_dim=300, SCRF_feature_dim=100, index_embeds_dim=10, ALLOWED_SPANLEN=7, 
                 softdict_text_field_embedder=None,
                 length_embedder=None,
                 encoder=None,
                 BILOU_tag_projection_layer=None):
        super(HSCRF, self).__init__()
        self.ix_to_tag = ix_to_tag
        self.entity_tag_ids = [ky for ky,val in ix_to_tag.items() if val != "O"]
        self.tag_to_ix = {v:k for k,v in self.ix_to_tag.items()}
        self.tagset_size = len(ix_to_tag) + 2 # including <start, end>
        self.index_embeds_dim = index_embeds_dim
        self.SCRF_feature_dim = SCRF_feature_dim
        self.ALLOWED_SPANLEN = ALLOWED_SPANLEN
        
        self.softdict_text_field_embedder = softdict_text_field_embedder
        self.length_embedder = length_embedder
        self.encoder = encoder
        self.BILOU_tag_projection_layer = BILOU_tag_projection_layer
        
        self.start_id = self.tagset_size - 1
        self.stop_id = self.tagset_size - 2
        
        self.tanher = nn.Tanh()
        
        self.ix_to_tag[self.start_id] = 'START'
        self.ix_to_tag[self.stop_id] = 'STOP'
        

        self.index_embeds = nn.Embedding(self.ALLOWED_SPANLEN, self.index_embeds_dim)
        self.init_embedding(self.index_embeds.weight)

        self.dense = nn.Linear(word_rep_dim, self.SCRF_feature_dim)
        self.init_linear(self.dense)

        # 4 for SBIE, 3 for START, STOP, O and 2 for START and O
        self.CRF_tagset_size = 4*(self.tagset_size-3)+2

        self.transition = nn.Parameter(
            torch.zeros(self.tagset_size, self.tagset_size))

        span_word_embedding_dim = 2*self.SCRF_feature_dim + self.index_embeds_dim + 4*4
        self.new_hidden2CRFtag = nn.Linear(span_word_embedding_dim, self.CRF_tagset_size)
        self.init_linear(self.new_hidden2CRFtag)


    def init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def get_logloss_denominator(self, scores, mask):
        """
        calculate all path scores of SCRF with dynamic programming
        args:
            scores (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask   (batch_size) : mask for words
        """

        logalpha = Variable(torch.FloatTensor(self.batch_size, self.sent_len+1, self.tagset_size).fill_(-10000.)).cuda(util.get_device_of(mask))
        logalpha[:, 0, self.start_id] = 0.
        istarts = [0] * self.ALLOWED_SPANLEN + list(range(self.sent_len - self.ALLOWED_SPANLEN+1))
        for i in range(1, self.sent_len+1):
            tmp = scores[:, istarts[i]:i, i-1] + \
                    logalpha[:, istarts[i]:i].unsqueeze(3).expand(self.batch_size, i - istarts[i], self.tagset_size, self.tagset_size)
            tmp = tmp.transpose(1, 3).contiguous().view(self.batch_size, self.tagset_size, (i-istarts[i])*self.tagset_size)
            max_tmp, _ = torch.max(tmp, dim=2)
            tmp = tmp - max_tmp.view(self.batch_size, self.tagset_size, 1)
            logalpha[:, i] = max_tmp + torch.log(torch.sum(torch.exp(tmp), dim=2))

        mask = mask.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.tagset_size)
        alpha = torch.gather(logalpha, 1, mask).squeeze(1)
        return alpha[:,self.stop_id].sum()

    def decode(self, factexprscalars, mask):
        """
        decode SCRF labels with dynamic programming
        args:
            factexprscalars (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask            (batch_size) : mask for words
        """

        batch_size = factexprscalars.size(0)
        sentlen = factexprscalars.size(1)
        factexprscalars = factexprscalars.data
        logalpha = torch.FloatTensor(batch_size, sentlen+1, self.tagset_size).fill_(-10000.).cuda(util.get_device_of(mask))
        logalpha[:, 0, self.start_id] = 0.
        starts = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda(util.get_device_of(mask))
        ys = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda(util.get_device_of(mask))

        for j in range(1, sentlen + 1):
            istart = 0
            if j > self.ALLOWED_SPANLEN:
                istart = max(0, j - self.ALLOWED_SPANLEN)
            f = factexprscalars[:, istart:j, j - 1].permute(0, 3, 1, 2).contiguous().view(batch_size, self.tagset_size, -1) + \
                logalpha[:, istart:j].contiguous().view(batch_size, 1, -1).expand(batch_size, self.tagset_size, (j - istart) * self.tagset_size)
            logalpha[:, j, :], argm = torch.max(f, dim=2)
            starts[:, j-1, :] = (argm / self.tagset_size + istart)
            ys[:, j-1, :] = (argm % self.tagset_size)

        batch_scores = []
        batch_spans = []
        for i in range(batch_size):
            spans = {}
            batch_scores.append(max(logalpha[i, mask[i]-1]))
            end = mask[i]-1
            y = self.stop_id
            while end >= 0:
                start = int(starts[i, end, y])
                y_1 = int(ys[i, end, y])
                if self.ix_to_tag[int(y)] not in ('START', 'STOP'):
                    spans[(int(start),int(end))] = self.ix_to_tag[int(y)]
                y = y_1
                end = start - 1
            batch_spans.append(spans)
            pass
        return batch_spans, batch_scores

    def get_logloss_numerator(self, goldfactors, scores, mask):
        """
        get scores of best path
        args:
            goldfactors (batch_size, tag_len, 4) : path labels
            scores      (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : all tag scores
            mask        (batch_size, tag_len) : mask for goldfactors
        """
        batch_size = scores.size(0)
        sent_len = scores.size(1)
        tagset_size = scores.size(3)
        goldfactors = goldfactors[:, :, 0]*sent_len*tagset_size*tagset_size + goldfactors[:,:,1]*tagset_size*tagset_size+goldfactors[:,:,2]*tagset_size+goldfactors[:,:,3]
        factorexprs = scores.view(batch_size, -1)
        val = torch.gather(factorexprs, 1, goldfactors)
        numerator = val.masked_select(mask.byte())
        return numerator


    def HSCRF_scores(self, global_feats, token_indices):
        """
        calculate SCRF scores with HSCRF
        args:
            global_feats (batch_size, sentence_len, featsdim) : word representations
        """

        # 3 for O, STOP, START
        validtag_size = self.tagset_size-3
        scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size)).cuda(util.get_device_of(global_feats))
        diag0 = torch.LongTensor(range(self.sent_len)).cuda(util.get_device_of(global_feats))
        # m10000 for STOP
        m10000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 1)).cuda(util.get_device_of(global_feats))
        # m30000 for O, START, STOP
        m30000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 3)).cuda(util.get_device_of(global_feats))
        for span_len in range(min(self.ALLOWED_SPANLEN, self.sent_len-1)):
            emb_x = self.concat_features(global_feats, token_indices, span_len)
            emb_x = self.new_hidden2CRFtag(emb_x)
            if span_len == 0:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0) + emb_x[:, 0, :, :validtag_size].unsqueeze(2),
                                 self.transition[:, -2:].unsqueeze(0).unsqueeze(0) + emb_x[:, 0, :, -2:].unsqueeze(2),
                                 m10000), 3)
                scores[:, diag0, diag0] = tmp
            elif span_len == 1:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-1, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1, :, 3*validtag_size:4*validtag_size]).unsqueeze(2), m30000[:, 1:]), 3)
                scores[:, diag0[:-1], diag0[1:]] = tmp

            elif span_len == 2:
                tmp = torch.cat((self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-2, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1, :, 2*validtag_size:3*validtag_size] +
                                                            emb_x[:, 2, :, 3*validtag_size:4*validtag_size]).unsqueeze(2), m30000[:, 2:]), 3)
                scores[:, diag0[:-2], diag0[2:]] = tmp

            elif span_len >= 3:
                tmp0 = self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-span_len, self.tagset_size, validtag_size) + \
                                                           (emb_x[:, 0, :, validtag_size:2*validtag_size] +
                                                            emb_x[:, 1:span_len, :, 2*validtag_size:3*validtag_size].sum(1) +
                                                            emb_x[:, span_len,:, 3*validtag_size:4*validtag_size]).unsqueeze(2)
                tmp = torch.cat((tmp0, m30000[:, span_len:]), 3)
                scores[:, diag0[:-span_len], diag0[span_len:]] = tmp
        return scores

    def concat_features(self, emb_z, token_indices, span_len):
        """
        concatenate two features
        args:
            emb_z (batch_size, sentence_len, featsdim) : contextualized word representations
            token_indices: Dict[str, LongTensor], indices of different fields
            span_len: a number (from 0)
        """
        batch_size = emb_z.size(0)
        sent_len = emb_z.size(1)
        hidden_dim = emb_z.size(2)
        emb_z = emb_z.unsqueeze(1).expand(batch_size, 1, sent_len, hidden_dim)
        
        span_exprs = [emb_z[:, :, i:i + span_len + 1] for i in range(sent_len - span_len)]
        span_exprs = torch.cat(span_exprs, 1)
        
        endpoint_vec = (span_exprs[:, :, 0]-span_exprs[:, :, span_len]).unsqueeze(2).expand(batch_size, sent_len-span_len, span_len+1, hidden_dim)
        
        index = Variable(torch.LongTensor(range(span_len+1))).cuda(util.get_device_of(emb_z))
        index = self.index_embeds(index).unsqueeze(0).unsqueeze(0).expand(batch_size, sent_len-span_len, span_len+1, self.index_embeds_dim)
        
        
        BILOU_features = self.get_BILOU_features(token_indices, sent_len, span_len)
        
        new_emb = torch.cat((span_exprs, BILOU_features, endpoint_vec, index), 3)
        
        return new_emb.transpose(1,2).contiguous()    
    
    
    def get_BILOU_features(self, token_indices, sent_len, span_len):
        
        span_level_token_indices = {}        
        for ky,val in list(token_indices.items()):
            if ky == 'elmo':
                continue
            val = val.unsqueeze(1)
            span_level_token_indices[ky] = torch.cat([val[:, :, i:i + span_len + 1] for i in range(sent_len - 1 - span_len)], 1)

        spans_embedded = self.softdict_text_field_embedder(span_level_token_indices, num_wrapping_dims=1)
        spans_mask = util.get_text_field_mask(span_level_token_indices, num_wrapping_dims=1)
        
        dim_2_pad = self.ALLOWED_SPANLEN - spans_embedded.size(2)
        p2d = (0,0,0, dim_2_pad)
        # now shape (batch_size, num_span, max_span_width, dim)
        spans_embedded = F.pad(spans_embedded, p2d, "constant", 0.)
        spans_mask = F.pad(spans_mask, (0, dim_2_pad), "constant", 0.)
        
        batch_size = spans_mask.size(0)
        num_spans = spans_mask.size(1)
        length_vec = torch.autograd.Variable(torch.LongTensor(range(self.ALLOWED_SPANLEN))).cuda(util.get_device_of(spans_mask))
        length_vec = self.length_embedder(length_vec).unsqueeze(0).unsqueeze(0).expand(batch_size, num_spans, -1,-1)
        
        spans_encoded = self.encoder(spans_embedded, spans_mask)
        
        spans_encoded = torch.cat((spans_encoded, length_vec), 3).contiguous()
        
        # shape (batch_size, num_spans, max_span_wid, 4* span_tags) BILU
        span_logits = self.BILOU_tag_projection_layer(spans_encoded)
        span_logits = self.tanher(span_logits)
        span_logits = torch.cat([span_logits, span_logits.new_zeros(batch_size, 1, span_logits.size(2), span_logits.size(3))], dim=1)
        
        return span_logits[:,:,:span_len+1,:].detach()
        

    def forward(self, feats, token_indices, mask_word, tags, mask_tag):
        """
        calculate loss
        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask_word (batch_size) : sentence lengths
            tags (batch_size, tag_len, 4) : target
            mask_tag (batch_size, tag_len) : tag_len <= sentence_len
        """
        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        
        self.SCRF_scores = self.HSCRF_scores(feats, token_indices)
        forward_score = self.get_logloss_denominator(self.SCRF_scores, mask_word)
        numerator = self.get_logloss_numerator(tags, self.SCRF_scores, mask_tag)
        
        return (forward_score - numerator.sum()) / self.batch_size

    def get_scrf_decode(self, mask):
        """
        decode with SCRF
        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask  (batch_size) : mask for words
        """
        batch_spans, batch_scores = self.decode(self.SCRF_scores, mask)
        return batch_spans

    
