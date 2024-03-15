import torch
from torch import nn
import torch.nn.functional as F
from .crf import LinearCRF
from .span_fewshotmodel import FewShotSpanModel
from util.span_sample import convert_bio2spans
from .loss_model import MaxLoss
from copy import deepcopy
import numpy as np

class SelectedJointModel(FewShotSpanModel):
    def __init__(self, span_encoder, num_tag, ment_label2idx, schema, use_crf, max_loss, use_oproto, dot=False, normalize="none", 
                 temperature=None, use_focal=False, type_lam=1):
        super(SelectedJointModel, self).__init__(span_encoder)
        self.dot = dot
        self.normalize = normalize
        self.temperature = temperature
        self.use_oproto = use_oproto
        self.proto = None
        self.num_tag = num_tag
        self.use_crf = use_crf
        self.ment_label2idx = ment_label2idx
        self.ment_idx2label = {idx: label for label, idx in self.ment_label2idx.items()}
        self.schema = schema
        self.cls = nn.Linear(span_encoder.word_dim, self.num_tag)
        self.type_lam = type_lam
        self.proto = None
        if self.use_crf:
            self.crf_layer = LinearCRF(self.num_tag, schema=schema, add_constraint=True, label2idx=ment_label2idx)
        if use_focal:
            raise ValueError("not support focal loss")
        self.ment_loss_fct = MaxLoss(gamma=max_loss)
        self.base_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        print("use cross entropy loss")
        print("use dot : {} use normalizatioin: {} use temperature: {}".format(self.dot, self.normalize,
                                                                               self.temperature if self.temperature else "none"))
        self.cached_o_proto = torch.zeros(span_encoder.span_dim, requires_grad=False)
        self.init_weights()
        return

    def init_weights(self):
        self.cls.weight.data.normal_(mean=0.0, std=0.02)
        if self.cls.bias is not None:
            self.cls.bias.data.zero_()
        if self.use_crf:
            self.crf_layer.init_params()
        return

    def type_loss_fct(self, logits, targets, inst_weights=None):
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

    def forward_type_step(self, query, encoder_mode=None, query_bottom_hiddens=None): 
        if query['span_mask'].sum().item() == 0: # there is no query mentions
            print("no query mentions")
            empty_tensor = torch.tensor([], device=query['word'].device)
            zero_tensor = torch.tensor([0], device=query['word'].device)
            return empty_tensor, empty_tensor, empty_tensor, zero_tensor
        query_span_emb = self.word_encoder(query['word'], query['word_mask'], word_to_piece_inds=query['word_to_piece_ind'],
                                            word_to_piece_ends=query['word_to_piece_end'], span_indices=query['span_indices'], 
                                            mode=encoder_mode, bottom_hiddens=query_bottom_hiddens)
        logits = self.__get_proto_dist__(query_span_emb, query['span_mask'])
        golds = query["span_tag"][query["span_mask"].eq(1)].view(-1)
        if query["span_weights"] is not None:
            query_span_weights = query["span_weights"][query["span_mask"].eq(1)].view(-1)
        else:
            query_span_weights = None
        if self.use_oproto:
            loss = self.type_loss_fct(logits, golds, inst_weights=query_span_weights)
        else:
            loss = self.type_loss_fct(logits[:, 1:], golds - 1, inst_weights=query_span_weights)
        _, preds = torch.max(logits, dim=-1)
        return logits, preds, golds, loss

    def init_proto(self, support_data, encoder_mode=None, support_bottom_hiddens=None):
        self.eval()
        with torch.no_grad():
            support_span_emb = self.word_encoder(support_data['word'], support_data['word_mask'], word_to_piece_inds=support_data['word_to_piece_ind'],
                                                        word_to_piece_ends=support_data['word_to_piece_end'], span_indices=support_data['span_indices'],
                                                        mode=encoder_mode, bottom_hiddens=support_bottom_hiddens)
            self.cached_o_proto = self.cached_o_proto.to(support_span_emb.device)
            proto = self.__get_proto__(support_span_emb, support_data['span_tag'], support_data['span_mask'])
        self.proto = nn.Parameter(proto.data, requires_grad=True)
        return

    def forward_ment_step(self, batch, crf_mode=True, encoder_mode=None):
        res = self.word_encoder(batch['word'], batch['word_mask'],
                                        batch['word_to_piece_ind'],
                                        batch['word_to_piece_end'], mode=encoder_mode,
                                        )
        word_emb = res[0]
        bottom_hiddens = res[1]

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
        return logits, pred, gold, tot_loss, bottom_hiddens
    
    def joint_inner_update(self, support_data, inner_steps, lr_inner):
        self.init_proto(support_data, encoder_mode="type")
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
            _, _, _, ment_loss, support_bottom_hiddens = self.forward_ment_step(support_data, crf_mode=False, encoder_mode="ment")
            _, _, _, type_loss = self.forward_type_step(support_data, encoder_mode="type", query_bottom_hiddens=support_bottom_hiddens)
            
            loss = ment_loss + type_loss * self.type_lam
            loss.backward()
            inner_opt.step()
        return

    def decode_ments(self, episode_preds):
        episode_preds = episode_preds.detach()
        span_indices = torch.zeros((episode_preds.size(0), 100, 2), dtype=torch.long)
        span_masks = torch.zeros(span_indices.size()[:2], dtype=torch.long)
        span_labels = torch.full_like(span_masks, fill_value=1, dtype=torch.long)
        max_span_num = 0
        for i, pred in enumerate(episode_preds):
            seqs = []
            for idx in pred:
                if idx.item() == -1:
                    break
                seqs.append(self.ment_idx2label[idx.item()])
            ents = convert_bio2spans(seqs, self.schema)
            max_span_num = max(max_span_num, len(ents))
            for j, x in enumerate(ents):
                span_indices[i, j, 0] = x[1]
                span_indices[i, j, 1] = x[2]
            span_masks[i, :len(ents)] = 1
        return span_indices, span_masks, span_labels, max_span_num


    # forward proto maml
    def forward_joint_meta(self, batch, inner_steps, lr_inner, mode):
        no_grads = ["proto"]
        if batch['query']['span_mask'].sum().item() == 0: # there is no query mentions
            print("no query mentions")
            no_grads.append("type_adapters")   
        names, params = self.get_named_params(no_grads=no_grads)
        weights = deepcopy(params)
        meta_grad = []
        episode_losses = []
        episode_ment_losses = []
        episode_type_losses = []
        query_ment_logits = []
        query_ment_preds = []
        query_ment_golds = []
        query_type_logits = []
        query_type_preds = []
        query_type_golds = []
        two_stage_query_ments = []
        two_stage_query_masks = []
        two_stage_max_snum = 0
        current_support_num = 0
        current_query_num = 0
        support, query = batch["support"], batch["query"]
        data_keys = ['word', 'word_mask', 'word_to_piece_ind', 'word_to_piece_end', 'seq_len', 'ment_labels', 'span_indices', 'span_mask', 'span_tag', 'span_weights']

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            one_support = {
                k: support[k][current_support_num:current_support_num + sent_support_num] for k in data_keys if k in support
            }
            one_query = {
                k: query[k][current_query_num:current_query_num + sent_query_num] for k in data_keys if k in query
            }
            self.zero_grad()
            self.joint_inner_update(one_support, inner_steps, lr_inner) # inner update parameters on support data
            if mode == "train":
                qy_ment_logits, qy_ment_pred, qy_ment_gold, qy_ment_loss, qy_bottom_hiddens = self.forward_ment_step(one_query, crf_mode=False, encoder_mode="ment") # evaluate on query data
                qy_type_logits, qy_type_pred, qy_type_gold, qy_type_loss = self.forward_type_step(one_query, encoder_mode="type", query_bottom_hiddens=qy_bottom_hiddens) # evaluate on query data
                qy_loss = qy_ment_loss + self.type_lam * qy_type_loss
                if one_query['span_mask'].sum().item() == 0:
                    pass
                else:
                    grad = torch.autograd.grad(qy_loss, params) # meta-update
                    meta_grad.append(grad)
            elif mode == "test-onestage":
                self.eval()
                with torch.no_grad():
                    qy_ment_logits, qy_ment_pred, qy_ment_gold, qy_ment_loss, qy_bottom_hiddens = self.forward_ment_step(one_query, encoder_mode="ment") # evaluate on query data
                    qy_type_logits, qy_type_pred, qy_type_gold, qy_type_loss = self.forward_type_step(one_query, encoder_mode="type", query_bottom_hiddens=qy_bottom_hiddens) # evaluate on query data
                    qy_loss = qy_ment_loss + self.type_lam * qy_type_loss
            else:
                assert mode == "test-twostage"
                self.eval()
                with torch.no_grad():
                    qy_ment_logits, qy_ment_pred, qy_ment_gold, qy_ment_loss, qy_bottom_hiddens = self.forward_ment_step(one_query, encoder_mode="ment") # evaluate on query data
                    span_indices, span_mask, span_tag, max_span_num = self.decode_ments(qy_ment_pred)
                    one_query['span_indices'] = span_indices[:, :max_span_num, :].to(qy_ment_logits.device)
                    one_query['span_mask'] = span_mask[:, :max_span_num].to(qy_ment_logits.device)
                    one_query['span_tag'] = span_tag[:, :max_span_num].to(qy_ment_logits.device)
                    one_query['span_weights'] = None
                    two_stage_max_snum = max(two_stage_max_snum, max_span_num)
                    two_stage_query_masks.append(span_mask)
                    two_stage_query_ments.append(span_indices)
                    qy_type_logits, qy_type_pred, qy_type_gold, qy_type_loss = self.forward_type_step(one_query, encoder_mode="type", query_bottom_hiddens=qy_bottom_hiddens) # evaluate on query data
                    qy_loss = qy_ment_loss + self.type_lam * qy_type_loss

            episode_losses.append(qy_loss.item())
            episode_ment_losses.append(qy_ment_loss.item())
            episode_type_losses.append(qy_type_loss.item())
            query_ment_preds.append(qy_ment_pred)
            query_ment_golds.append(qy_ment_gold)
            query_ment_logits.append(qy_ment_logits)
            query_type_preds.append(qy_type_pred)
            query_type_golds.append(qy_type_gold)
            query_type_logits.append(qy_type_logits)
            self.load_weights(names, weights)

            current_query_num += sent_query_num
            current_support_num += sent_support_num

        self.zero_grad()

        if mode == "test-twostage":
            two_stage_query_masks = torch.cat(two_stage_query_masks, dim=0)[:, :two_stage_max_snum]
            two_stage_query_ments = torch.cat(two_stage_query_ments, dim=0)[:, :two_stage_max_snum, :]
            return {'loss': np.mean(episode_losses), 'ment_loss': np.mean(episode_ment_losses), 'type_loss': np.mean(episode_type_losses), 'names': names, 'grads': meta_grad, 
                    'ment_preds': query_ment_preds, 'ment_golds': query_ment_golds, 'ment_logits': query_ment_logits, 
                    'type_preds': query_type_preds, 'type_golds': query_type_golds, 'type_logits': query_type_logits,
                    'pred_spans': two_stage_query_ments, 'pred_masks': two_stage_query_masks
                    }
        else:
            return {'loss': np.mean(episode_losses), 'ment_loss': np.mean(episode_ment_losses), 'type_loss': np.mean(episode_type_losses), 'names': names, 'grads': meta_grad, 
                    'ment_preds': query_ment_preds, 'ment_golds': query_ment_golds, 'ment_logits': query_ment_logits, 
                    'type_preds': query_type_preds, 'type_golds': query_type_golds, 'type_logits': query_type_logits, 
                    }

    def get_named_params(self, no_grads=[]):
        names = []
        params = []
        for n, p in self.named_parameters():
            if any([pn in n for pn in no_grads]):
                continue
            if p.requires_grad:
                names.append(n)
                params.append(p)
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