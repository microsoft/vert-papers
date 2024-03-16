import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from util.span_sample import convert_bio2spans

class FewShotSpanModel(nn.Module):
    def __init__(self, my_word_encoder):
        nn.Module.__init__(self)
        self.word_encoder = nn.DataParallel(my_word_encoder)

    def forward(self, batch):
        raise NotImplementedError

    def span_accuracy(self, pred, gold):
        return np.mean(pred == gold)

    def merge_ment(self, span_list, seq_lens, subsentence_nums):
        new_span_list = []
        subsent_id = 0
        for snum in subsentence_nums:
            sent_st_id = 0
            whole_sent_span_list = []
            for k in range(subsent_id, subsent_id + snum):
                tmp = [[x[0] + sent_st_id, x[1] + sent_st_id] for x in span_list[k]]
                tmp = sorted(tmp, key=lambda x: x[1], reverse=False)
                if len(whole_sent_span_list) > 0 and len(tmp) > 0:
                    if tmp[0][0] == whole_sent_span_list[-1][1] + 1:
                        whole_sent_span_list[-1][1] = tmp[0][1]
                        tmp = tmp[1:]
                whole_sent_span_list.extend(tmp)
                sent_st_id += seq_lens[k]
            subsent_id += snum
            new_span_list.append(whole_sent_span_list)
        assert len(new_span_list) == len(subsentence_nums)
        return new_span_list

    def get_mention_prob(self, ment_probs, query):
        span_indices = query["span_indices"].detach().cpu()
        span_masks = query['span_mask'].detach().cpu()
        ment_probs = ment_probs.detach().cpu().tolist()
        cand_indices_list = []
        cand_prob_list = []
        cur_span_num = 0
        for k in range(len(span_indices)):
            mask = span_masks[k, :]
            effect_indices = span_indices[k, mask.eq(1)].tolist()
            effect_probs = ment_probs[cur_span_num: cur_span_num + len(effect_indices)]
            cand_indices_list.append(effect_indices)
            cand_prob_list.append(effect_probs)
            cur_span_num += len(effect_indices)
        return cand_indices_list, cand_prob_list

    def filter_by_threshold(self, cand_indices_list, cand_prob_list, threshold):
        final_indices_list = []
        final_prob_list = []
        for indices, probs in zip(cand_indices_list, cand_prob_list):
            final_indices_list.append([])
            final_prob_list.append([])
            for x, y in zip(indices, probs):
                if y > threshold:
                    final_indices_list[-1].append(x)
                    final_prob_list[-1].append(y)
        return final_indices_list, final_prob_list

    def seqment_eval(self, ep_preds, query, idx2label, schema): 
        query_seq_lens = query["seq_len"].detach().cpu().tolist()
        ment_indices_list = []
        sid = 0
        for batch_preds in ep_preds:
            for pred in batch_preds.detach().cpu().tolist():
                seqs = [idx2label[idx] for idx in pred[:query_seq_lens[sid]]]
                ents = convert_bio2spans(seqs, schema)
                ment_indices_list.append([[x[1], x[2]] for x in ents])
                sid += 1
        pred_ments_list = self.merge_ment(ment_indices_list, query_seq_lens, query['subsentence_num'])
        gold_ments_list = []            
        subsent_id = 0
        query['ment_labels'] = query['ment_labels'].cpu().tolist()
        for snum in query['subsentence_num']:
            whole_sent_labels = []
            for k in range(subsent_id, subsent_id + snum):
                whole_sent_labels += query['ment_labels'][k][:query_seq_lens[k]]
            ents = convert_bio2spans([idx2label[idx] for idx in whole_sent_labels], schema)
            gold_ments_list.append([[x[1], x[2]] for x in ents])
            subsent_id += snum
        pred_cnt, gold_cnt, hit_cnt = self.metrics_by_entity(pred_ments_list, gold_ments_list)
        logs = {
            "index": query["index"],
            "seq_len": query_seq_lens,
            "pred": pred_ments_list,
            "gold": gold_ments_list,
            "sentence_num": query["sentence_num"],
            "subsentence_num": query['subsentence_num']
        }
        metric_logs = {
            "ment_pred_cnt": pred_cnt,
            "ment_gold_cnt": gold_cnt,
            "ment_hit_cnt": hit_cnt,
        }
        return metric_logs, logs

    def ment_eval(self, ep_probs, query, threshold=0.5):
        if len(ep_probs) == 0:
            print("no mention")
            return {}, {}
        probs = torch.cat(ep_probs, dim=0)
        cand_indices_list, cand_prob_list = self.get_mention_prob(probs, query)
        ment_indices_list, ment_prob_list = self.filter_by_threshold(cand_indices_list, cand_prob_list, threshold)
        gold_ments_list = []
        for sent_spans in query['spans']:
            gold_ments_list.append([])
            for tagid, sp_st, sp_ed in sent_spans:
                gold_ments_list[-1].append([sp_st, sp_ed])
        assert len(ment_indices_list) == len(query['seq_len'])
        assert len(gold_ments_list) == len(query['seq_len'])
        assert len(ment_indices_list) >= len(query['index'])
        query_seq_lens = query["seq_len"].detach().cpu().tolist()
        pred_ments_list = self.merge_ment(ment_indices_list, query_seq_lens, query['subsentence_num'])
        gold_ments_list = self.merge_ment(gold_ments_list, query_seq_lens, query['subsentence_num'])
        pred_cnt, gold_cnt, hit_cnt = self.metrics_by_entity(pred_ments_list, gold_ments_list)
        logs = {
            "index": query["index"],
            "seq_len": query_seq_lens,
            "pred": pred_ments_list,
            "gold": gold_ments_list,
            "before_ind": ment_indices_list,
            "before_prob": ment_prob_list,
            "sentence_num": query["sentence_num"],
            "subsentence_num": query['subsentence_num']
        }
        metric_logs = {
            "ment_pred_cnt": pred_cnt,
            "ment_gold_cnt": gold_cnt,
            "ment_hit_cnt": hit_cnt,
        }
        return metric_logs, logs

    def metrics_by_ment(self, pred_spans_list, gold_spans_list):
        pred_cnt, gold_cnt, hit_cnt = 0, 0, 0
        for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
            pred_spans = set(map(lambda x: (x[1], x[2]), pred_spans))
            gold_spans = set(map(lambda x: (x[1], x[2]), gold_spans))
            pred_cnt += len(pred_spans)
            gold_cnt += len(gold_spans)
            hit_cnt += len(pred_spans.intersection(gold_spans))
        return pred_cnt, gold_cnt, hit_cnt

    def get_emission(self, ep_logits, query):
        assert len(query['label2tag']) == len(query['sentence_num'])
        span_indices = query["span_indices"]
        cur_sent_num = 0
        cand_indices_list = []
        cand_prob_list = []
        label2tag_list = []
        for i, query_sent_num in enumerate(query['sentence_num']):
            probs = F.softmax(ep_logits[i], dim=-1).detach().cpu()
            cur_span_num = 0
            for j in range(query_sent_num):
                mask = query['span_mask'][cur_sent_num, :] 
                effect_indices = span_indices[cur_sent_num, mask.eq(1)].detach().cpu().tolist()
                cand_indices_list.append(effect_indices)
                cand_prob_list.append(probs[cur_span_num:cur_span_num + len(effect_indices)].numpy())
                label2tag_list.append(query['label2tag'][i])
                cur_sent_num += 1
                cur_span_num += len(effect_indices)
        return cand_indices_list, cand_prob_list, label2tag_list

    def to_triple_score(self, indices_list, prob_list, label2tag):
        tri_list = []
        for sp_id, (i, j) in enumerate(indices_list):
            k = np.argmax(prob_list[sp_id, :])
            if label2tag[k] == 'O':
                assert k == 0
                continue
            else:
                tri_list.append([label2tag[k], i, j, prob_list[sp_id, k]])
        return tri_list

    def greedy_search(self, span_indices_list, prob_list, label2tag_list, seq_len_list, overlap=False, threshold=-1):
        output_spans_list = []
        for sid in range(len(span_indices_list)):
            output_spans_list.append([])
            tri_list = self.to_triple_score(span_indices_list[sid], prob_list[sid], label2tag_list[sid])
            sorted_tri_list = sorted(tri_list, key=lambda x: x[-1], reverse=True)
            used_words = np.zeros(seq_len_list[sid])
            for tag, sp_st, sp_ed, score in sorted_tri_list:
                if score < threshold:
                    continue
                if sum(used_words[sp_st: sp_ed + 1]) > 0 and (not overlap):
                    continue
                used_words[sp_st: sp_ed + 1] = 1
                assert tag != "O"
                output_spans_list[sid].append([tag, sp_st, sp_ed])
        return output_spans_list

    def merge_entity(self, span_list, seq_lens, subsentence_nums):
        new_span_list = []
        subsent_id = 0
        for snum in subsentence_nums:
            sent_st_id = 0
            whole_sent_span_list = []
            for k in range(subsent_id, subsent_id + snum):
                tmp = [[x[0], x[1] + sent_st_id, x[2] + sent_st_id] for x in span_list[k]]
                tmp = sorted(tmp, key=lambda x: x[2], reverse=False)
                if len(whole_sent_span_list) > 0 and len(tmp) > 0:
                    if whole_sent_span_list[-1][0] == tmp[0][0] and tmp[0][1] == whole_sent_span_list[-1][2] + 1:
                        whole_sent_span_list[-1][2] = tmp[0][2]
                        tmp = tmp[1:]
                whole_sent_span_list.extend(tmp)
                sent_st_id += seq_lens[k]
            subsent_id += snum
            new_span_list.append(whole_sent_span_list)
        assert len(new_span_list) == len(subsentence_nums)
        return new_span_list

    def seq_eval(self, ep_preds, query, schema):
        query_seq_lens = query["seq_len"].detach().cpu().tolist()
        subsent_label2tag_ids = []
        subsent_pred_ids = []
        for k, batch_preds in enumerate(ep_preds):
            batch_preds = batch_preds.detach().cpu().tolist()
            for pred in batch_preds:
                subsent_pred_ids.append(pred)
                subsent_label2tag_ids.append(k)

        sent_gold_labels = []
        sent_pred_labels = []
        pred_spans_list = []
        gold_spans_list = []
        subsent_id = 0
        query['word_labels'] = query['word_labels'].cpu().tolist()
        for snum in query['subsentence_num']:
            whole_sent_gids = []
            whole_sent_pids = []
            for k in range(subsent_id, subsent_id + snum):
                whole_sent_gids += query['word_labels'][k][:query_seq_lens[k]]
                whole_sent_pids += subsent_pred_ids[k][:query_seq_lens[k]]
            label2tag = query['label2tag'][subsent_label2tag_ids[subsent_id]]
            sent_gold_labels.append([label2tag[lid] for lid in whole_sent_gids])
            sent_pred_labels.append([label2tag[lid] for lid in whole_sent_pids])
            gold_spans_list.append(convert_bio2spans(sent_gold_labels[-1], schema))
            pred_spans_list.append(convert_bio2spans(sent_pred_labels[-1], schema))
            subsent_id += snum
        pred_cnt, gold_cnt, hit_cnt = self.metrics_by_entity(pred_spans_list, gold_spans_list)
        ment_pred_cnt, ment_gold_cnt, ment_hit_cnt = self.metrics_by_ment(pred_spans_list, gold_spans_list)
        logs = {
            "index": query["index"],
            "seq_len": query_seq_lens,
            "pred": pred_spans_list,
            "gold": gold_spans_list,
            "sentence_num": query["sentence_num"],
            "subsentence_num": query['subsentence_num']
        }
        metric_logs = {
            "ment_pred_cnt": ment_pred_cnt,
            "ment_gold_cnt": ment_gold_cnt,
            "ment_hit_cnt": ment_hit_cnt,
            "ent_pred_cnt": pred_cnt,
            "ent_gold_cnt": gold_cnt,
            "ent_hit_cnt": hit_cnt
        }
        return metric_logs, logs

    def greedy_eval(self, ep_logits, query, overlap=False, threshold=-1):
        if len(ep_logits) == 0:
            print("no entity")
            return {}, {}
        query_seq_lens = query["seq_len"].detach().cpu().tolist()
        cand_indices_list, cand_prob_list, label2tag_list = self.get_emission(ep_logits, query)
        pred_spans_list = self.greedy_search(cand_indices_list, cand_prob_list, label2tag_list, query_seq_lens, overlap, threshold)

        gold_spans_list = []
        subsent_idx = 0
        for sent_spans, label2tag in zip(query['spans'], label2tag_list):
            gold_spans_list.append([])
            for tagid, sp_st, sp_ed in sent_spans:
                gold_spans_list[-1].append([label2tag[tagid], sp_st, sp_ed])
            subsent_idx += 1
        assert len(pred_spans_list) == len(query['seq_len'])
        assert len(gold_spans_list) == len(query['seq_len'])
        assert len(pred_spans_list) >= len(query['index'])
        pred_spans_list = self.merge_entity(pred_spans_list, query_seq_lens, query['subsentence_num'])
        gold_spans_list = self.merge_entity(gold_spans_list, query_seq_lens, query['subsentence_num'])
        pred_cnt, gold_cnt, hit_cnt = self.metrics_by_entity(pred_spans_list, gold_spans_list)
        ment_pred_cnt, ment_gold_cnt, ment_hit_cnt = self.metrics_by_ment(pred_spans_list, gold_spans_list)
        logs = {
            "index": query["index"],
            "seq_len": query_seq_lens,
            "pred": pred_spans_list,
            "gold": gold_spans_list,
            "before_ind": cand_indices_list,
            "before_prob": cand_prob_list,
            "label_tag": label2tag_list,
            "sentence_num": query["sentence_num"],
            "subsentence_num": query['subsentence_num']
        }
        metric_logs = {
            "ment_pred_cnt": ment_pred_cnt,
            "ment_gold_cnt": ment_gold_cnt,
            "ment_hit_cnt": ment_hit_cnt,
            "ent_pred_cnt": pred_cnt,
            "ent_gold_cnt": gold_cnt,
            "ent_hit_cnt": hit_cnt
        }
        return metric_logs, logs

    def metrics_by_entity(self, pred_spans_list, gold_spans_list): 
        pred_cnt, gold_cnt, hit_cnt = 0, 0, 0
        for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
            pred_spans = set(map(lambda x: tuple(x), pred_spans))
            gold_spans = set(map(lambda x: tuple(x), gold_spans))
            pred_cnt += len(pred_spans)
            gold_cnt += len(gold_spans)
            hit_cnt += len(pred_spans.intersection(gold_spans))
        return pred_cnt, gold_cnt, hit_cnt

    def metrics_by_ment(self, pred_spans_list, gold_spans_list):
        pred_cnt, gold_cnt, hit_cnt = 0, 0, 0
        for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
            pred_spans = set(map(lambda x: (x[1], x[2]), pred_spans))
            gold_spans = set(map(lambda x: (x[1], x[2]), gold_spans))
            pred_cnt += len(pred_spans)
            gold_cnt += len(gold_spans)
            hit_cnt += len(pred_spans.intersection(gold_spans))
        return pred_cnt, gold_cnt, hit_cnt