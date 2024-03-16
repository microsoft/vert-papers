import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FewShotTokenModel(nn.Module):
    def __init__(self, my_word_encoder):
        nn.Module.__init__(self)
        self.word_encoder = nn.DataParallel(my_word_encoder)

    def forward(self, batch):
        raise NotImplementedError

    def span_accuracy(self, pred, gold):
        return np.mean(pred == gold)
    
    def metrics_by_pos(self, preds, golds):
        hit_cnt = 0
        tot_cnt = 0
        for i in range(len(preds)):
            tot_cnt += len(preds[i])
            for j in range(len(preds[i])):
                if preds[i][j] == golds[i][j]:
                    hit_cnt += 1
        return hit_cnt, tot_cnt

    def seq_eval(self, ep_preds, query):
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
            subsent_id += snum
        hit_cnt, tot_cnt = self.metrics_by_pos(sent_pred_labels, sent_gold_labels)
        logs = {
            "index": query["index"],
            "seq_len": query_seq_lens,
            "pred": sent_pred_labels,
            "gold": sent_gold_labels,
            "sentence_num": query["sentence_num"],
            "subsentence_num": query['subsentence_num']
        }
        metric_logs = {
            "gold_cnt": tot_cnt,
            "hit_cnt": hit_cnt
        }
        return metric_logs, logs