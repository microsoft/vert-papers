import json
from collections import defaultdict
import random
import torch
import prettytable as pt
import numpy as np


def cal_prf(tot_hit_cnt, tot_pred_cnt, tot_gold_cnt):
    precision = tot_hit_cnt / tot_pred_cnt if tot_hit_cnt > 0 else 0
    recall = tot_hit_cnt / tot_gold_cnt if tot_hit_cnt > 0 else 0
    f1 = precision * recall * 2 / (precision + recall) if (tot_hit_cnt > 0) else 0
    return precision, recall, f1


def cal_episode_prf(logs):
    pred_list = logs["pred"]
    gold_list = logs["gold"]
    indexes = logs["index"]

    k = 0
    query_episode_start_sent_idlist = []
    for ep_subsent_num in logs["sentence_num"]:
        query_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["subsentence_num"][k]
            k += 1
    query_episode_start_sent_idlist.append(k)

    subsent_id = 0
    sent_id = 0
    ep_prf = {}
    for cur_preds, cur_golds, cur_index, snum in zip(pred_list, gold_list, indexes, logs["subsentence_num"]):
        if sent_id in query_episode_start_sent_idlist:
            ep_id = query_episode_start_sent_idlist.index(sent_id)
            ep_prf[ep_id] = {"hit": 0, "pred": 0, "gold": 0}

        pred_spans = set(map(lambda x: tuple(x), cur_preds))
        gold_spans = set(map(lambda x: tuple(x), cur_golds))
        ep_prf[ep_id]["pred"] += len(pred_spans)
        ep_prf[ep_id]["gold"] += len(gold_spans)
        ep_prf[ep_id]["hit"] += len(pred_spans.intersection(gold_spans))
        subsent_id += snum
        sent_id += 1
    ep_p = 0
    ep_r = 0
    ep_f1 = 0
    for ep in ep_prf.values():
        p, r, f1 = cal_prf(ep["hit"], ep["pred"], ep["gold"])
        ep_p += p
        ep_r += r
        ep_f1 += f1
    ep_p /= len(ep_prf)
    ep_r /= len(ep_prf)
    ep_f1 /= len(ep_prf)
    return ep_p, ep_r, ep_f1

def eval_ment_log(samples, logs):
    indexes = logs["index"]
    pred_list = logs["pred"]
    gold_list = logs["gold"]
    gold_cnt_mp = defaultdict(int)
    hit_cnt_mp = defaultdict(int)
    for cur_idx, cur_preds, cur_golds in zip(indexes, pred_list, gold_list):
        tagged_spans = samples[cur_idx].spans
        for r, x, y in tagged_spans:
            gold_cnt_mp[r] += 1
            if [x, y] in cur_preds:
                hit_cnt_mp[r] += 1
    tot_miss = 0
    tb = pt.PrettyTable(["Type", "recall", "miss_span", "tot_span"])
    for gtype in sorted(gold_cnt_mp.keys()):
        rscore = hit_cnt_mp[gtype] / gold_cnt_mp[gtype]
        miss_cnt = gold_cnt_mp[gtype] - hit_cnt_mp[gtype]
        tb.add_row([
            gtype, "{:.4f}".format(rscore), miss_cnt, gold_cnt_mp[gtype]
        ])
        tot_miss += miss_cnt
    tb.add_row(["Total", "{:.4f}".format(sum(hit_cnt_mp.values()) / sum(gold_cnt_mp.values())), tot_miss, sum(gold_cnt_mp.values())])
    print(tb)
    return

def write_ep_ment_log_json(samples, logs, output_fn):
    k = 0
    subsent_id = 0
    query_episode_end_subsent_idlist = []
    for ep_subsent_num in logs["sentence_num"]:
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["subsentence_num"][k]
            subsent_id += logs["subsentence_num"][k]
            k += 1
        query_episode_end_subsent_idlist.append(subsent_id)
    indexes = logs["index"]
    if "before_prob" in logs:
        split_ind_list = logs["before_ind"]
        split_prob_list = logs["before_prob"]
    else:
        split_ind_list = None
        split_prob_list = None
    sent_pred_list = logs["pred"]
    split_slen_list = logs["seq_len"]
    subsent_num_list = logs["subsentence_num"]
    log_lines = []
    cur_query_res = {}
    subsent_id = 0
    sent_id = 0
    for snum, cur_index in zip(subsent_num_list, indexes):
        if subsent_id in query_episode_end_subsent_idlist:
            log_lines.append(cur_query_res)
            cur_query_res = {}
        cur_probs = []
        if split_prob_list is not None:
            cur_sent_st = 0
            for k in range(subsent_id, subsent_id + snum):
                for x, y in zip(split_ind_list[k], split_prob_list[k]):
                    cur_probs.append(([x[0] + cur_sent_st, x[1] + cur_sent_st], y))
                cur_sent_st += split_slen_list[k]
        else:
            for x in sent_pred_list[sent_id]:
                cur_probs.append(([x[0], x[1]], 1))
        cur_query_res[samples[cur_index].index] = cur_probs
        subsent_id += snum
        sent_id += 1
    assert subsent_id in query_episode_end_subsent_idlist
    log_lines.append(cur_query_res)
    cur_query_res = {}
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        output_lines = []
        for line in log_lines:
            output_lines.append(json.dumps(line) + "\n")
        fp.writelines(output_lines)
    return

def write_ment_log(samples, logs, output_fn):
    indexes = logs["index"]
    pred_list = logs["pred"]
    gold_list = logs["gold"]
    subsent_num_list = logs["subsentence_num"]
    split_slen_list = logs["seq_len"]
    if "before_prob" in logs:
        split_ind_list = logs["before_ind"]
        split_prob_list = logs["before_prob"]
    else:
        split_ind_list = None
        split_prob_list = None
    log_lines = []
    assert len(pred_list) == len(indexes)
    assert len(gold_list) == len(indexes)
    subsent_id = 0
    for cur_preds, cur_golds, cur_index, snum in zip(pred_list, gold_list, indexes, subsent_num_list):
        cur_sample = samples[cur_index]
        cur_probs = []
        if split_prob_list is not None:
            cur_sent_st = 0
            for k in range(subsent_id, subsent_id + snum):
                for x, y in zip(split_ind_list[k], split_prob_list[k]):
                    cur_probs.append(([x[0] + cur_sent_st, x[1] + cur_sent_st], y))
                cur_sent_st += split_slen_list[k]
        subsent_id += snum
        log_lines.append("index:{}\n".format(cur_sample.index))
        log_lines.append("pred:\n")
        for x in cur_preds:
            log_lines.append(" ".join(cur_sample.words[x[0]: x[1] + 1]) + " " + str(x) + "\n")
        log_lines.append("gold:\n")
        for x in cur_golds:
            log_lines.append(" ".join(cur_sample.words[x[0]: x[1] + 1]) + " " + str(x) + "\n")
        log_lines.append("log:\n")
        log_lines.append(json.dumps(cur_probs) + "\n")
        log_lines.append("\n")
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(log_lines)
    return


def eval_ent_log(samples, logs):
    indexes = logs["index"]
    pred_list = logs["pred"]
    gold_list = logs["gold"]
    assert len(indexes) == len(pred_list)
    gold_cnt_mp = defaultdict(int)
    hit_cnt_mp = defaultdict(int)
    pred_cnt_mp = defaultdict(int)
    hit_mentcnt_mp = defaultdict(int)
    fp_cnt_mp = defaultdict(int)
    for cur_idx, cur_preds, cur_golds in zip(indexes, pred_list, gold_list):
        used_token = np.array([0 for i in range(len(samples[cur_idx].words))])
        cur_ments = [[x[1], x[2]] for x in cur_preds]
        tagged_spans = samples[cur_idx].spans
        for r, x, y in tagged_spans:
            gold_cnt_mp[r] += 1
            used_token[x: y + 1] = 1
            if [r, x, y] in cur_preds:
                hit_cnt_mp[r] += 1
            if [x, y] in cur_ments:
                hit_mentcnt_mp[r] += 1
        for r, x, y in cur_preds:
            pred_cnt_mp[r] += 1
            if sum(used_token[x: y + 1]) == 0:
                fp_cnt_mp[r] += 1
    tb = pt.PrettyTable(["Type", "precision", "recall", "f1", "overall_ment_recall", "error_miss_ment", "false_span", "span_type_error", "correct"])
    for gtype in sorted(gold_cnt_mp.keys()):
        pscore = hit_cnt_mp[gtype] / pred_cnt_mp[gtype] if hit_cnt_mp[gtype] > 0 else 0
        rscore = hit_cnt_mp[gtype] / gold_cnt_mp[gtype] if hit_cnt_mp[gtype] > 0 else 0
        fscore = pscore * rscore * 2 / (pscore + rscore) if hit_cnt_mp[gtype] > 0 else 0
        ment_rscore = hit_mentcnt_mp[gtype] / gold_cnt_mp[gtype]
        miss_ment_cnt = gold_cnt_mp[gtype] - hit_mentcnt_mp[gtype]
        fp_cnt = fp_cnt_mp[gtype]
        type_error_cnt = hit_mentcnt_mp[gtype] - hit_cnt_mp[gtype]
        correct_cnt = hit_cnt_mp[gtype]
        tb.add_row([
            gtype, "{:.4f}".format(pscore), "{:.4f}".format(rscore), "{:.4f}".format(fscore), "{:.4f}".format(ment_rscore), miss_ment_cnt, fp_cnt, type_error_cnt, correct_cnt
        ])
    pscore = sum(hit_cnt_mp.values()) / sum(pred_cnt_mp.values()) if sum(hit_cnt_mp.values()) > 0 else 0
    rscore = sum(hit_cnt_mp.values()) / sum(gold_cnt_mp.values()) if sum(hit_cnt_mp.values()) > 0 else 0
    fscore = pscore * rscore * 2 / (pscore + rscore) if sum(hit_cnt_mp.values()) > 0 else 0
    ment_rscore = sum(hit_mentcnt_mp.values()) / sum(gold_cnt_mp.values())
    miss_ment_cnt = sum(gold_cnt_mp.values()) - sum(hit_mentcnt_mp.values())
    type_error_cnt = sum(hit_mentcnt_mp.values()) - sum(hit_cnt_mp.values())
    tb.add_row([
        "Overall", "{:.4f}".format(pscore), "{:.4f}".format(rscore), "{:.4f}".format(fscore), "{:.4f}".format(ment_rscore), miss_ment_cnt, sum(fp_cnt_mp.values()),type_error_cnt, sum(hit_cnt_mp.values())
    ])
    print(tb)
    return

def write_ent_log(samples, logs, output_fn):
    k = 0
    support_episode_start_sent_idlist = []
    for ep_subsent_num in logs["support_sentence_num"]:
        support_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["support_subsentence_num"][k]
            k += 1
    support_episode_start_sent_idlist.append(k)
    k = 0
    query_episode_start_sent_idlist = []
    for ep_subsent_num in logs["sentence_num"]:
        query_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["subsentence_num"][k]
            k += 1
    query_episode_start_sent_idlist.append(k)
    assert len(support_episode_start_sent_idlist) == len(query_episode_start_sent_idlist)
    indexes = logs["index"]
    pred_list = logs["pred"]
    gold_list = logs["gold"]
    split_ind_list = logs["before_ind"] if "before_ind" in logs else None
    split_prob_list = logs["before_prob"] if "before_prob" in logs else None
    split_slen_list = logs["seq_len"]
    split_label_tag_list = logs["label_tag"] if "label_tag" in logs else None

    log_lines = []
    subsent_id = 0
    sent_id = 0
    for cur_preds, cur_golds, cur_index, snum in zip(pred_list, gold_list, indexes, logs["subsentence_num"]):
        if sent_id in query_episode_start_sent_idlist:
            ep_id = query_episode_start_sent_idlist.index(sent_id)
            support_sent_id_1 = support_episode_start_sent_idlist[ep_id]
            support_sent_id_2 = support_episode_start_sent_idlist[ep_id + 1]
            log_lines.append("="*20+"\n")
            support_indexes = logs["support_index"][support_sent_id_1:support_sent_id_2]
            log_lines.append("support:{}\n".format(str(support_indexes)))
            for x in support_indexes:
                log_lines.append(str(samples[x].words) + "\n")
                log_lines.append(str([sp[0] + ":" + " ".join(samples[x].words[sp[1]: sp[2] + 1]) for sp in samples[x].spans]) + "\n")
            if split_label_tag_list is not None:
                log_lines.append(json.dumps(split_label_tag_list[subsent_id]) + "\n")
            log_lines.append("\n")
        cur_sample = samples[cur_index]
        cur_probs = []
        if split_ind_list is not None:
            cur_sent_st = 0
            for k in range(subsent_id, subsent_id + snum):
                for x, y_list in zip(split_ind_list[k], split_prob_list[k]):
                    cur_probs.append(str([x[0] + cur_sent_st, x[1] + cur_sent_st])
                                    + ": " + ",".join(["{:.5f}".format(y) for y in y_list])
                                    + ", "
                                    + " ".join(cur_sample.words[x[0] + cur_sent_st: x[1] + cur_sent_st + 1]) + "\n")
                cur_sent_st += split_slen_list[k]
        subsent_id += snum

        log_lines.append("index:{}\n".format(cur_index))
        log_lines.append(str(cur_sample.words) + "\n")
        log_lines.append("pred:\n")
        for x in cur_preds:
            log_lines.append(" ".join(cur_sample.words[x[1]: x[2] + 1]) + " " + str(x) + "\n")
        log_lines.append("gold:\n")
        for x in cur_golds:
            log_lines.append(" ".join(cur_sample.words[x[1]: x[2] + 1]) + " " + str(x) + "\n")
        log_lines.append("log:\n")
        log_lines.extend(cur_probs)
        log_lines.append("\n")

        sent_id += 1
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines(log_lines)
    return log_lines


def write_ent_pred_json(samples, logs, output_fn):
    k = 0
    support_episode_start_sent_idlist = []
    for ep_subsent_num in logs["support_sentence_num"]:
        support_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["support_subsentence_num"][k]
            k += 1
    support_episode_start_sent_idlist.append(k)
    k = 0
    query_episode_start_sent_idlist = []
    for ep_subsent_num in logs["sentence_num"]:
        query_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["subsentence_num"][k]
            k += 1
    query_episode_start_sent_idlist.append(k)
    assert len(support_episode_start_sent_idlist) == len(query_episode_start_sent_idlist)
    indexes = logs["index"]
    pred_list = logs["pred"]
    log_lines = []
    cur_query_res = None
    support_indexes = None
    subsent_id = 0
    sent_id = 0
    for cur_preds, cur_index, snum in zip(pred_list, indexes, logs["subsentence_num"]):
        if sent_id in query_episode_start_sent_idlist:
            if cur_query_res is not None:
                log_lines.append({"support": support_indexes, "query": cur_query_res})
            ep_id = query_episode_start_sent_idlist.index(sent_id)
            support_sent_id_1 = support_episode_start_sent_idlist[ep_id]
            support_sent_id_2 = support_episode_start_sent_idlist[ep_id + 1]
            support_indexes = logs["support_index"][support_sent_id_1:support_sent_id_2]
            cur_query_res = []

        cur_sample = samples[cur_index]
        subsent_id += snum
        cur_query_res.append({"index":cur_index, "pred":cur_preds, "gold": cur_sample.spans})
        sent_id += 1
    if len(cur_query_res) > 0:
        log_lines.append({"support": support_indexes, "query": cur_query_res})
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        output_lines = []
        for line in log_lines:
            output_lines.append(json.dumps(line) + "\n")
        fp.writelines(output_lines)
    return


def write_pos_pred_json(samples, logs, output_fn):
    k = 0
    support_episode_start_sent_idlist = []
    for ep_subsent_num in logs["support_sentence_num"]:
        support_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["support_subsentence_num"][k]
            k += 1
    support_episode_start_sent_idlist.append(k)
    k = 0
    query_episode_start_sent_idlist = []
    for ep_subsent_num in logs["sentence_num"]:
        query_episode_start_sent_idlist.append(k)
        cur_subsent_num = 0
        while cur_subsent_num < ep_subsent_num:
            cur_subsent_num += logs["subsentence_num"][k]
            k += 1
    query_episode_start_sent_idlist.append(k)
    assert len(support_episode_start_sent_idlist) == len(query_episode_start_sent_idlist)
    indexes = logs["index"]
    pred_list = logs["pred"]
    log_lines = []
    cur_query_res = None
    support_indexes = None
    subsent_id = 0
    sent_id = 0
    for cur_preds, cur_index, snum in zip(pred_list, indexes, logs["subsentence_num"]):
        if sent_id in query_episode_start_sent_idlist:
            if cur_query_res is not None:
                log_lines.append({"support": support_indexes, "query": cur_query_res})
            ep_id = query_episode_start_sent_idlist.index(sent_id)
            support_sent_id_1 = support_episode_start_sent_idlist[ep_id]
            support_sent_id_2 = support_episode_start_sent_idlist[ep_id + 1]
            support_indexes = logs["support_index"][support_sent_id_1:support_sent_id_2]
            cur_query_res = []

        cur_sample = samples[cur_index]
        subsent_id += snum
        cur_query_res.append({"index":cur_index, "pred":cur_preds, "gold": cur_sample.tags})
        sent_id += 1
    if len(cur_query_res) > 0:
        log_lines.append({"support": support_indexes, "query": cur_query_res})
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        output_lines = []
        for line in log_lines:
            output_lines.append(json.dumps(line) + "\n")
        fp.writelines(output_lines)
    return

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return
