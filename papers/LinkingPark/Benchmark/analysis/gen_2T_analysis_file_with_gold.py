# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import random
import json
from Utils.utils import load_json_table
from Utils.utils import get_item_name
from prettytable import PrettyTable
from tqdm import tqdm
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_gold(fn):
    with open(fn, mode="r", encoding="utf-8") as fp:
        gold = dict()
        for line in fp:
            tab = json.loads(line.strip())
            gold[tab['tab_id']] = tab
        return gold


def load_pred(log_fn, gold):
    pred = dict()
    with open(log_fn, encoding='utf-8', mode='r') as fp:
        for line in tqdm(fp):
            tab = json.loads(line.strip())
            if tab['tab_id'] in gold:
                pred[tab['tab_id']] = tab
    return pred


def load_targets(fn):
    targets = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            targets.add(line.strip().replace('\"', ""))
        return targets


def format_result(tab_id, row_tab):
    col_size = pred[tab_id]['col_size']
    row_size = pred[tab_id]['row_size']
    properties = pred[tab_id]["coarse_properties"]
    col_types = pred[tab_id]["tab_pred_type"]
    entities = pred[tab_id]["revisit_predict_entities"]

    table = []
    # append header
    table.append(["col{}".format(i) for i in range(1, col_size + 1)])

    # pred type: id | name | confidence
    table.append([("\"{}|{}|{:.2f}\"".format(
        col_types[str(j)]["best_type"],
        col_types[str(j)]["best_type_name"],
        col_types[str(j)]["confidence"]) if (str(j) in gold[tab_id]['cta'] and "{},{}".format(tab_id, j) in cta_targets) else "\\") for j in range(col_size)])

    cta_wrong_num = 0
    for x in gold[tab_id]['cta']:
        if gold[tab_id]['cta'][x] != col_types[x]["best_type"]:
            cta_wrong_num += 1

    type_candid_line = []
    # tie nodes
    for j in range(0, col_size):
        if str(j) in gold[tab_id]['cta']:
            type_candid_line.append(
                "\"{}\"".format(",".join(["{}|{}".format(*x) for x in col_types[str(j)]["tie_nodes"]])))
        else:
            type_candid_line.append("\\")

    table.append(type_candid_line)
    # gold cta
    table.append([("\"{}|{}\"".format(
        ', '.join(gold[tab_id]['cta'][str(j)]),
        ', '.join([get_item_name(x) for x in gold[tab_id]['cta'][str(j)]])
    ) if (str(j) in gold[tab_id]['cta'] and "{},{}".format(tab_id, j) in cta_targets) else "\\") for j in range(col_size)])

    # presult = ["Property"]
    presult = []
    for j in range(col_size):
        if str(j) in properties and len(properties[str(j)]) > 0:
            presult.append("\"{}|{}\"".format(properties[str(j)][0]["property"], properties[str(j)][0]["property_name"]))
        else:
            presult.append("\\")

    table.append(presult)
    table.append("\n")
    cea_wrong_num = 0

    for i in range(row_size):
        row_annotation = []
        for j in range(col_size):
            entity = entities[str((i, j))]

            idx = str((i, j))
            if idx in gold[tab_id]['cea'] and "{},{},{}".format(tab_id, i+1, j) in cea_targets:
                if entity["entity"] not in gold[tab_id]['cea'][idx]:
                    row_annotation.append("\"*** {}|{}|{}->{}|{} ***\"".format(row_tab[i][j],
                                                                       entity["entity"],
                                                                       entity["entity_title"].replace(" ", "_"),
                                                                       ', '.join(gold[tab_id]['cea'][idx]),
                                                                       ', '.join([get_item_name(x) for x in gold[tab_id]['cea'][idx]])))

                    cea_wrong_num += 1
                else:
                    row_annotation.append("\"{}|{}|{}->{}|{}\"".format(row_tab[i][j],
                                                                               entity["entity"],
                                                                               entity["entity_title"].replace(" ", "_"),
                                                                               ', '.join(gold[tab_id]['cea'][idx]),
                                                                               ', '.join([get_item_name(x) for x in gold[tab_id]['cea'][idx]])))
            else:
                row_annotation.append("\"{}|\\|\\\"".format(row_tab[i][j]))
        table.append(row_annotation)

    ret = {"format_line": "{}\n".format("\n".join([','.join(line) for line in table])),
           "cea_wrong_num": cea_wrong_num,
           "cta_wrong_num": cta_wrong_num}
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Input/round4_table.json")
    parser.add_argument("--gold-fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Golden/2T.jsonl")
    parser.add_argument("--cea-target-fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Benchmark/2T_WD/2T_WD/targets/CEA_2T_WD_Targets.csv")
    parser.add_argument("--cta-target-fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Benchmark/2T_WD/2T_WD/targets/CTA_2T_WD_Targets.csv")
    parser.add_argument("--log-fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Output/round4_token_char_add_wiki_cand_500_alpha_0.2_beta_0.5_static_type_property/all_output_tables.jsonl")
    parser.add_argument("--analysis-dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Analysis")
    parser.add_argument("--sample-size",
                        type=int,
                        default=50000)
    parser.add_argument("--round",
                        type=str,
                        default='2T')
    print(parser.parse_args())
    args = process_relative_path_config(parser.parse_args())
    print(args)

    random.seed(1234)

    log_fn = os.path.split(os.path.split(args.log_fn)[0])[1]
    analysis_dir = os.path.join(args.analysis_dir, log_fn)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    analysis_dir = os.path.join(analysis_dir, "{}".format(args.round))
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    gold = load_gold(args.gold_fn)
    pred = load_pred(args.log_fn, gold)
    print('data loaded.')

    cea_targets = load_targets(args.cea_target_fn)
    cta_targets = load_targets(args.cta_target_fn)

    table_ids = list(gold.keys())
    random.shuffle(table_ids)
    table_ids = table_ids[:args.sample_size]

    cea_wrong_num = 0
    cta_wrong_num = 0

    tables = load_json_table(args.input_fn)

    for tab_id in tqdm(table_ids):
        # row_tab = pd_read_csv(os.path.join(args.data_dir, "{}.csv".format(tab_id)))
        row_tab = tables[tab_id]
        ret = format_result(tab_id, row_tab)
        cea_wrong_num += ret['cea_wrong_num']
        cta_wrong_num += ret['cta_wrong_num']
        file_name = "{}_cea_{}_cta_{}.csv".format(tab_id, ret['cea_wrong_num'], ret['cta_wrong_num'])
        # if ret['cta_wrong_num'] > 0:
        with open(os.path.join(analysis_dir, file_name), mode="w", encoding="utf-8") as fp:
            fp.write(ret['format_line'])

    results = PrettyTable()
    results.field_names = [
        "cea_wrong_num",
        "cta_wrong_num"
    ]

    results.add_row(
        [cea_wrong_num, cta_wrong_num]
    )
    print(results)
