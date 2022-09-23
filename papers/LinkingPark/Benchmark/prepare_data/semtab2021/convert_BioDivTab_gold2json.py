# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
BioDivTab first row is the table header
"""

import json
from TableAnnotator.Config.config_utils import process_relative_path_config
import argparse


# entity prefix is different from SemTab 2020 which is "http://www.wikidata.org/entity/"
entity_prefix = "https://www.wikidata.org/wiki/"


def read_gold_labels(cea_fn, cta_fn):
    cea = dict()
    cta = dict()
    with open(cea_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id = words[0]
            row_id = words[1]
            col_id = words[2]
            if tab_id not in cea:
                cea[tab_id] = dict()
            cea[tab_id][str((int(row_id), int(col_id)))] = words[3][len(entity_prefix):]

    with open(cta_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id = words[0]
            col_id = words[1]
            if tab_id not in cta:
                cta[tab_id] = dict()
            cta[tab_id][col_id] = words[2][len(entity_prefix):]

    gold = dict()
    tab_ids = set(cea.keys()) | set(cta.keys())
    for tab_id in tab_ids:
        gold[tab_id] = {
            "tab_id": tab_id,
            "cea": cea.get(tab_id, {}),
            "cta": cta.get(tab_id, {}),
        }

    return gold


def dump_jsonl(gold, out_fn):
    with open(out_fn, mode="w", encoding="utf-8") as fp:
        for tab_id in gold:
            # try:
            fp.write("{}\n".format(json.dumps(gold[tab_id])))
            # except:
            #     print(tab_id)
            #     print(gold[tab_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab21-Data/SemTab/Benchmark/Round3/BioDivTab/gt"
    )
    parser.add_argument(
        "--out-fn",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Golden/Semtab21_round3_biodivtab.jsonl"
    )
    args = process_relative_path_config(parser.parse_args())

    gold = read_gold_labels(
        f"{args.in_dir}/CEA_biodivtab_2021_gt.csv",
        f"{args.in_dir}/CTA_biodivtab_2021_gt.csv"
    )
    dump_jsonl(gold, args.out_fn)
