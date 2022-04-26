# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config

entity_prefix = "http://www.wikidata.org/entity/"
property_prefix = "http://www.wikidata.org/prop/direct/"


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
            cea[tab_id][str((int(row_id)-1, int(col_id)))] = [x[len(entity_prefix):] if x != "NIL" else x for x in words[3].split()]

    with open(cta_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id = words[0]
            col_id = words[1]
            if tab_id not in cta:
                cta[tab_id] = dict()
            cta[tab_id][col_id] = [x[len(entity_prefix):] for x in words[2].split()]

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
        "--d",
        type=str,
        default="$BASE_DATA_DIR/SemTab/2T_WD/gt"
    )
    parser.add_argument(
        "--out_d",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Golden"
    )
    args = process_relative_path_config(parser.parse_args())
    gold = read_gold_labels(os.path.join(args.d, "CEA_2T_WD_gt.csv"),
                            os.path.join(args.d, "CTA_2T_WD_gt.csv"))
    dump_jsonl(gold, os.path.join(args.out_d, "2T.jsonl"))
