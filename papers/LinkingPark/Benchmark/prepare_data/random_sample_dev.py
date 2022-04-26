# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import json
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config

random.seed(1234)


def load_round4(fn):
    ids = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            table = json.loads(line.strip())
            ids.add(table['tab_id'])
        return ids


def sample_dev(in_fn, out_fn, tab_ids, topk=100):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        lines = []
        for line in fp:
            words = line.strip().split('\t')
            if words[0] in tab_ids:
                lines.append(line)
        # lines = fp.readlines()
        random.shuffle(lines)
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            re_fp.writelines(lines[:topk])


def sample_dev_v2(in_fn, out_fn, topk=100):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        lines = fp.readlines()
        random.shuffle(lines)
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            re_fp.writelines(lines[:topk])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="$BASE_DATA_DIR/Input"
    )
    args = process_relative_path_config(parser.parse_args())
    for round in range(1, 4):
        sample_dev_v2(f"{args.input_dir}/round{round}_table.json",
                      f"{args.input_dir}/round{round}_table_dev.json",
                      topk=500)
