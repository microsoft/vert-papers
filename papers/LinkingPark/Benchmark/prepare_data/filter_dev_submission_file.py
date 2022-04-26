# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_tab_ids(fn):
    tab_ids = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            words = line.strip().split('\t')
            tab_ids.add(words[0])
    return tab_ids


def filter_gold_output(tab_ids, gold_fn, out_fn):
    with open(gold_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in fp:
                words = line.strip().replace("\"", "").split(",")
                if words[0] in tab_ids:
                    re_fp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input/dev"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Benchmark/SemTab2020_Data/SemTab2020_Table_GT_Target/GT"
    )
    args = process_relative_path_config(parser.parse_args())
    for i in range(1, 5):
        tab_ids = load_tab_ids(os.path.join(args.data_dir, f"round{i}_table_dev.json"))
        filter_gold_output(tab_ids,
                           os.path.join(args.out_dir, "CEA", f"CEA_Round{i}_gt.csv"),
                           os.path.join(args.out_dir, "CEA", "dev", f"CEA_Round{i}_dev_gt.csv"))
        filter_gold_output(tab_ids,
                           os.path.join(args.out_dir, "CTA", f"CTA_Round{i}_gt.csv"),
                           os.path.join(args.out_dir, "CTA", "dev", f"CTA_Round{i}_dev_gt.csv"))
        filter_gold_output(tab_ids,
                           os.path.join(args.out_dir, "CPA", f"CPA_Round{i}_gt.csv"),
                           os.path.join(args.out_dir, "CPA", "dev", f"CPA_Round{i}_dev_gt.csv"))