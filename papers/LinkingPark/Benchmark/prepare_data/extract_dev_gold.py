# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_tab_ids(fn):
    tab_ids = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            words = line.strip().split()
            tab_ids.add(words[0])
        return tab_ids


def filter_gold(in_fn, tab_ids, re_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(re_fn, mode="w", encoding="utf-8") as re_fp:
            for line in fp:
                tab_id = line.replace('"', '').split(',')[0]
                if tab_id in tab_ids:
                    re_fp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="$BASE_DATA_DIR"
    )
    args = process_relative_path_config(parser.parse_args())

    for round in range(1, 4):
        tab_ids = load_tab_ids(f"{args.base_path}/SemTab/Input/dev/round{round}_table_dev.json")
        print('len(tab_ids) = {}'.format(len(tab_ids)))
        filter_gold(f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CEA/CEA_Round{round}_gt.csv",
                    tab_ids,
                    f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CEA/CEA_Round{round}_dev_gt.csv")
        filter_gold(f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CTA/CTA_Round{round}_gt.csv",
                    tab_ids,
                    f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CTA/CTA_Round{round}_dev_gt.csv")
        filter_gold(f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CPA/CPA_Round{round}_gt.csv",
                    tab_ids,
                    f"{args.base_path}/SemTab/SemTab2020_Data/SemTab2020_Table_GT_Target/GT/CPA/CPA_Round{round}_dev_gt.csv")
