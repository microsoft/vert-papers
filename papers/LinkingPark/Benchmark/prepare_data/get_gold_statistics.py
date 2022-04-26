# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from prettytable import PrettyTable
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def get_gold_statistics(round_dir, data_round):
    with open(os.path.join(round_dir, 'CEA', f'CEA_Round{data_round}_gt.csv'), mode='r') as fp:
        cea_targets = len(fp.readlines())

    with open(os.path.join(round_dir, 'CTA', f'CTA_Round{data_round}_gt.csv'), mode='r') as fp:
        cta_targets = len(fp.readlines())

    with open(os.path.join(round_dir, 'CPA', f'CPA_Round{data_round}_gt.csv'), mode='r') as fp:
        cpa_targets = len(fp.readlines())

    return cea_targets, cta_targets, cpa_targets


def print_targets(base_dir):
    results = PrettyTable()
    results.field_names = [
        "round",
        "cea_targets",
        "cta_targets",
        "cpa_targets"
    ]
    for i in range(1, 5):
        cea_targets, cta_targets, cpa_targets = get_gold_statistics(base_dir, i)
        results.add_row([f'Round{i}', cea_targets, cta_targets, cpa_targets])

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Benchmark/SemTab2020_Data/SemTab2020_Table_GT_Target/GT"
    )
    args = process_relative_path_config(parser.parse_args())
    print_targets(args.gt_dir)
