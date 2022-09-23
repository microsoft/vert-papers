# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from Utils.utils_data import pd_read_csv_keep_header
from TableAnnotator.Config.config_utils import process_relative_path_config
import argparse


def convert_format(in_dir, out_fn):
    with open(out_fn, mode="w", encoding="utf-8") as re_fp:
        in_fns = os.listdir(in_dir)
        for in_fn in in_fns:
            tab_id = os.path.basename(in_fn).split('.')[0]
            table = pd_read_csv_keep_header(os.path.join(in_dir, in_fn))
            re_fp.write(f"{tab_id}\t{json.dumps(table)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab21-Data/SemTab/Benchmark/Round3/BioDivTab/tables"
    )
    parser.add_argument(
        "--out-fn",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input/Semtab21_round3_biodivtab_table.json"
    )
    args = process_relative_path_config(parser.parse_args())
    convert_format(args.in_dir, args.out_fn)
