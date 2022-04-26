# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from Utils.utils_data import pd_read_csv
from TableAnnotator.Config.config_utils import process_relative_path_config
import argparse


def convert_format(in_dir, out_fn):
    with open(out_fn, mode="w", encoding="utf-8") as re_fp:
        in_fns = os.listdir(in_dir)
        for in_fn in in_fns:
            tab_id = os.path.basename(in_fn).split('.')[0]
            table = pd_read_csv(os.path.join(in_dir, in_fn))
            re_fp.write(f"{tab_id}\t{json.dumps(table)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Benchmark/2T_WD/2T_WD"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input"
    )
    args = process_relative_path_config(parser.parse_args())
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    convert_format(args.in_dir, os.path.join(args.out_dir, "round4_table_tt.json"))
    print("Done.")
