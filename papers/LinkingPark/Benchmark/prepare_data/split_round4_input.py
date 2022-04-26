# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_2t_table_ids(fn):
    tough_table_ids = set()
    with open(fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            table = json.loads(line.strip())
            tough_table_ids.add(table["tab_id"])
        return tough_table_ids


def split_files(tough_tables_ids, in_fn, tt_fn, semtab_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(tt_fn, mode="w", encoding="utf-8") as tt_fp:
            with open(semtab_fn, mode="w", encoding="utf-8") as semtab_fp:
                for line in fp:
                    words = line.strip().split('\t')
                    if words[0] in tough_tables_ids:
                        tt_fp.write(line)
                    else:
                        semtab_fp.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tt_fn",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Golden/2T.jsonl"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input"
    )
    args = process_relative_path_config(parser.parse_args())
    tough_tables_ids = load_2t_table_ids(args.tt_fn)

    split_files(tough_tables_ids,
                f"{args.input_dir}/round4_table.json",
                f"{args.input_dir}/round4_table_tt.json",
                f"{args.input_dir}/round4_table_semtab.json")
