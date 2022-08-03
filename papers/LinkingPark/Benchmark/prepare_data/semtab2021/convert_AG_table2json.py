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
        default="$BASE_DATA_DIR/SemTab21-Data/SemTab/Benchmark/"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input"
    )
    args = process_relative_path_config(parser.parse_args())
    for r in [2, 3]:
        in_dir = os.path.join(args.in_dir, f"Round{r}/HardTable/tables")
        out_fn = f"{args.out_dir}/Semtab21_round{r}_hard_table.json"
        convert_format(in_dir, out_fn)
