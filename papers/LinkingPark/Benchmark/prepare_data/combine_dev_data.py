# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-dir",
        type=str,
        default="$BASE_DATA_DIR/SemTab/Input/dev"
    )
    args = process_relative_path_config(parser.parse_args())

    with open(os.path.join(args.dev_dir, "merged_dev.json"),
              mode="w",
              encoding="utf-8") as re_fp:
        for i in range(1, 5):
            with open(os.path.join(args.dev_dir, f"round{i}_table_dev.json"),
                      mode="r",
                      encoding="utf-8") as fp:
                re_fp.writelines(fp.readlines())