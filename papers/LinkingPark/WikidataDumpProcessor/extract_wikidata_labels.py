# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pickle
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def extract_labels_data(fn, re_fn):
    with open(fn, encoding="utf-8", mode="r") as fp:
        label_map = dict()

        for line in tqdm(fp):
            node = json.loads(line.strip())
            label_map[node['id']] = node['labels']

        with open(re_fn, mode="wb") as re_fp:
            pickle.dump(label_map, re_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_info_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/meta.jsonl"
    )
    parser.add_argument(
        "--re_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/label_map.pkl"
    )
    args = process_relative_path_config(parser.parse_args())

    extract_labels_data(args.meta_info_fn,
                        args.re_fn)
    print("done.")