# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pickle
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def extract_alias_data(fn, re_fn):
    with open(fn, encoding="utf-8", mode="r") as fp:
        alias_map = dict()

        for line in tqdm(fp):
            aliases = set()
            node = json.loads(line.strip())
            for lang_code in node["labels"]:
                aliases.add(node["labels"][lang_code])

            for alias in node["aliases"]:
                aliases.add(alias)

            for alias in aliases:
                if alias not in alias_map:
                    alias_map[alias] = []
                alias_map[alias].append(node["id"])
            # num += 1
            # if num % 10000 == 0:
            #     print("processing {} lines...".format(num))

        with open(re_fn, mode="wb") as re_fp:
            pickle.dump(alias_map, re_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_info_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/meta.jsonl"
    )
    parser.add_argument(
        "--re_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/wikidata_alias_map2.pkl"
    )
    args = process_relative_path_config(parser.parse_args())

    extract_alias_data(args.meta_info_fn,
                       args.re_fn)
    print("done.")