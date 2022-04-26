# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pickle
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def extract_type_rank_data(fn, re_rank_fn):
    with open(fn, encoding="utf-8", mode="r") as fp:
        type_rank_map = dict()
        # num = 0
        for line in tqdm(fp):
            node = json.loads(line.strip())
            type_rank_map[node["id"]] = {
                "instance_of": [],
                "subclass_of": []
            }

            types = set()
            if 'P31' in node["property_values"]:
                for value in node["property_values"]["P31"]:
                    if value["datatype"] == "wikibase-item":
                        type_rank_map[node["id"]]["instance_of"].append(value["datavalue"]["value"]["id"])
                        types.add(value["datavalue"]["value"]["id"])

            if 'P279' in node["property_values"]:
                for value in node["property_values"]["P279"]:
                    if value["datatype"] == "wikibase-item":
                        type_rank_map[node["id"]]["subclass_of"].append(value["datavalue"]["value"]["id"])
                        types.add(value["datavalue"]["value"]["id"])

            type_rank_map[node["id"]]["types"] = list(types)
            # num += 1
            # if num % 10000 == 0:
            #     print("processing {} lines...".format(num))

        with open(re_rank_fn, mode="wb") as re_fp:
            pickle.dump(type_rank_map, re_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--property_fn",
        type=str,
        default=r"$BASE_DATA_DIR/wikidata\wikidata-20200525\properties.jsonl"
    )
    parser.add_argument(
        "--re_fn",
        type=str,
        default=r"$BASE_DATA_DIR/wikidata\wikidata-20200525\type_map_rank.pkl"
    )
    args = process_relative_path_config(parser.parse_args())

    extract_type_rank_data(args.property_fn,
                           args.re_fn)
    print("done.")