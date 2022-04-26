# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import json
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_pkl(fn):
    with open(fn, mode="rb") as fp:
        pkl = pickle.load(fp)
        return pkl


def extract_aliases(ent_info):
    aliases = set()
    for lang_code in ent_info["labels"]:
        aliases.add(ent_info["labels"][lang_code])

    for alias in ent_info["aliases"]:
        aliases.add(alias)
    return list(aliases)


def generate_ent_meta_table(ent_types, in_fn, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in tqdm(fp):
                entity = json.loads(line.strip())
                ent_name = entity["labels"].get("en", "No_label_defined")
                aliases = extract_aliases(entity)
                types = []
                types_rank = {
                    'instance_of': [],
                    'subclass_of': []
                }

                if entity["id"] in ent_types:
                    types = ent_types[entity["id"]]["types"]
                    types_rank = {
                        'instance_of': ent_types[entity["id"]]['instance_of'],
                        'subclass_of': ent_types[entity["id"]]['subclass_of']
                    }

                obj = {
                    "id": entity["id"],
                    "ent_name": ent_name,
                    "aliases": aliases,
                    "types": types,
                    "types_rank": types_rank
                }

                re_fp.write("{}\n".format(json.dumps(obj)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_map_rank_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/type_map_rank.pkl"
    )
    parser.add_argument(
        "--in_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/meta.jsonl"
    )
    parser.add_argument(
        "--re_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/meta_to_store.jsonl"
    )
    args = process_relative_path_config(parser.parse_args())
    ent_types = load_pkl(args.type_map_rank_fn)
    generate_ent_meta_table(ent_types,
                            args.in_fn,
                            args.re_fn)

