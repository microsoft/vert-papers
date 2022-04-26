# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from copy import deepcopy
from tqdm import tqdm
import json
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config

k_level = 2


def load_type_map(fn):
    with open(fn, mode="rb") as fp:
        pkl = pickle.load(fp)
        type_map = dict()
        for ent_id in pkl:
            type_map[ent_id] = set(pkl[ent_id]["types"])
        return type_map


def track_parents(type_map, node_id: str, level: int):
    """
    :param node_id: a Q number
    :param level: the number of levels above the current level that we want to go to
    :return: types: a set that contains all the parent types (instance of) of the node up to the number
             levels specified. i.e: If Belgium is a federal system, it's also form of government because federal
             system is an instance of form of government.
    """
    types = deepcopy(type_map.get(node_id, set()))
    pre_level_types = types
    for i in range(level):
        new_level_types = set()
        for t in pre_level_types:
            if t in type_map:
                parents = type_map[t]
            else:
                continue
            for p in parents:
                if p not in types:
                    new_level_types.add(p)
        pre_level_types = new_level_types
        types |= pre_level_types
    return types


def generate_property_values_table(type_map, in_fn, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in tqdm(fp):
                entity = json.loads(line.strip())
                t_labels = track_parents(type_map, entity["id"], k_level)
                obj = {
                    "id": entity['id'],
                    "properties": entity["properties"],
                    "types": list(t_labels),
                    "property_values": entity["property_values"],
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
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/properties_converted.jsonl"
    )
    parser.add_argument(
        "--re_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/properties_converted_to_store.jsonl"
    )
    args = process_relative_path_config(parser.parse_args())
    type_map = load_type_map(args.type_map_rank_fn)
    generate_property_values_table(
        type_map,
        args.in_fn,
        args.re_fn)
