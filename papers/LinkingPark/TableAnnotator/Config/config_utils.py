# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from TableAnnotator.Config import config


def process_config(args: argparse.Namespace):

    # Read base environment variables
    data_path = os.environ.get("DATA_PATH")
    lp_path = os.environ.get("LP_HOME")

    if lp_path is None:
        print(f"Warning: LP_HOME environment variable not set. Using default value: {config.DEFAULT_SYSTEM_DIR}")
    else:
        config.base_system_dir = lp_path

    if data_path is None:
        print(f"Warning: DATA_PATH environment variable not set. Using default value: {config.DEFAULT_DATA_DIR}")
    else:
        config.base_data_dir = data_path

    args_dict = {}
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    else:
        print("Warming: Config must be a namespace. " + type(args))
        return args

    if 'port' in args_dict.keys():
        config.port = args.port

    # rewrite relative paths by var references
    for key, value in args_dict.items():
        if isinstance(value, str):
            if '$BASE_DATA_DIR' in value or '$BASE_SYSTEM_DIR' in value:
                args_dict[key] = value.replace('$BASE_DATA_DIR', config.base_data_dir).replace('$BASE_SYSTEM_DIR', config.base_data_dir)
            elif value.startswith('./') or value.startswith('.\\'):
                path = data_path
                if '/configs/' in value or '\\configs\\' in value or '/static' in value or '\\static' in value:
                    path = lp_path
                args_dict[key] = value.replace('./', path + '/').replace('.\\', path + '\\')

    params = {
        "k_level": config.k_level,
        "candid_map_fn": config.candid_map_fn,
        "type_count_fn": config.type_count_fn,
        "property_value_fn": config.property_value_fn,
        "entity_meta_fn": config.entity_meta_fn,
        "init_prune_topk": args_dict['init_prune_topk'],
        "topk": args_dict['topk'],
        "keep_N": args_dict['keep_N'],
        "alpha": args_dict['alpha'],
        "beta": args_dict['beta'],
        "gamma": args_dict['gamma'],
        "max_iter": args_dict['max_iter'],
        "use_characteristics": args_dict['use_characteristics'],
        "min_final_diff": args_dict['min_final_diff'],
        # weight for property
        "strict_match_weight": 1.0,
        "fuzzy_match_weight": 0.8,
        "characteristic_match_weight": 0.7,
        "row_feature_only": args_dict['row_feature_only'],
        "ent_feature": args_dict['ent_feature'],
        "alias_map_fn": args_dict['alias_map_fn'],
        "id_mapping_fn": args_dict['id_mapping_fn'],
        "index_name": args_dict['index_name'],
        "in_links_fn": args_dict['in_links_fn'],
        "kb_store": config.kb_store,
        "schema_entity_meta_fn": config.schema_entity_meta_fn,
        "candid_gen_method": config.candid_gen_method,
    }

    return params


def process_relative_path_config(args: argparse.Namespace):

    # Read base environment variables
    data_path = os.environ.get("BASE_DATA_DIR")

    if data_path is None:
        print(f"Warning: BASE_DATA_DIR environment variable not set. Using default value: {config.DEFAULT_DATA_DIR}")
    else:
        config.base_data_dir = data_path

    args_dict = {}
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    else:
        print("Warming: Config must be a namespace. " + type(args))
        return args

    # rewrite relative paths by var references
    for key, value in args_dict.items():
        if isinstance(value, str):
            if '$BASE_DATA_DIR' in value or '$BASE_SYSTEM_DIR' in value:
                args_dict[key] = value.replace('$BASE_DATA_DIR', config.base_data_dir).replace('$BASE_SYSTEM_DIR', config.base_data_dir)

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':

    vals = {
        "in_links_fn": "$BASE_DATA_DIR/wikidata/incoming_links/in_coming_links_num.pkl",
        "port": 2
    }

    ns = argparse.Namespace(**vals)
    config = process_config(ns)
