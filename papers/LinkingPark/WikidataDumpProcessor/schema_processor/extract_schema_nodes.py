# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from tqdm import tqdm
import json
from TableAnnotator.Config import shortname
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def extract_schema_nodes(in_fn, out_fn):
    properties_nodes = set()
    type_nodes = set()
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        for line in tqdm(fp):
            node = json.loads(line.strip())
            if node['id'][0] == 'P':
                properties_nodes.add(node['id'])
            type_nodes |= set(node[shortname.TYPES])

    schema_nodes = type_nodes | properties_nodes
    print("number of types : {}".format(len(type_nodes)))
    print("number of properties: {}".format(len(properties_nodes)))
    with open(out_fn, mode="wb") as fp:
        pickle.dump(schema_nodes, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_to_store_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/meta_to_store_compress.jsonl")
    parser.add_argument("--schema_nodes_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/schema_nodes_latest.pkl")
    args = process_relative_path_config(parser.parse_args())
    extract_schema_nodes(args.meta_to_store_fn,
                         args.schema_nodes_fn)