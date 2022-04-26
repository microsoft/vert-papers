# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from tqdm import tqdm
import json
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_schema_set(schema_fn):
    with open(schema_fn, mode="rb") as fp:
        schema_ent_set = pickle.load(fp)
        return schema_ent_set


def extract_schema_nodes(in_fn, schema_ent_set, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in tqdm(fp):
                node = json.loads(line.strip())
                if node['id'] in schema_ent_set:
                    re_fp.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_to_store_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/meta_to_store_compress.jsonl")
    parser.add_argument("--schema_nodes_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/schema_nodes_latest.pkl")
    parser.add_argument("--schema_meta_to_store_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/schema_meta_to_store_compress.jsonl")
    args = process_relative_path_config(parser.parse_args())
    schema_ent_set = load_schema_set(args.schema_nodes_fn)
    extract_schema_nodes(args.meta_to_store_fn,
                         schema_ent_set,
                         args.schema_meta_to_store_fn)