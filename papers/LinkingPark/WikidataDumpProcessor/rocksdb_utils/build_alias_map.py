# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import rocksdb
import json
from tqdm import tqdm
import pickle
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def build_db(db, in_fn):
    with open(in_fn, mode="rb") as fp:
        alias_map = pickle.load(fp)
        for alias in tqdm(alias_map):
            candid = alias_map[alias]
            db.put(bytes(alias, 'utf-8'),
                   bytes(json.dumps(candid), 'utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path",
                        type=str,
                        default="$BASE_DATA_DIR/RocksDB/AliasMap/data_keep_disambiguation.db")
    parser.add_argument("--in_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_keep_disambiguation.pkl")
    args = process_relative_path_config(parser.parse_args())
    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.max_open_files = 300000
    opts.write_buffer_size = 67108864
    opts.max_write_buffer_number = 3
    opts.target_file_size_base = 67108864

    opts.table_factory = rocksdb.BlockBasedTableFactory(
        filter_policy=rocksdb.BloomFilterPolicy(10),
        block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
        block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2)))

    db = rocksdb.DB(args.db_path, opts)
    build_db(db, args.in_fn)