# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
In this script, we define some common usage utils functions
"""

import datetime

import json
import redis
from GlobalConfig import global_config
from TableAnnotator.Config import shortname


entity_meta_info = redis.Redis(host=global_config.ent_meta_redis_host, port=global_config.ent_meta_redis_port, db=0)
property_meta_info = redis.Redis(host=global_config.property_redis_host, port=global_config.property_redis_port, db=0)

if global_config.use_rocks_db:
    import rocksdb

    def setup_rocksdb(db_path):
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 30000
        opts.write_buffer_size = 67108864
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 67108864

        opts.table_factory = rocksdb.BlockBasedTableFactory(
            filter_policy=rocksdb.BloomFilterPolicy(10),
            block_cache=rocksdb.LRUCache(2 * (1024 ** 3)),
            block_cache_compressed=rocksdb.LRUCache(500 * (1024 ** 2)))

        db = rocksdb.DB(db_path, opts, read_only=True)
        return db


        # rocksdb_entity_meta_info = setup_rocksdb(config.rocksdb_meta_info_path)
        #
        # def get_rocksdb_item_name(item_id):
        #     obj = rocksdb_get(rocksdb_entity_meta_info, item_id)
        #     if obj:
        #         return obj['ent_name'].replace(" ", "_")
        #     return "NIL"


def is_date(s):
    try:
        datetime.datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        try:
            datetime.datetime.strptime(s, "%Y/%m/%d")
            return True
        except ValueError:
            return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    if len(s) == 0:
        return False
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def is_float_but_not_int(s):
    if is_int(s):
        return False
    return is_float(s)


def rocksdb_get(db, key):
    val = db.get(bytes(key, 'utf-8'))
    if val:
        return json.loads(val)
    return None


def redis_get(r, key):
    response = r.get(key)
    if response:
        try:
            return json.loads(response)
        except UnicodeDecodeError:
            print(response)
    return None


def load_json_table(fn):
    with open(fn, mode="r", encoding="utf-8") as fp:
        tables = dict()
        for line in fp:
            words = line.strip().split("\t")
            tab = json.loads(words[1])
            # stringlize table
            str_tab = []
            for row in tab:
                str_tab.append([str(cell) for cell in row])
            tables[words[0]] = str_tab
        return tables


def get_item_name(item_id):
    response = entity_meta_info.get(item_id)
    if response:
        data = json.loads(response)
        return data[shortname.ENT_NAME].replace(" ", "_")
    return "NIL"

