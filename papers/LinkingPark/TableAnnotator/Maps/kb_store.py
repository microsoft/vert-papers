# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import redis
from GlobalConfig import global_config
from Utils.utils import redis_get

if global_config.use_rocks_db:
    from Utils.utils import setup_rocksdb, rocksdb_get


class KBStore(object):
    def __init__(self, params):
        self.kb_store = params['kb_store']
        if params['kb_store'] == 'redis':
            self.info = redis.Redis(host=params["host"], port=params["port"])
        elif params['kb_store'] == 'rocksdb':
            assert global_config.use_rocks_db is True
            self.info = setup_rocksdb(params["db_path"])
        else:
            raise ValueError(f"unsupported kb_store {self.kb_store}")

    def get_obj(self, item_id):
        if self.kb_store == 'redis':
            return redis_get(self.info, item_id)
        else:
            return rocksdb_get(self.info, item_id)
