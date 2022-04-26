# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

'''

In this script, we define some global config parameters to ease transferring between different machines

'''

base_data_dir = os.environ.get("BASE_DATA_DIR")
semtab_benchmark_dir = f"{base_data_dir}/SemTab/Benchmark/SemTab2020_Data/SemTab2020_Table_GT_Target"
tough_table_benchmark_dir = f"{base_data_dir}/SemTab/Benchmark/2T_WD/2T_WD"
# KB Store
# RocksDB
rocksdb_meta_info_path = f"{base_data_dir}/RocksDB/Compress/EntityMetaInfo/data.db"
rocksdb_property_val_info_path = f"{base_data_dir}/RocksDB/Compress/PropertyValueInfo/data.db"
# Redis
ent_meta_redis_host, ent_meta_redis_port = "localhost", 6390
# property_redis_host, property_redis_port = "localhost", 6391
property_redis_host, property_redis_port = "localhost", 6400
ent_label_redis_host, ent_label_redis_port = "localhost", 6382
# alias_table_rm_dis_redis_host, alias_table_rm_dis_redis_port = "localhost", 6393
# alias_table_keep_dis_redis_host, alias_table_keep_dis_redis_port = "localhost", 6392
# rocksdb_alias_table_rm_dis_path = f"{base_data_dir}/RocksDB/AliasMap/data_rm_disambiguation.db"
# rocksdb_alias_table_keep_dis_path = f"{base_data_dir}/RocksDB/AliasMap/data_keep_disambiguation.db"
use_rocks_db = False
