# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

# Common configs

# Candidate generation
allow_int = True                # True -> Keep integer cells as entity cells
keep_cell_filtering = False     # True -> Filter float , data time, int cells

DEFAULT_PORT = '6009'
DEFAULT_DATA_DIR = os.environ.get("BASE_DATA_DIR")
DEFAULT_SYSTEM_DIR = os.environ.get("BASE_DATA_DIR")

base_data_dir = DEFAULT_DATA_DIR
base_system_dir = DEFAULT_SYSTEM_DIR
kb_dir = os.path.join(base_data_dir, 'SemTab/CandidateGeneration')
schema_entity_meta_fn = os.path.join(base_data_dir, 'wikidata/schema_meta_to_store_compress.jsonl')

debug = False

kb_store = "redis"  # other choices: "RAM" or "rocksdb"
# "RAM" means directly loading the jsonl files into RAM, property_value_fn and entity_meta_fn should be properly set.
# kb_store = "RAM" # other choices: "RAM" or "rocksdb"
# kb_store = "rocksdb" # other choices: "RAM" or "rocksdb"

# alias_map_store = "RAM" # other choices: "RAM" or "redis" or "rocksdb"

candid_gen_method = "online" # other choices: "offline" for quick experiments
# candid_gen_method = "offline"  # other choices: "offline" for quick experiments

# Type inference parameters

k_level = 2
type_count_fn = os.path.join(base_data_dir, "wikidata/wikidata-20200525/type_count.pkl")

# Fuzzy property linking resources
property_entity_types_fn = os.path.join(
    base_data_dir,
    "wikidata/alp_property/files_needed_for_prop_linking/entity_types.json"
)

type_property_stats_fn = os.path.join(
    base_data_dir,
    "wikidata/alp_property/files_needed_for_prop_linking/type_property_stats.json"
)

qnumbers_to_units_fn = os.path.join(
    base_data_dir,
    "wikidata/alp_property/files_needed_for_prop_linking/qnumbers_to_units.json"
)

unit_to_standard_unit_factors_fn = os.path.join(
    base_data_dir,
    "wikidata/alp_property/files_needed_for_prop_linking/unit_to_standard_unit_factors.json"
)

# Online configs
port = DEFAULT_PORT
log_fn = os.path.join(base_data_dir, "Log", "logs", "logs.txt")

# Experiment configs
# Evaluation Round
data_round = 4

# Used when candid_gen_method = "offline"
# DS + FS
candid = "alias_map_ds_es"
# DS only
# candid = "alias_map_ds_only"
candid_map_fn = os.path.join(kb_dir, f"Round{data_round}", f"all_in_one/{candid}.pkl")

# Used when kb_store = "RAM"
property_value_fn = os.path.join(kb_dir, f"Round{data_round}", f"{candid}_entity_set", "property_value_compress.jsonl")
entity_meta_fn = os.path.join(kb_dir, f"Round{data_round}", f"{candid}_entity_set", "meta_info_compress.jsonl")

# property_value_fn = os.path.join(base_dir, 'wikidata/properties_converted_to_store_compress.jsonl')
# entity_meta_fn = os.path.join(base_dir, 'wikidata/meta_to_store_compress.jsonl')
