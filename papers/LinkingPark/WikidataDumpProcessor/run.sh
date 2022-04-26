# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# step 1: parse raw wikidata dump
wikidata_dump_path='$BASE_DATA_DIR/wikidata'
dump_date='20200525'
mkdir ${wikidata_dump_path}\\wikidata-${dump_date}
python parse_bz_dump.py --bz_file ${wikidata_dump_path}\\wikidata-${dump_date}-all.json.bz2 --dump_dir ${wikidata_dump_path}\\wikidata-${dump_date}

# step 2: extract entity types from wikidata dump
python extract_ent_types.py --property_fn ${wikidata_dump_path}\\wikidata-${dump_date}\\properties.jsonl --re_fn ${wikidata_dump_path}\\wikidata-${dump_date}\\type_map_rank.pkl