# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import redis
import json
from tqdm import tqdm
import time
import pickle
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_pkl(fn):
    with open(fn, mode="rb") as fp:
        pkl = pickle.load(fp)
        return pkl


def build_db(r, label_map):
    num = 0
    for qid in tqdm(label_map):
        r.set(qid,
              json.dumps(label_map[qid]))
        num += 1
        if num % 1000000 == 0:
            time.sleep(30)
            print("processing {} lines".format(num))
    # db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alias_map_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_rm_disambiguation.pkl"
    )
    args = process_relative_path_config(parser.parse_args())
    label_map = load_pkl(args.alias_map_fn)
    print('loaded.')
    r = redis.Redis(host='localhost', port=6393, db=0)
    build_db(r, label_map)
    r.close()
