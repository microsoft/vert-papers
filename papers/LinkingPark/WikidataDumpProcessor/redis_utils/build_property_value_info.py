# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import redis
import json
from tqdm import tqdm
import time
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def build_db(r, in_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        num = 0
        for line in tqdm(fp):
            entity = json.loads(line.strip())
            r.set(entity['id'],
                  json.dumps(entity))
            num += 1
            if num % 100000 == 0:
                time.sleep(30)
                print("processing {} lines".format(num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/properties_converted_to_store_compress.jsonl"
    )
    args = process_relative_path_config(parser.parse_args())
    r = redis.Redis(host='localhost', port=6400, db=0)
    build_db(r, args.in_fn)
    r.close()
