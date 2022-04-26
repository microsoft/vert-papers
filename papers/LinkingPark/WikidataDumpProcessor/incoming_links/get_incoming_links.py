# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pickle
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def get_incoming_links(in_fn, out_fn):
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        in_links = dict()
        for line in tqdm(fp):
            node = json.loads(line.strip())
            for p in node['property_values']:
                for v in node['property_values'][p]:
                    if v['dtype'] == 'wikibase-entityid':
                        dest_node = v['value']
                        if dest_node not in in_links:
                            in_links[dest_node] = 1
                        else:
                            in_links[dest_node] += 1
        with open(out_fn, mode="wb") as fp:
            pickle.dump(in_links, fp)
        print(len(in_links), 'items is linked.')
        print('over')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikidata_dir", type=str, default="$BASE_DATA_DIR/wikidata")
    args = process_relative_path_config(parser.parse_args())
    get_incoming_links(f"{args.wikidata_dir}/properties_converted.jsonl",
                       f"{args.wikidata_dir}/incoming_links/in_coming_links_num.pkl")
