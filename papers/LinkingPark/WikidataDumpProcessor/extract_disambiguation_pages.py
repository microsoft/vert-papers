# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_pkl(fn):
    with open(fn, mode="rb") as fp:
        pkl = pickle.load(fp)
        return pkl


def filter_disambiguation_pages(ent_types, re_fn):
    disambiguation_set = set()
    for ent in tqdm(ent_types):
        _types = set(ent_types[ent]["types"])
        if len(_types & {"Q4167410", "Q1151870"}) > 0:
            disambiguation_set.add(ent)

    print("number of disambiguation pages: {}".format(len(disambiguation_set)))
    with open(re_fn, mode="wb") as fp:
        pickle.dump(disambiguation_set, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_map_rank_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/type_map_rank.pkl"
    )
    parser.add_argument(
        "--out_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/disambiguation_pages.pkl"
    )
    args = process_relative_path_config(parser.parse_args())
    ent_types = load_pkl(args.type_map_rank_fn)
    filter_disambiguation_pages(ent_types,
                                args.out_fn)
