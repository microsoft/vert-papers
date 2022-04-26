# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def load_alias_map(fn):
    with open(fn, mode="rb") as fp:
        alias_map = pickle.load(fp)
        return alias_map


def load_disambiguation_pages(fn):
    with open(fn, mode="rb") as fp:
        disambiguation_pages = pickle.load(fp)
        return disambiguation_pages


def load_wiki_id2wikidata_id(fn):
    with open(fn, mode="rb") as fp:
        unify_kb_map = pickle.load(fp)
        wiki_id2wikidata_id = dict()

        for wiki_id in unify_kb_map:
            if unify_kb_map[wiki_id]["wikidata_id"]:
                wiki_id2wikidata_id[str(wiki_id)] = unify_kb_map[wiki_id]["wikidata_id"]
        return wiki_id2wikidata_id


def load_wikidata_alias_map(fn):
    with open(fn, mode="rb") as fp:
        wikidata_alias_map = pickle.load(fp)
        return wikidata_alias_map


def merge(wikipedia_alias_map, wikidata_alias_map,
          wiki_id2wikidata_id, disambiguation_pages,
          out_fn, min_threshold=0.01, rm_disambiguation=True):
    filtered_wikipedia_alias_map = dict()
    for alias in wikipedia_alias_map:
        wikidata_ids = set()
        for c in wikipedia_alias_map[alias]:
            wiki_id = c[1]
            if wiki_id in wiki_id2wikidata_id and c[2] >= min_threshold:
                wikidata_id = wiki_id2wikidata_id[wiki_id]
                wikidata_ids.add(wikidata_id)
        if len(wikidata_ids) > 0:
            filtered_wikipedia_alias_map[alias] = wikidata_ids

    for alias in filtered_wikipedia_alias_map:
        if alias not in wikidata_alias_map:
            wikidata_alias_map[alias] = list(filtered_wikipedia_alias_map[alias])
        else:
            wikidata_alias_map[alias] = list(filtered_wikipedia_alias_map[alias]
                                             | set(wikidata_alias_map[alias]))

    if rm_disambiguation:
        result = dict()
        for alias in wikidata_alias_map:
            candid = wikidata_alias_map[alias]
            candid = [c for c in candid if c not in disambiguation_pages]
            if len(candid) > 0:
                result[alias] = candid
    else:
        result = wikidata_alias_map

    with open(out_fn, mode="wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alias_map_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/WikipediaData/p_e_m.pkl"
    )
    parser.add_argument(
        "--unify_wikipedia_kb_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/WikipediaData/unify_wikipedia_kb.pkl"
    )
    parser.add_argument(
        "--wikidata_alias_map_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata_alias_map.pkl"
    )
    parser.add_argument(
        "--disambiguation_page_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/disambiguation_pages.pkl"
    )
    parser.add_argument(
        "--alias_map_dir",
        type=str,
        default="$BASE_DATA_DIR/wikidata/merged_alias_map"
    )
    args = process_relative_path_config(parser.parse_args())
    wikipedia_alias_map = load_alias_map(args.alias_map_fn)
    wiki_id2wikidata_id = load_wiki_id2wikidata_id(args.unify_wikipedia_kb_fn)
    wikidata_alias_map = load_wikidata_alias_map(args.wikidata_alias_map_fn)
    disambiguation_pages = load_disambiguation_pages(args.disambiguation_page_fn)
    merge(wikipedia_alias_map, wikidata_alias_map,
          wiki_id2wikidata_id, disambiguation_pages,
          f"{args.alias_map_dir}/alias_map_rm_disambiguation.pkl", min_threshold=0.01, rm_disambiguation=True)

    merge(wikipedia_alias_map, wikidata_alias_map,
          wiki_id2wikidata_id, disambiguation_pages,
          f"{args.alias_map_dir}/alias_map_keep_disambiguation.pkl", min_threshold=0.01, rm_disambiguation=False)
