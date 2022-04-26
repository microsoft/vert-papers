# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from TableAnnotator.Maps.mixed_candid_map import MixedCandidateMap
import os
from Utils.utils import load_json_table
from tqdm import tqdm
import pickle
from prettytable import PrettyTable
from TableAnnotator.Config.config_utils import process_relative_path_config


def read_target_set(target_fn):
    target_set = set()
    with open(target_fn, encoding="utf-8", mode="r") as fp:
        for line in fp:
            line = line.strip().replace("\"", "")
            words = line.split(',')
            target_set.add(tuple(words))
    return target_set


def gen_candidates(tab_fn,
                   candid_gen,
                   out_fn,
                   topk=1000,
                   target_only=True,
                   elastic_search_only=False,
                   dictionary_only=False):
    tables = load_json_table(tab_fn)
    alias_table = dict()
    tab_num, col_num, row_num, cell_num, matched_cell_num = 0, 0, 0, 0, 0
    total_cells = 0
    for tab_id in tables:
        for row in tables[tab_id]:
            for cell in row:
                total_cells += 1
    print("total number of cells {}".format(total_cells))
    pbar = tqdm(total=total_cells)

    for tab_id in tables:
        tab_num += 1
        col_num += len(tables[tab_id][0])
        row_num += len(tables[tab_id])

        for row_id, row in enumerate(tables[tab_id]):
            for col_id, cell in enumerate(row):
                pbar.update(1)
                if (not ((tab_id, str(row_id+1), str(col_id)) in cea_targets)) and target_only:
                    continue
                cell_num += 1
                if cell in alias_table:
                    if len(alias_table[cell]) > 0:
                        matched_cell_num += 1
                    continue
                candid = candid_gen.gen_candidates_with_features(cell,
                                                                 elastic_search_only=elastic_search_only,
                                                                 dictionary_only=dictionary_only,
                                                                 topk=topk)
                alias_table[cell] = candid
                if len(candid) > 0:
                    matched_cell_num += 1
            #     break
        # break
    pbar.close()

    with open(out_fn, mode="wb") as fp:
        pickle.dump(alias_table, fp)

    results = PrettyTable()
    results.field_names = [
        "Table Num",
        "Column Num",
        "Row Num",
        "Cell Num",
        "Matched Cell Num"
    ]

    results.add_row([tab_num, col_num, row_num, cell_num, matched_cell_num])
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="$BASE_DATA_DIR/SemTab/Input")
    parser.add_argument("--candid_dir", type=str, default="$BASE_DATA_DIR/SemTab/CandidateGeneration/")
    parser.add_argument("--benchmark_dir", type=str, default="$BASE_DATA_DIR/SemTab/Benchmark/SemTab2020_Data/SemTab2020_Table_GT_Target/")
    parser.add_argument("--method", type=str, default="_strict_match_first_rm_dis_rm_cell_filter")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--target_only", type=bool, default=False)
    parser.add_argument("--index_name",
                        type=str,
                        default="wikidata_keep_disambiguation")  # choices: wikidata_keep_disambiguation or wikidata_rm_disambiguation
    parser.add_argument("--in_links_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/incoming_links/in_coming_links_num.pkl")
    parser.add_argument("--alias_map_fn",
                        type=str,
                        default="$BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_keep_disambiguation.pkl")  # choices: alias_map_keep_disambiguation.pkl alias_map_rm_disambiguation.pkl
    parser.add_argument("--topk",
                        type=int,
                        default=1000)
    parser.add_argument("--output_dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/CandidateGeneration")
    parser.add_argument("--elastic_search_only",
                        type=bool,
                        default=False)
    parser.add_argument("--dictionary_only",
                        type=bool,
                        default=False)
    args = process_relative_path_config(parser.parse_args())

    # create Data folders if not exists
    output_dir = os.path.join(args.output_dir, "Round{}".format(args.round))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cur_dir = os.path.join(output_dir, "all_in_one")
    print('Dump to => {}'.format(cur_dir))
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)

    out_fn = os.path.join(cur_dir, f"alias_map{args.method}.pkl")

    params = {
        "alias_map_fn": args.alias_map_fn,
        "index_name": args.index_name,
        "in_links_fn": args.in_links_fn
    }
    candid_gen = MixedCandidateMap(params)
    cea_targets = read_target_set(os.path.join(args.benchmark_dir,
                                               "Round{}".format(args.round),
                                               "CEA_Round{}_Targets.csv".format(args.round)))
    gen_candidates(os.path.join(args.data_dir, "round{}_table.json".format(args.round)),
                   candid_gen,
                   out_fn,
                   topk=args.topk,
                   target_only=args.target_only,
                   elastic_search_only=args.elastic_search_only,
                   dictionary_only=args.dictionary_only)
