# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from Utils.utils_data import read_cea_target, read_cta_target, read_cpa_target,\
    write_CEA_result, write_CTA_result, write_CPA_result, load_cache_result
from TableAnnotator.Config.config_utils import process_relative_path_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Benchmark/SemTab2020_Data/SemTab2020_Table_GT_Target")
    parser.add_argument("--log_fn",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Output/round4_token_char_cand_500_alpha_0.2_beta_0.5_static_type_property/all_output_tables.jsonl")
    parser.add_argument("--round",
                        type=int,
                        default=4)
    parser.add_argument("--result_dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/Submission/")
    args = process_relative_path_config(parser.parse_args())

    log_fn = os.path.split(os.path.split(args.log_fn)[0])[1]
    result_dir = os.path.join(args.result_dir, log_fn)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(os.path.join(result_dir, "CEA")):
        os.mkdir(os.path.join(result_dir, "CEA"))

    if not os.path.exists(os.path.join(result_dir, "CTA")):
        os.mkdir(os.path.join(result_dir, "CTA"))

    if not os.path.exists(os.path.join(result_dir, "CPA")):
        os.mkdir(os.path.join(result_dir, "CPA"))

    # read CEA targets and CTA targets
    cea_targets = read_cea_target(os.path.join(args.data_dir, f"Round{args.round}", f"CEA_Round{args.round}_Targets.csv"))
    cta_targets = read_cta_target(os.path.join(args.data_dir, f"Round{args.round}", f"CTA_Round{args.round}_Targets.csv"))
    cpa_targets = read_cpa_target(os.path.join(args.data_dir, f"Round{args.round}", f"CPA_Round{args.round}_Targets.csv"))

    col_entities, col_types, col_properties = load_cache_result(args.log_fn)
    # write results
    write_CEA_result(os.path.join(result_dir, 'CEA', "CEA.csv"),
                     col_entities, cea_targets)
    write_CTA_result(os.path.join(result_dir, 'CTA', "CTA.csv"),
                     col_types, cta_targets)
    write_CPA_result(os.path.join(result_dir, 'CPA', "CPA.csv"),
                     col_properties, cpa_targets)