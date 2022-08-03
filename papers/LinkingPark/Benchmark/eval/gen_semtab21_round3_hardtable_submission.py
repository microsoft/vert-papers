import argparse
import os
from Utils.utils_data import read_cea_target, read_cta_target, read_cpa_target,\
    write_CEA_result, write_CTA_result, write_CPA_result, load_cache_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        type=str,
                        required=True)
    parser.add_argument("--log_fn",
                        type=str,
                        required=True)
    parser.add_argument("--result_dir",
                        type=str,
                        required=True)
    args = parser.parse_args()

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
    cea_targets = read_cea_target(os.path.join(args.data_dir,
                                               "HardTablesR3_CEA_WD_Round3_Targets.csv"))
    cta_targets = read_cta_target(os.path.join(args.data_dir,
                                               "HardTablesR3_CTA_WD_Round3_Targets.csv"))
    cpa_targets = read_cpa_target(os.path.join(args.data_dir,
                                               "HardTablesR3_CPA_WD_Round3_Targets.csv"))

    col_entities, col_types, col_properties = load_cache_result(args.log_fn)
    # write results
    write_CEA_result(os.path.join(result_dir, 'CEA', "CEA.csv"),
                     col_entities, cea_targets, wikidata_prefix="http://www.wikidata.org/entity/")
    write_CTA_result(os.path.join(result_dir, 'CTA', "CTA.csv"),
                     col_types, cta_targets, wikidata_prefix="http://www.wikidata.org/entity/")
    write_CPA_result(os.path.join(result_dir, 'CPA', "CPA.csv"),
                     col_properties, cpa_targets, property_prefix="http://www.wikidata.org/prop/direct/")
