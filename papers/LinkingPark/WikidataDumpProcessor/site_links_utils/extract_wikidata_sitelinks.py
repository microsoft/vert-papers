# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
Parse wikidata raw json dump file
1. extract site links to acquire multi-language Wikipedia title to Wikidata mapping
'''

import json
import bz2
import sys
import os
import argparse
from tqdm import tqdm
from TableAnnotator.Config.config_utils import process_relative_path_config


def process_one_line_all_properties(json_obj):
    ent_id = json_obj['id']
    t = json_obj["type"]
    # en_label
    if "en" in json_obj["labels"] and "value" in json_obj["labels"]["en"]:
        en_label = json_obj['labels']['en']['value']
    else:
        en_label = "No_label_defined"

    # multi-language labels
    labels = dict()
    for lang_code in json_obj["labels"]:
        labels[lang_code] = json_obj["labels"][lang_code]['value']

    # multi-language aliases
    aliases = set()
    for lang_code in json_obj["aliases"]:
        for alias in json_obj["aliases"][lang_code]:
            aliases.add(alias['value'])

    # multi-language descriptions
    descriptions = dict()
    for lang_code in json_obj["descriptions"]:
        descriptions[lang_code] = json_obj["descriptions"][lang_code]['value']

    properties = list(json_obj["claims"].keys())
    property_values = {}
    for prop in json_obj["claims"]:
        values = []
        for snak in json_obj["claims"][prop]:
            mainsnak = snak["mainsnak"]
            datatype = mainsnak["datatype"]
            snaketype = mainsnak["snaktype"]
            if snaketype == "value":
                values.append({
                    "datatype": datatype,
                    "datavalue": mainsnak["datavalue"]
                })
        if values:
            property_values[prop] = values

    output = {
        "id": ent_id,
        "type": t,
        "en_label": en_label,
        "properties": properties,
        "property_values": property_values
    }

    ent_meta_info = {
        "id": ent_id,
        "labels": labels,
        "aliases": list(aliases),
        "descriptions": descriptions
    }

    return json.dumps(output), json.dumps(ent_meta_info)


def process_one_file(total_lines, in_fn, dump_dir):
    with bz2.open(in_fn, "rt") as in_fp:
        with open(os.path.join(dump_dir, "properties.jsonl"),
                  encoding="utf-8", mode="w") as property_out_fp:
            with open(os.path.join(dump_dir, "meta.jsonl"),
                      encoding="utf-8", mode="w") as meta_out_fp:
                with tqdm(total=total_lines) as pbar:
                    for line in in_fp:
                        pbar.update()
                        try:
                            json_obj = json.loads(line.strip().strip(','))
                            property_values, meta_info = process_one_line_all_properties(json_obj)

                            property_out_fp.write("{}\n".format(property_values))
                            meta_out_fp.write("{}\n".format(meta_info))

                        except Exception as e:
                            line = line.strip().strip(',')
                            print(line)

                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            print("Exception:", exc_type, "- line", exc_tb.tb_lineno)
                            if len(line) < 30:
                                print("Failed line:", line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz_file",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/WikidataDumps/wikidata-20200525-all.json.bz2",
                        help="bz file path")
    parser.add_argument("--dump_dir",
                        type=str,
                        default="$BASE_DATA_DIR/SemTab/WikidataDumps/tmp",
                        help="dump file path")
    args = process_relative_path_config(parser.parse_args())
    print("=============================================")
    print("==> start parsing wikidata bz file: {}...".format(args.bz_file))
    print("===> output dir: {}".format(args.dump_dir))
    num_lines = 87078954
    process_one_file(num_lines,
                     args.bz_file,
                     args.dump_dir)