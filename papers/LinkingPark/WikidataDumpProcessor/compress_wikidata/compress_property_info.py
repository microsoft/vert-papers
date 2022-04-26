# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


datatype_code = {
    "string": "S",
    "wikibase-entityid": "E",
    "quantity": "Q",
    "globecoordinate": "G",
    "monolingualtext": "M",
    "time": "T"
}

def compress(in_obj):
    out_obj = {
        "id": in_obj["id"],
        "P": in_obj["properties"],
        "T": in_obj["types"],
        "PV": {
        }
    }
    for p in in_obj["property_values"]:
        out_obj["PV"][p] = []
        for x in in_obj["property_values"][p]:
            if "unit" in x:
                out_obj["PV"][p].append(
                    {
                        "d": datatype_code[x['dtype']],
                        "v": x['value'],
                        "unit": x['unit']
                    }
                )
            else:
                out_obj["PV"][p].append(
                    {
                        "d": datatype_code[x['dtype']],
                        "v": x['value'],
                    }
                )
    return out_obj


def process_file(in_fn, out_fn):
    qnum = 0
    pnum = 0
    total = 0
    with open(in_fn, mode="r", encoding="utf-8") as fp:
        with open(out_fn, mode="w", encoding="utf-8") as re_fp:
            for line in tqdm(fp):
                total += 1
                in_obj = json.loads(line.strip())
                out_obj = compress(in_obj)
                if out_obj["id"][0] == 'Q':
                    qnum += 1
                if out_obj["id"][0] == 'P':
                    pnum += 1
                re_fp.write("{}\n".format(json.dumps(out_obj)))
            print("Q Num: {}".format(qnum))
            print("P Num: {}".format(pnum))
            print("Total Num: {}".format(total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_fn", type=str, default="$BASE_DATA_DIR/wikidata/properties_converted_to_store.jsonl")
    parser.add_argument("--out_fn", type=str, default="$BASE_DATA_DIR/wikidata/properties_converted_to_store_compress.jsonl")
    args = process_relative_path_config(parser.parse_args())

    for i in range(1, 5):
        print(f"processing round {i}")
        in_fn = f"$BASE_DATA_DIR/SemTab/CandidateGeneration/Round{i}/token_char_ent_label_inlinks_sort_entity_set/property_value.jsonl"
        out_fn = f"$BASE_DATA_DIR/SemTab/CandidateGeneration/Round{i}/token_char_ent_label_inlinks_sort_entity_set/property_value_compress.jsonl"
        process_file(in_fn, out_fn)

        in_fn = f"$BASE_DATA_DIR/SemTab/CandidateGeneration/Round{i}/token_ent_label_inlinks_sort_entity_set/property_value.jsonl"
        out_fn = f"$BASE_DATA_DIR/SemTab/CandidateGeneration/Round{i}/token_ent_label_inlinks_sort_entity_set/property_value_compress.jsonl"
        process_file(in_fn, out_fn)
