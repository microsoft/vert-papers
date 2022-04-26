# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
from tqdm import tqdm
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


def parse_one_file(fn, out_fn):

    with open(fn, encoding="utf-8", mode="r") as fp:
        with open(out_fn, encoding="utf-8", mode="w") as re_fp:
            for line in tqdm(fp):
                json_obj = json.loads(line.strip())

                wikidata = {
                    "id": json_obj["id"]
                }

                wikidata["type"] = json_obj["type"]
                wikidata["properties"] = json_obj["properties"]
                wikidata["property_values"] = {}

                for property in json_obj["property_values"]:
                    wikidata["property_values"][property] = []
                    for value in json_obj["property_values"][property]:
                        datavalue = value["datavalue"]
                        datatype = datavalue["type"]
                        value = datavalue["value"]
                        obj = {"dtype": datatype}
                        try:
                            if datatype == "string":
                                obj["value"] = value.strip()
                            elif datatype == "quantity":
                                # TODO why use regular expression to convert amount?
                                # amount = float(re.findall("[-]?[0-9]*\.?[0-9]+", value["amount"])[0])
                                amount = float(value["amount"])
                                try:
                                    unit = re.findall('Q\w+', value["unit"])[0]
                                # No units specified in wikidata
                                except IndexError:
                                    unit = 1
                                obj["value"] = amount
                                obj["unit"] = unit
                                if "lowerBound" in value:
                                    obj["lowerBound"] = float(value["lowerBound"])
                                    # print(datavalue)
                                if "upperBound" in value:
                                    obj["upperBound"] = float(value["upperBound"])
                                    # print(datavalue)
                            elif datatype == "time":
                                obj["value"] = value["time"]
                            elif datatype == "monolingualtext":
                                obj["value"] = value["text"]
                            elif datatype == "wikibase-entityid":
                                obj["value"] = value["id"]
                            else:
                                obj["value"] = value
                            if "value" in obj:
                                wikidata["property_values"][property].append(obj)
                            else:
                                print(datavalue)
                        except:
                            print(line)
                re_fp.write("{}\n".format(json.dumps(wikidata)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/properties.jsonl"
    )
    parser.add_argument(
        "--out_fn",
        type=str,
        default="$BASE_DATA_DIR/wikidata/wikidata-20200525/properties_converted.jsonl"
    )
    args = process_relative_path_config(parser.parse_args())
    parse_one_file(args.fn, args.out_fn)
    print("done.")