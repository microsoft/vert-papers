import os
import sys
import argparse
import re
import numpy as np

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--src_dir", default="result/", type=str, help="src model dir.")
parser.add_argument("--trans_dir", default="result/", type=str, help="trans model dir.")
parser.add_argument("--finetune_dir", default="result/", type=str, help="finetune dir.")
parser.add_argument(
    "--unitrans_src_dir", default="result/", type=str, help="unitrans src dir."
)
parser.add_argument(
    "--unitrans_finetune_dir",
    default="result/",
    type=str,
    help="unitrans finetune dir.",
)
parser.add_argument("--seeds", default="122,649,705,854,975", type=str, help="seeds")

args = parser.parse_args()

ITEMS = ["src", "trans", "finetune", "UniTrans_src", "UniTrans_finetune"]
PRED_MODS = ["dev", "test"]


def read_data(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [ii.replace("\n", "") for ii in f.readlines()]


def mkdir(origin_dir: str):
    """ mkdir file dir"""
    if not os.path.exists(origin_dir):
        os.mkdir(origin_dir)


def load_result_once(log_path: str):
    result = read_data(log_path)
    result_str = " ".join(result)
    fpr = re.findall(
        "f1 = (\d{1,3}\.\d{0,10}).*precision = (\d{1,3}\.\d{0,10}).*recall = (\d{1,3}\.\d{0,10})",
        result_str,
    )
    assert len(fpr) == 1, fpr
    f, p, r = fpr[0]
    return [float(ii) for ii in [p, r, f]]


def load_result(model_dir: str):
    log_dir = os.path.join(model_dir, "logs")
    test_p, dev_p = os.listdir(log_dir)[-2:]
    t_fpr = load_result_once(os.path.join(log_dir, test_p))
    d_fpr = load_result_once(os.path.join(log_dir, dev_p))
    return t_fpr, d_fpr


def statistical_pipeline():
    result = {ii: {jj: [] for jj in PRED_MODS} for ii in ITEMS}
    for seed in args.seeds.split(","):
        dirs = [
            args.src_dir,
            args.trans_dir,
            args.finetune_dir,
            args.unitrans_src_dir,
            args.unitrans_finetune_dir,
        ]
        dirs = [ii.replace("122", seed) for ii in dirs]
        for item, o_dir in zip(ITEMS, dirs):
            t_fpr, d_fpr = load_result(o_dir)
            result[item]["dev"].append(d_fpr)
            result[item]["test"].append(t_fpr)

    for item in ITEMS:
        for mod in PRED_MODS:
            res = np.array(result[item][mod])
            mean = np.mean(res, axis=0)
            mean[-1] = 2 / (1 / mean[0] + 1 / mean[1])
            result[item][mod] = [*res, mean]
    print(result)
    log_str = [
        [
            "{},{},{},,".format(*result[item][mod][ii])
            for item in ITEMS
            for mod in PRED_MODS
        ]
        for ii in range(len(result["src"]["dev"]))
    ]
    log_str = [
        ",,mBERT Dev,,,,mBERT Test,,,,Translate Dev,,,,Translate Test,,,,Fine Trans Dev,,,,Fine Trans Test,,,,UniTrans-mBERT Dev,,,,UniTrans-mBERT Test,,,,UniTrans-Trans Dev,,,,UniTrans-Trans Test"
    ] + ["".join(ii) for ii in log_str]
    mkdir(os.path.join(args.unitrans_src_dir.split("/")[0].split("\\")[0], "result"))

    with open(args.unitrans_src_dir.replace("-122", "") + ".csv", "w") as f:
        f.write("\n".join(log_str))


if __name__ == "__main__":
    statistical_pipeline()
