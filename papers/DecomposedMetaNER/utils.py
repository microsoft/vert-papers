# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random

import numpy as np
import torch


def load_file(path: str, mode: str = "list-strip"):
    if not os.path.exists(path):
        return [] if not mode else ""
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        if mode == "list-strip":
            data = [ii.strip() for ii in f.readlines()]
        elif mode == "str":
            data = f.read()
        elif mode == "list":
            data = list(f.readlines())
        elif mode == "json":
            data = json.loads(f.read())
        elif mode == "json-list":
            data = [json.loads(ii) for ii in f.readlines()]
    return data


def set_seed(seed, gpu_device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_device > -1:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
