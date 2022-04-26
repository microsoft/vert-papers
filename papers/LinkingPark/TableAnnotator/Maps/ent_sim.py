# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from TableAnnotator.Maps.entity_property_info import KBPropertyVal


class SparseEntitySim(object):
    def __init__(self, kb_feature: KBPropertyVal):
        self.kb_feature = kb_feature
        self.sim_cache = dict()

    def cosine_sim(self, e1, e2):
        if f"{e1}@@{e2}" in self.sim_cache:
            return self.sim_cache[f"{e1}@@{e2}"]
        f1_labels = self.kb_feature.query_kb_feature(e1)
        f2_labels = self.kb_feature.query_kb_feature(e2)

        f1_f2_labels = f1_labels & f2_labels
        if len(f1_f2_labels) == 0:
            return 0.0
        v1 = len(f1_f2_labels)
        e1_len = max(math.sqrt(len(f1_labels)), 1e-6)
        e2_len = max(math.sqrt(len(f2_labels)), 1e-6)
        sim = v1 / (e1_len * e2_len)
        self.sim_cache[f"{e1}@@{e2}"] = sim
        return sim

