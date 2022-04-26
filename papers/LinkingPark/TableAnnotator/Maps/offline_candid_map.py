# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from TableAnnotator.Util.utils import check_cell_filtering


class OfflineCandidateMap(object):
    def __init__(self, candid_map_fn):
        # do candidate generation using offline prepared candidate map
        self.mention_cache = OfflineCandidateMap.load_candid_map(candid_map_fn)

    @staticmethod
    def load_candid_map(fn):
        with open(fn, mode="rb") as fp:
            mention_cache = pickle.load(fp)
            return mention_cache

    def gen_candidates_with_features(self, cell_text, topk=10):
        if check_cell_filtering(cell_text):
            return []
        if cell_text in self.mention_cache:
            return self.mention_cache[cell_text][:topk]
        return []
