# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from CandidGen.wikidata_candid_map_typo import WikidataCandidateTypoMap
from CandidGen.elastic_search import ElasticSearcher
import Levenshtein
import pickle
import redis
from GlobalConfig import global_config
from TableAnnotator.Util.utils import check_cell_filtering
import json

ent_label_redis = redis.Redis(host=global_config.ent_label_redis_host,
                              port=global_config.ent_label_redis_port,
                              db=0)


def get_ent_labels(ent):
    response = ent_label_redis.get(ent)
    if response:
        return json.loads(response)
    return {}


def get_max_label_and_score(mention, ent):
    labels = get_ent_labels(ent)
    if len(labels) == 0:
        return "No_label_defined", 0.0
    max_score = 0.0
    max_label = "No_label_defined"
    for lang_code in labels:
        score = Levenshtein.ratio(mention, labels[lang_code])
        if score > max_score:
            max_score = score
            max_label = labels[lang_code]
    return max_label, max_score


class MixedCandidateMap(object):
    def __init__(self, params):
        self.params = params
        self.dict_search = WikidataCandidateTypoMap(params["alias_map_fn"])
        self.elastic_search = ElasticSearcher(index_name=params["index_name"])

        self.in_links = self.load_pkl(params["in_links_fn"])
        self.cache = dict()

    def load_pkl(self, fn):
        with open(fn, mode="rb") as fp:
            pkl = pickle.load(fp)
            return pkl

    def shortlist(self, mention, cands):
        cands = [(x[0], x[2]) for x in cands]
        set1 = cands[:30]
        cands.sort(key=lambda x: Levenshtein.distance(x[1], mention))
        set2 = cands[:30]
        new_cands = []
        for x in (set1 + set2):
            new_cands += x[0].split(";")
        return new_cands

    def gen_candidates(self,
                       mention,
                       elastic_search_only=False,
                       dictionary_only=False):
        assert not (elastic_search_only and dictionary_only), "elastic_search_only and dictionary_search_only can not be both true"
        if check_cell_filtering(mention):
            return []

        if dictionary_only:
            candid = self.dict_search.gen_candid_spell_corrector(mention,
                                                                 strict_match_first=True,
                                                                 min_edit_len=10)
            candid = [x[0] for x in candid]
            return list(set(candid))
        elif elastic_search_only:
            candid = self.elastic_search.query_data(mention, top_k=50, score_func="token_char")
            candid = self.shortlist(mention, candid)
            return list(set(candid))
        else:
            candid = self.dict_search.gen_candid_spell_corrector(mention,
                                                                 strict_match_first=True,
                                                                 min_edit_len=10)
            candid = [x[0] for x in candid]
            if len(candid) > 0:
                return list(set(candid))
            candid = self.elastic_search.query_data(mention, top_k=50, score_func="token_char")
            candid = self.shortlist(mention, candid)
            return list(set(candid))

    def add_feature(self, mention, candid):
        result = []
        for ent in candid:
            max_label, max_score = get_max_label_and_score(mention, ent)
            in_links_count = self.in_links.get(ent, 1)
            result.append((ent, max_score, in_links_count))
        total_count = sum([x[2] for x in result])
        candid_list = [(x[0], x[1], x[2] / total_count) for x in result]
        sorted_candid_list = sorted(candid_list, key=lambda x: x[2], reverse=True)
        return sorted_candid_list

    def gen_candidates_with_features(self,
                                     mention,
                                     elastic_search_only=False,
                                     dictionary_only=False,
                                     topk=100):
        if mention in self.cache:
            return self.cache[mention]
        candid = self.gen_candidates(mention,
                                     elastic_search_only=elastic_search_only,
                                     dictionary_only=dictionary_only)
        result = self.add_feature(mention, candid)[:topk]
        self.cache[mention] = result
        return result
