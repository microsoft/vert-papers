# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Any
from copy import deepcopy
from TableAnnotator.Config import config
from KBMapping.mappings import KBMapper
from Utils.utils import is_date, is_int, is_float, is_float_but_not_int


class Util:
    @staticmethod
    def add_vocab(vocab, key):
        if not key in vocab:
            vocab[key] = 1
        else:
            vocab[key] += 1


class InputTable(object):
    def __init__(self, tab: List[List[str]], tab_id: str):
        """

        :param tab: two-dim list [[row1], [row2], [row3]]
        """
        self.tab = tab
        self.tab_id = tab_id
        self.col_size = len(self.tab[0])
        self.row_size = len(self.tab)
        for i in range(self.row_size):
            for j in range(self.col_size):
                self.tab[i][j] = str(self.tab[i][j])

    def index_one_cell(self, i: int, j: int):
        return self.tab[i][j]

    def index_one_row(self, i: int):
        return self.tab[i]

    def index_one_col(self, j: int):
        return [self.tab[i][j] for i in range(self.row_size)]

    def get_main_column_idx(self):
        for j in range(self.col_size):
            col = self.index_one_col(j)
            is_lexical = [is_int(cell) or is_float(cell) or is_date(cell) for cell in col]
            if all(is_lexical):
                continue
            return j
        return 0


class OutputTable(object):
    def __init__(self, tab: InputTable, map_ids: bool = False, mapper: KBMapper = None):

        """
        :param tab: Input table
        :param map_ids: Map IDs to Satori IDs
        :param mapper: Mapper class
        """
        self.tab = tab
        self.mentions = dict()
        self.candid_list = dict()
        self.candid_list_before_shortlist = dict()
        self.map_ids = map_ids
        self.mapper = mapper

    def short_list_entities(self, candid: List[Tuple[str, float, float]], keep_N: int = 5):
        return candid[:keep_N]

    def remove_low_prior_entities(self,
                                  candid: List[Tuple[str, float, float]],
                                  min_p: float = 0.01) -> List[Tuple[str, float, float]]:
        ret_candid = []
        for c in candid:
            if c[-1] >= min_p:
                ret_candid.append(c)
        return ret_candid

    def gen_candidates(self, candid_map, topk, keep_N, lower_cell=False):
        # candidate generation
        for i in range(self.tab.row_size):
            for j in range(self.tab.col_size):
                cell = self.tab.index_one_cell(i, j)
                if lower_cell:
                    cell = cell.lower()
                raw_candid = candid_map.gen_candidates_with_features(cell, topk=topk)
                candid = self.short_list_entities(raw_candid, keep_N)
                self.mentions[(i, j)] = cell
                self.candid_list[(i, j)] = candid
                self.candid_list_before_shortlist[(i, j)] = raw_candid

    def gen_ent_set(self):
        ent_set = set()
        for cell_id in self.candid_list_before_shortlist:
            for x in self.candid_list_before_shortlist[cell_id]:
                ent_set.add(x[0])
        return ent_set

    def resort_final_prior(self,
                           min_final_diff=0.01):
        # resort_prior if top2 diff < min_final_diff
        for i in range(self.tab.row_size):
            for j in range(self.tab.col_size):
                candid_info = self.candid_entities_info[(i, j)]
                if len(candid_info) >= 2:
                    top2diff = candid_info[0]['final'] - candid_info[1]['final']
                    if top2diff <= min_final_diff:
                        if candid_info[0]["popularity"] < candid_info[1]['popularity']:
                            self.predict_entities[(i, j)] = candid_info[1]

    def init_pred(self,
                  alpha: float,
                  beta: float,
                  gamma: float,
                  property_feature_cache: Dict[Tuple[int, int, str], Any],
                  init_prune_topk=100):
        """

        :param alpha: col_ctx_score * alpha
        :param beta: row_ctx_score * beta
        :param gamma: popularity * gamma
        :property_feature_cache: Dict[Tuple[int, int, str], Any], i-th row, j-th col, c candidate
        :return:
        """
        # init prediction by edit distance
        self.predict_entities = dict()
        self.candid_entities_info = dict()

        for i in range(self.tab.row_size):
            for j in range(self.tab.col_size):
                if self.candid_list[(i, j)] == []:
                    self.predict_entities[(i, j)] = {"entity": "NIL",
                                                     "str_sim": 0.0,
                                                     "popularity": 0.0,
                                                     "col_ctx_score": 0.0,
                                                     "row_ctx_score": 0.0,
                                                     "final": 0.0}
                    self.candid_entities_info[(i, j)] = [{"entity": "NIL",
                                                          "str_sim": 0.0,
                                                          "popularity": 0.0,
                                                          "col_ctx_score": 0.0,
                                                          "row_ctx_score": 0.0,
                                                          "final": 0.0}]
                else:
                    candid_entities_info = [{"entity": ent,
                                             "str_sim": str_sim,
                                             "popularity": popularity,
                                             "col_ctx_score": 0.0,
                                             "row_ctx_score": property_feature_cache[(i, j, ent)]["score"],
                                             "final": (1 - alpha - beta - gamma) * str_sim +
                                                      gamma * popularity +
                                                      beta * property_feature_cache[(i, j, ent)]["score"]} \
                                            for ent, str_sim, popularity in self.candid_list[(i, j)]]
                    sorted_candid_entities_info = sorted(candid_entities_info, key=lambda x: x["final"], reverse=True)
                    self.predict_entities[(i, j)] = sorted_candid_entities_info[0]
                    self.candid_entities_info[(i, j)] = sorted_candid_entities_info[:init_prune_topk]
                    self.candid_list[(i, j)] = [(x['entity'], x['str_sim'], x['popularity'])
                                                for x in self.candid_entities_info[(i, j)]]

    def reassign_pred(self, candid_entities_info: Dict[Tuple[int, int], List[Dict[str, Any]]]):
        has_changed = False
        for i in range(self.tab.row_size):
            for j in range(self.tab.col_size):
                candid_info = candid_entities_info[(i, j)]
                sorted_candid_info = sorted(candid_info, key=lambda x: x["final"], reverse=True)
                if len(sorted_candid_info) > 0:
                    if sorted_candid_info[0]["entity"] != self.predict_entities[(i, j)]["entity"]:
                        has_changed = True
                    self.predict_entities[(i, j)] = deepcopy(sorted_candid_info[0])
                self.candid_entities_info[(i, j)] = sorted_candid_info
        return has_changed

    def set_revisit_pred(self, revisit_predict_entities):
        self.revisit_predict_entities = revisit_predict_entities

    def index_one_item(self, collection, i, j):
        return collection[(i, j)]

    def index_one_row(self, collection, i):
        return [collection[(i, j)] for j in range(self.tab.col_size)]

    def index_one_col(self, collection, j):
        return [collection[(i, j)] for i in range(self.tab.row_size)]

    def add_ent_title(self, entity_meta_info):
        # for revisit entities
        for pos in self.revisit_predict_entities:
            self.revisit_predict_entities[pos]["entity_title"] = \
                entity_meta_info.get_item_name(self.revisit_predict_entities[pos]["entity"])
        # for predicted entities
        for pos in self.predict_entities:
            self.predict_entities[pos]["entity_title"] = \
                entity_meta_info.get_item_name(self.predict_entities[pos]["entity"])

    def set_main_col_idx(self, main_col_idx):
        self.main_col_idx = main_col_idx

    def set_tab_pred_type(self, tab_pred_type):
        self.tab_pred_type = tab_pred_type

    def str_key(self, input):
        output = dict()
        for k in input:
            output[str(k)] = input[k]
        return output

    def set_property_feature_cache(self, property_feature_cache):
        self.property_feature_cache = property_feature_cache

    def set_tf_property_weights(self, tf_property_item_weights, tf_lexical_item_weights):
        self.tf_property_item_weights = tf_property_item_weights
        self.tf_lexical_item_weights = tf_lexical_item_weights

    def set_pre_property(self, pre_property):
        self.pre_property = pre_property

    def dump_one_tab(self):

        return {"main_col_idx": self.main_col_idx,
                "tab_id": self.tab.tab_id,
                "row_size": self.tab.row_size,
                "col_size": self.tab.col_size,
                "revisit_predict_entities": self.str_key(self.revisit_predict_entities),
                "predict_entities": self.str_key(self.predict_entities),
                "tab_pred_type": self.tab_pred_type,
                "coarse_candid_entities_info": self.str_key(self.candid_entities_info),
                "tf_property_item_weights": self.str_key(self.tf_property_item_weights),
                "tf_lexical_item_weights": self.str_key(self.tf_lexical_item_weights),
                "property_feature_cache": self.str_key(self.property_feature_cache),
                "coarse_properties": self.pre_property}

    def dump_one_tab_v2(self):
        return {"tab_id": self.tab.tab_id,
                "row_size": self.tab.row_size,
                "col_size": self.tab.col_size,
                "predict_entities": self.str_key(self.predict_entities),
                "tab_pred_type": self.tab_pred_type}

    def extract_entities(self, predict_entities):
        tight_pred_entities = dict()
        for x in predict_entities:
            tight_pred_entities[x] = {
                "entity": predict_entities[x]["entity"],
                "entity_title": predict_entities[x]["entity_title"],
                "final": predict_entities[x]["final"]
            }
        return tight_pred_entities

    def pack_ent_predictions(self, predict_entities):

        output = [[None for j in range(self.tab.col_size)] for i in range(self.tab.row_size)]

        for x in predict_entities:

            entity_id = predict_entities[x]["entity"]

            obj = {
                "entity": entity_id,
                "entity_title": predict_entities[x]["entity_title"],
                "confidence": predict_entities[x]["final"]
            }

            if self.map_ids and self.mapper is not None:
                satori_id = self.mapper.query(entity_id)
                if satori_id is not None:
                    obj["entity_sid"] = satori_id

            row_id, col_id = x[0], x[1]
            output[row_id][col_id] = obj
        return output

    def pack_property_predictions(self, predict_properties):
        output = dict()
        for col_id in predict_properties:
            if len(predict_properties[col_id]) > 0:
                output[col_id] = predict_properties[col_id][0]
        return output

    def pack_type_predictions(self, tab_pred_type):
        output = [None for j in range(self.tab.col_size)]
        for col_id in tab_pred_type:
            output[col_id] = {
                "type_id": tab_pred_type[col_id]["best_type"],
                "type_name": tab_pred_type[col_id]["best_type_name"],
                "confidence": tab_pred_type[col_id]["confidence"],
            }
        return output

    def gen_online(self):
        return {"main_col_idx": self.main_col_idx,
                "tab_id": self.tab.tab_id,
                "row_size": self.tab.row_size,
                "col_size": self.tab.col_size,
                "col_properties": self.pack_property_predictions(self.pre_property),
                "entities": self.pack_ent_predictions(self.revisit_predict_entities),
                "col_types": self.pack_type_predictions(self.tab_pred_type)}


def check_cell_filtering(cell_text):
    if config.keep_cell_filtering:
        if config.allow_int:
            if is_float_but_not_int(cell_text) or is_date(cell_text):
                return True
        else:
            if is_float(cell_text) or is_int(cell_text) or is_date(cell_text):
                return True
    if len(cell_text) == 0:
        return True
    return False
