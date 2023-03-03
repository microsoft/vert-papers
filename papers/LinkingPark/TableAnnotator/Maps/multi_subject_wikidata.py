# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any, Set, Tuple, List
from TableAnnotator.Maps.multi_subject_type_voter import TypeInference
from TableAnnotator.Util.numerical_utils import NumericalPropertyLinker
from TableAnnotator.Maps.entity_property_info import KBPropertyVal
from TableAnnotator.Maps.entity_meta_info import EntityMetaInfo
from TableAnnotator.Maps.ent_sim import SparseEntitySim
from TableAnnotator.Config import shortname


class Wikidata(object):

    def __init__(self,
                 config: Dict[str, Any],
                 property_info: KBPropertyVal,
                 entity_meta_info: EntityMetaInfo):

        self.config: Dict[str, Any] = config
        self.property_info = property_info
        self.entity_meta_info = entity_meta_info
        self.ent_sim = SparseEntitySim(self.property_info)

        # Stores the types of an entity
        self.wikidata_ent_type_map = TypeInference(config,
                                                   entity_meta_info)
        self.numerical_property_linker = NumericalPropertyLinker(config['use_characteristics'])

    def gen_property_item_weights(self,
                                  main_col_predict_entities: List[Dict[str, Any]],
                                  target_col_predict_entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """
            Given main column's predict entities & target column's predict entities, make statistics possible property
            along with their freq, sorted in weights
        """
        weights = dict()
        for i, x in enumerate(main_col_predict_entities):
            main_ent = x["entity"]
            target_ent = target_col_predict_entities[i]["entity"]
            main_ent_property_values = self.property_info.get_property_values(main_ent)
            exists_properties = set()
            for property in main_ent_property_values:
                for value in main_ent_property_values[property]:
                    if value[shortname.DTYPE] == shortname.ENT and value[shortname.VALUE] == target_ent:
                        exists_properties.add(property)
                        break
            for property in exists_properties:
                if property not in weights:
                    weights[property] = 1
                else:
                    weights[property] += 1
        # normalize
        for property in weights:
            weights[property] /= len(main_col_predict_entities)
        return weights

    def gen_candid_property_item_weights(self,
                                         main_col_candid_entities: List[Tuple[str, float]],
                                         target_col_candid_entities: List[Tuple[str, float]]) -> Dict[str, float]:
        weights = dict()
        for i in range(len(main_col_candid_entities)):
            main_col_list = main_col_candid_entities[i]
            target_col_list = target_col_candid_entities[i]
            target_col_set = set([x[0] for x in target_col_list])
            exists_properties = set()
            for e1 in main_col_list:
                main_ent = e1[0]
                main_ent_property_values = self.property_info.get_property_values(main_ent)
                for property in main_ent_property_values:
                    for value in main_ent_property_values[property]:
                        if value[shortname.DTYPE] == shortname.ENT and \
                                value[shortname.VALUE] in target_col_set:
                            exists_properties.add(property)
                            break

            for property in exists_properties:
                if property not in weights:
                    weights[property] = 1
                else:
                    weights[property] += 1

        # normalize
        for property in weights:
            weights[property] /= len(main_col_candid_entities)
        return weights

    def gen_candid_property_lexical_weights(self,
                                            main_col_candid_entities: List[Tuple[str, float]],
                                            target_col: List[str]) -> Dict[str, float]:
        weights = dict()
        for i in range(len(main_col_candid_entities)):
            main_col_list = main_col_candid_entities[i]
            cell_mention = target_col[i]
            exists_properties = dict()
            for e1 in main_col_list:
                main_ent = e1[0]
                main_ent_property_values = self.property_info.get_property_values(main_ent)
                for property in main_ent_property_values:
                    for value in main_ent_property_values[property]:
                        pnumber = property
                        ptype = value[shortname.DTYPE]
                        pvalue = value[shortname.VALUE]

                        if ptype == shortname.QUANTITY:
                            punits = value["unit"]
                        else:
                            punits = ""
                        if self.numerical_property_linker.is_match(main_ent, pnumber, ptype, pvalue, punits, cell_mention,
                                                                   "Direct Match", use_characteristics=False):
                            exists_properties[property] = self.config["strict_match_weight"]
                            continue

                        if self.numerical_property_linker.is_match(main_ent, pnumber, ptype, pvalue, punits, cell_mention,
                                                                   "Fuzzy Match", use_characteristics=False):
                            if exists_properties.get(property, 0.0) < self.config["fuzzy_match_weight"]:
                                exists_properties[property] = self.config["fuzzy_match_weight"]
                            continue

                        if self.config["use_characteristics"] and self.numerical_property_linker.is_match(main_ent, pnumber, ptype, pvalue, punits, cell_mention,
                                                                                                          "Fuzzy Match", use_characteristics=True):
                            if exists_properties.get(property, 0.0) < self.config["characteristic_match_weight"]:
                                exists_properties[property] = self.config["characteristic_match_weight"]
                            continue

            for property in exists_properties:
                if property not in weights:
                    weights[property] = exists_properties[property]
                else:
                    weights[property] += exists_properties[property]
        # normalize
        for property in weights:
            weights[property] /= len(main_col_candid_entities)
        return weights

    def retrieve_item_pair_possible_properties(self,
                                               e1: str,
                                               e2: str) -> List[str]:
        """
            Given e1 and e2, return all possible properties between them
        """
        possible_properties = set()
        main_ent_property_values = self.property_info.get_property_values(e1)
        for property in main_ent_property_values:
            for value in main_ent_property_values[property]:
                if value[shortname.DTYPE] == shortname.ENT and \
                        value[shortname.VALUE] == e2:
                    possible_properties.add(property)
                    break
        return list(possible_properties)

    def retrieve_head_item_tail_candid_pair_possible_properties(self,
                                                                e1: str,
                                                                candid: List[Tuple[str, float]]) -> List[str]:
        possible_properties = set()
        main_ent_property_values = self.property_info.get_property_values(e1)
        candid_set = set([x[0] for x in candid])

        for property in main_ent_property_values:
            for value in main_ent_property_values[property]:
                if value[shortname.DTYPE] == shortname.ENT and \
                        value[shortname.VALUE] in candid_set:
                    possible_properties.add(property)
                    break
        return list(possible_properties)

    def retrieve_head_candid_tail_item_pair_possible_properties(self,
                                                                candid: List[Tuple[str, float]],
                                                                e2: str) -> List[str]:
        possible_properties = set()
        for e1 in candid:
            e1 = e1[0]
            main_ent_property_values = self.property_info.get_property_values(e1)
            for property in main_ent_property_values:
                for value in main_ent_property_values[property]:
                    if value[shortname.DTYPE] == shortname.ENT and \
                            value[shortname.VALUE] == e2:
                        possible_properties.add(property)
                        break
        return list(possible_properties)

    def retrieve_lexical_matched_possible_properties(self,
                                                     e1: str,
                                                     cell_mention: str) -> Dict[str, float]:
        """
            Given cell text and e1, return most align property and value's similarity based on different datatype
        """
        possible_properties = dict()
        main_ent_property_values = self.property_info.get_property_values(e1)
        for property in main_ent_property_values:
            for value in main_ent_property_values[property]:
                pnumber = property
                ptype = value[shortname.DTYPE]
                pvalue = value[shortname.VALUE]

                if ptype == shortname.QUANTITY:
                    punits = value["unit"]
                else:
                    punits = ""

                if self.numerical_property_linker.is_match(e1, pnumber, ptype, pvalue, punits, cell_mention,
                                                           "Direct Match", use_characteristics=False):
                    possible_properties[property] = self.config["strict_match_weight"]
                    continue

                if self.numerical_property_linker.is_match(e1, pnumber, ptype, pvalue, punits, cell_mention,
                                                           "Fuzzy Match", use_characteristics=False):
                    if possible_properties.get(property, 0.0) < self.config["fuzzy_match_weight"]:
                        possible_properties[property] = self.config["fuzzy_match_weight"]
                    continue

                if self.config["use_characteristics"] and self.numerical_property_linker.is_match(e1, pnumber, ptype, pvalue, punits, cell_mention,
                                                                                                  "Fuzzy Match", use_characteristics=True):
                    if possible_properties.get(property, 0.0) < self.config["characteristic_match_weight"]:
                        possible_properties[property] = self.config["characteristic_match_weight"]
                    continue

        return possible_properties

    def cosine_ent_sim(self, e1, e2):
        return self.ent_sim.cosine_sim(e1, e2)

    def name_labels(self, labels: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
        ret_labels = []
        for x in labels:
            if "->" in x[0]:
                pname = self.entity_meta_info.get_item_name(x[0].split("->")[0])
                idname = self.entity_meta_info.get_item_name(x[0].split("->")[1])
                label_name = "{}->{}".format(pname, idname)
            else:
                label_name = self.entity_meta_info.get_item_name(x[0])
            ret_labels.append((label_name, x[1], x[2]))
        return ret_labels
