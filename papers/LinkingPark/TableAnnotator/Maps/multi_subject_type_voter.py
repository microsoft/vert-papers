# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Set, List, Tuple
import pickle
from TableAnnotator.Maps.entity_meta_info import EntityMetaInfo
from TableAnnotator.Config import shortname


class TypeInference(object):
    def __init__(self,
                 config,
                 entity_meta_info: EntityMetaInfo):
        self.config = config
        self.entity_meta_info = entity_meta_info
        self.type_count = TypeInference.load_type_count(config["type_count_fn"])

    @staticmethod
    def load_type_count(fn):
        with open(fn, mode="rb") as fp:
            type_count = pickle.load(fp)
            return type_count

    @staticmethod
    def cal_avg_levels(ent_types_level, node):
        s = 0
        for types_level in ent_types_level:
            s += types_level.get(node, 100)
        return s

    def cal_instance_of_rank(self, node):
        score = dict()
        instance_of_types = self.entity_meta_info.get_entity_rank_types(node)[shortname.INSTANCE_OF]
        for i, x in enumerate(instance_of_types):
            score[x] = i
        return score

    def vote_wikidata_type(self,
                           predict_type_info: Dict[str, int],
                           ent_types_level: List[Dict[str, int]],
                           ent_instance_rank: List[Dict[str, int]],
                           is_main=True) -> Tuple[str, int, Dict, Dict, List, Dict]:

        if len(predict_type_info) == 0:
            return "", 0, {}, {}, [], {}

        tie_nodes_debug = {}
        # by count
        max_count = max(predict_type_info.values())
        max_nodes = [t for t in predict_type_info if predict_type_info[t] == max_count]
        tie_nodes_debug["by_count"] = max_nodes

        max_nodes_avg_levels = dict()
        avg_ent_instance_rank = dict()
        for t in max_nodes:
            max_nodes_avg_levels[t] = TypeInference.cal_avg_levels(ent_types_level, t)
            avg_ent_instance_rank[t] = TypeInference.cal_avg_levels(ent_instance_rank, t)

        if is_main:
            # by min level
            min_level_score = min(max_nodes_avg_levels.values())
            max_min_nodes = [t for t in max_nodes if max_nodes_avg_levels[t] == min_level_score]
            tie_nodes_debug["by_level"] = max_min_nodes
            # by min population
            sorted_nodes = sorted(max_min_nodes, key=lambda x: self.type_count.get(x, 1000000))
            min_population = min([self.type_count.get(x, 1000000) for x in max_min_nodes])
            min_population_tie_nodes = [t for t in max_min_nodes if self.type_count.get(t, 1000000) == min_population]
            tie_nodes_debug["by_population"] = min_population_tie_nodes
            tie_nodes = min_population_tie_nodes

        else:
            # by min level
            # sorted_nodes = sorted(max_nodes, key=lambda x:max_nodes_avg_levels[x])
            min_level_score = min(max_nodes_avg_levels.values())
            max_min_nodes = [t for t in max_nodes if max_nodes_avg_levels[t] == min_level_score]
            tie_nodes_debug["by_level"] = max_min_nodes

            # by population
            min_population = min([self.type_count.get(x, 1000000) for x in max_min_nodes])
            min_population_tie_nodes = [t for t in max_min_nodes if self.type_count.get(t, 1000000) == min_population]
            tie_nodes_debug["by_population"] = min_population_tie_nodes

            sorted_nodes = sorted(max_min_nodes, key=lambda x: avg_ent_instance_rank[x])
            min_avg_ent_instance_rank = min([avg_ent_instance_rank[t] for t in max_min_nodes])
            tie_nodes = [t for t in max_min_nodes if
                         avg_ent_instance_rank[t] == min_avg_ent_instance_rank]
            tie_nodes_debug["by_avg_ent_instance_rank"] = tie_nodes

        if len(sorted_nodes) == 0:
            return "", 0, max_nodes_avg_levels, avg_ent_instance_rank, tie_nodes, tie_nodes_debug
        else:
            return sorted_nodes[0], predict_type_info[sorted_nodes[0]], max_nodes_avg_levels, avg_ent_instance_rank, tie_nodes, tie_nodes_debug

    def pred_type(self, tab, output_tab):
        # vote type
        # maps from an entity type to its normalized count across top predicted entities
        tab_pred_type = dict()
        for j in range(tab.col_size):
            predict_type_info = dict()
            ent_types_level = []
            ent_instance_rank_scores = []
            for i in range(tab.row_size):
                predict_entity = output_tab.index_one_item(output_tab.predict_entities, i, j)
                ent_types, types_level = self.entity_meta_info.track_parents_level_ins_sub(
                    predict_entity["entity"],
                    self.config["k_level"])
                score = self.cal_instance_of_rank(predict_entity["entity"])
                ent_types_level.append(types_level)
                ent_instance_rank_scores.append(score)
                for t in ent_types:
                    if t not in predict_type_info:
                        predict_type_info[t] = 1
                    else:
                        predict_type_info[t] += 1

            # if j == main_col_idx:
            best_type, count, max_nodes_avg_levels, \
            avg_instance_of_subclass_of_scores, \
            tie_nodes, tie_nodes_debug = self.vote_wikidata_type(
                predict_type_info,
                ent_types_level,
                ent_instance_rank_scores,
                is_main=True)
            # else:
            #     best_type, count, max_nodes_avg_levels, \
            #     avg_instance_of_subclass_of_scores, \
            #     tie_nodes, tie_nodes_debug = self.vote_wikidata_type(
            #         predict_type_info,
            #         ent_types_level,
            #         ent_instance_rank_scores,
            #         is_main=False)

            named_predict_type_info = dict()
            for node in predict_type_info:
                named_predict_type_info[node] = {"name": self.entity_meta_info.get_item_name(node),
                                                 "count": predict_type_info[node]}
            named_tie_nodes_debug = dict()
            for key in tie_nodes_debug:
                named_tie_nodes_debug[key] = [(x, self.entity_meta_info.get_item_name(x)) for x in tie_nodes_debug[key]]
            named_tie_nodes = [(node, self.entity_meta_info.get_item_name(node)) for node in tie_nodes]
            best_type_name = self.entity_meta_info.get_item_name(best_type).replace(" ", "_")
            pred_type_details = {"best_type": best_type,
                                 "best_type_name": best_type_name,
                                 "confidence": count / tab.row_size,#}
                                 "tie_nodes": named_tie_nodes,
                                 "predict_type_info": named_predict_type_info,
                                 "ent_types_level": ent_types_level,
                                 "avg_level_scores": max_nodes_avg_levels,
                                 "instance_of_subclass_of_scores": ent_instance_rank_scores,
                                 "avg_instance_of_subclass_of_scores": avg_instance_of_subclass_of_scores,
                                 "tie_nodes_debug": named_tie_nodes_debug}
            tab_pred_type[j] = pred_type_details
        return tab_pred_type
