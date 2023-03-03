# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Any
import copy
from TableAnnotator.Maps.multi_subject_wikidata import Wikidata
from TableAnnotator.Util.utils import Util, InputTable, OutputTable
from TableAnnotator.Maps.offline_candid_map import OfflineCandidateMap
from TableAnnotator.Maps.mixed_candid_map import MixedCandidateMap
from TableAnnotator.Maps.entity_meta_info import EntityMetaInfo
from TableAnnotator.Maps.entity_property_info import KBPropertyVal
from KBMapping.mappings import KBMapper
from datetime import datetime


class LinkingPark(object):

    def __init__(self, params: Dict[str, Any]):

        self.config: Dict[str, Any] = params

        # Load wikidata - satori pairs
        self.mapper = KBMapper()
        self.mapper.init_pairs(self.config['id_mapping_fn'])

        # Stores prior information
        if self.config["candid_gen_method"] == "online":
            self.candid_gen = MixedCandidateMap(params)
        elif self.config["candid_gen_method"] == "offline":
            self.candid_gen = OfflineCandidateMap(params["candid_map_fn"])
        else:
            raise ValueError("unsupported candid_gen_method: {}".format(self.config["candid_gen_method"]))
        print("loading prior done.")

        self.entity_meta_info = EntityMetaInfo(params)
        self.property_info = KBPropertyVal(params)

        # Stores wikidata information
        self.wikidata = Wikidata(params,
                                 self.property_info,
                                 self.entity_meta_info)
        print("wikidata loaded.")

        self.iter_steps_count = dict()
        self.time_counter = {
            "kb_access": 0,
            "gen_candid": 0,
            "model_run": 0,
        }

    def _extract_col_feature(self,
                             ent: str,
                             row_idx: int,
                             col_idx: int,
                             row_size: int,
                             output_tab: OutputTable):

        col_support_scores = []

        for k, c2 in enumerate(output_tab.index_one_col(output_tab.predict_entities, col_idx)):
            # calculate entity similarity with other assigned entities in the same column
            if k != row_idx:
                sim_score = self.wikidata.cosine_ent_sim(ent, c2["entity"])
                col_support_scores.append(sim_score)

        col_ent_sim_score = sum(col_support_scores) / (row_size - 1) \
            if row_size > 1 else 0.0

        return col_ent_sim_score

    def _extract_row_feature(self,
                             ent: str,
                             tab: InputTable,
                             output_tab: OutputTable,
                             row_idx: int,
                             col_idx: int,
                             tf_property_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                             tf_lexical_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                             property_mask: Dict[Tuple[int, int], int]):

        tail_item_matched_properties = {}
        head_item_matched_properties = {}
        head_lexical_matched_properties = {}
        rows_support_scores = {}
        directions = {}
        total_related_columns = 0
        total_score = 0.0

        for other_idx in range(tab.col_size):
            if other_idx == col_idx:
                continue

            # other_idx and col_idx relationships: either subject-object or object-subject
            # IF other_idx is subject

            tail_possible_item_property_feature = self.wikidata.retrieve_head_candid_tail_item_pair_possible_properties(
                output_tab.index_one_item(output_tab.candid_list, row_idx, other_idx), ent)

            if len(tail_possible_item_property_feature) > 0:
                s_tail = max(tf_property_item_weights[(other_idx, col_idx)].get(p, 0.0) for p in tail_possible_item_property_feature)
            else:
                s_tail = 0.0

            # IF other_idx is object

            head_possible_item_property_feature = self.wikidata.retrieve_head_item_tail_candid_pair_possible_properties(
                ent, output_tab.index_one_item(output_tab.candid_list, row_idx, other_idx))
            if len(head_possible_item_property_feature) > 0:
                s_head_ent = max(tf_property_item_weights[(col_idx, other_idx)].get(p, 0.0)
                                 for p in head_possible_item_property_feature)
            else:
                s_head_ent = 0.0

            head_possible_lexical_property_feature = self.wikidata.retrieve_lexical_matched_possible_properties(
                ent,
                tab.index_one_cell(row_idx, other_idx))
            if len(head_possible_lexical_property_feature) > 0:
                s_head_literal = max(
                    tf_lexical_item_weights[(col_idx, other_idx)].get(p, 0.0) * head_possible_lexical_property_feature[p] for p in
                    head_possible_lexical_property_feature)
            else:
                s_head_literal = 0.0

            s_head = max(s_head_ent, s_head_literal)

            if other_idx < col_idx:
                if s_tail >= s_head:
                    d = "Tail"
                else:
                    d = "Head"
            else:
                if s_head >= s_tail:
                    d = "Head"
                else:
                    d = "Tail"
            directions[(other_idx, col_idx)] = d

            rows_support_scores[(other_idx, col_idx)] = max(s_head, s_tail)
            tail_item_matched_properties[(other_idx, col_idx)] = tail_possible_item_property_feature
            head_item_matched_properties[(other_idx, col_idx)] = head_possible_item_property_feature
            head_lexical_matched_properties[(other_idx, col_idx)] = head_possible_lexical_property_feature

            # total_related_columns += property_mask[(other_idx, col_idx)]
            # total_score += property_mask[(other_idx, col_idx)] * max(s_head, s_tail)
            if property_mask[(other_idx, col_idx)] or property_mask[(col_idx, other_idx)]:
                total_related_columns += 1
                total_score += max(property_mask[(col_idx, other_idx)] * s_head,
                                   property_mask[(other_idx, col_idx)] * s_tail)

        row_support_score = total_score / total_related_columns if total_related_columns > 0 else 0.0
        property_info = {
                "score": row_support_score,
                "tail_item_matched_properties": tail_item_matched_properties,
                "head_item_matched_properties": head_item_matched_properties,
                "head_lexical_matched_properties": head_lexical_matched_properties
            }
        return property_info

    def _extract_global_row_feature(self,
                                    tab: InputTable,
                                    output_tab: OutputTable,
                                    tf_property_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                                    tf_lexical_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                                    property_mask: Dict[Tuple[int, int], int]):

        property_feature_cache = dict()

        # calculate context score in each row and column for each cell
        for j in range(tab.col_size):
            for i in range(tab.row_size):
                # iterate each cell
                candid = output_tab.index_one_item(output_tab.candid_list, i, j)
                for k, c in enumerate(candid):
                    # extract row ent_sim features
                    property_info = self._extract_row_feature(c[0], tab,
                                                              output_tab, i, j,
                                                              tf_property_item_weights,
                                                              tf_lexical_item_weights,
                                                              property_mask)
                    property_feature_cache[(i, j, c[0])] = property_info
        return property_feature_cache

    def _generate_property_tf_weights(self,
                                      tab: InputTable,
                                      output_tab: OutputTable,
                                      main_col_idx=0):
        # global tf weights
        tf_property_item_weights = dict()
        tf_lexical_item_weights = dict()

        # generate main' column & per target column's TF weights for row_ctx_scores
        for j in range(tab.col_size):
            if j == main_col_idx:
                continue
            # use all candidates
            main_candid = output_tab.index_one_col(output_tab.candid_list, main_col_idx)
            target_candid = output_tab.index_one_col(output_tab.candid_list, j)

            # compare KB's values w.r.t entity in meaning space
            item_weights = self.wikidata.gen_candid_property_item_weights(
                main_candid,
                target_candid)
            tf_property_item_weights[j] = item_weights

            # compare KB's values w.r.t cell text in lexical space
            lexical_weights = self.wikidata.gen_candid_property_lexical_weights(
                main_candid,
                tab.index_one_col(j))
            tf_lexical_item_weights[j] = lexical_weights

        return tf_property_item_weights, tf_lexical_item_weights

    @staticmethod
    def _generate_all_pair_property_mask(tf_property_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                                         tf_lexical_item_weights: Dict[Tuple[int, int], Dict[str, float]],
                                         min_property_threshold=0.3):
        mask = dict()
        for x in tf_property_item_weights:
            if len(tf_property_item_weights[x]) > 0:
                max_ent_score = max(tf_property_item_weights[x].values())
            else:
                max_ent_score = 0.0
            if len(tf_lexical_item_weights[x]) > 0:
                max_literal_score = max(tf_lexical_item_weights[x].values())
            else:
                max_literal_score = 0.0
            if max(max_ent_score, max_literal_score) >= min_property_threshold:
                mask[x] = 1
            else:
                mask[x] = 0
        return mask

    def _generate_all_pair_property_tf_weights(self,
                                               tab: InputTable,
                                               output_tab: OutputTable):
        # global tf weights
        tf_property_item_weights = dict()
        tf_lexical_item_weights = dict()

        # generate main' column & per target column's TF weights for row_ctx_scores
        for main_col_idx in range(tab.col_size):
            for j in range(tab.col_size):
                if j == main_col_idx:
                    continue
                # use all candidates
                main_candid = output_tab.index_one_col(output_tab.candid_list, main_col_idx)
                target_candid = output_tab.index_one_col(output_tab.candid_list, j)

                # compare KB's values w.r.t entity in meaning space
                item_weights = self.wikidata.gen_candid_property_item_weights(
                    main_candid,
                    target_candid)
                # subject column index <-> object column index
                tf_property_item_weights[(main_col_idx, j)] = item_weights

                # compare KB's values w.r.t cell text in lexical space
                lexical_weights = self.wikidata.gen_candid_property_lexical_weights(
                    main_candid,
                    tab.index_one_col(j))
                tf_lexical_item_weights[(main_col_idx, j)] = lexical_weights

        return tf_property_item_weights, tf_lexical_item_weights

    def ica_predict(self,
                    tab,
                    output_tab: OutputTable,
                    init_prune_topk,
                    max_iter,
                    alpha,
                    beta,
                    gamma,
                    min_property_threshold=0.7,
                    row_feature_only=True):

        # generate all-pairs property tf weights
        tf_property_item_weights, tf_lexical_item_weights = self._generate_all_pair_property_tf_weights(tab,
                                                                                                        output_tab)

        property_mask = LinkingPark._generate_all_pair_property_mask(tf_property_item_weights,
                                                                     tf_lexical_item_weights,
                                                                     min_property_threshold=min_property_threshold)

        # generate global row feature
        property_feature_cache = self._extract_global_row_feature(tab, output_tab,
                                                                  tf_property_item_weights,
                                                                  tf_lexical_item_weights,
                                                                  property_mask=property_mask)

        # init predictions
        output_tab.init_pred(alpha, beta, gamma, property_feature_cache, init_prune_topk)

        has_changed = True
        iter_step = 0

        # property_feature_cache = dict()
        '''
        property feature cache: (i, j) ->
                                {
                                   "score": float,
                                   "lexical_matched_properties": [],
                                   "item_matched_properties": [],
                                   "lexical_matched_properties_score": [],
                                   "item_matched_properties_score": []
                                }
        where score = SUM(MAX(lexical_matched_properties_score, item_matched_properties_score)) / LEN(item_matched_properties_score)
        '''

        while has_changed and iter_step < max_iter:
            iter_step += 1

            # calculate context score in each row and column for each cell
            candid_entities_info = dict()
            for j in range(tab.col_size):
                for i in range(tab.row_size):
                    # iterate each cell

                    # extract col features
                    candid = output_tab.index_one_item(output_tab.candid_list, i, j)
                    candid_info = []
                    for k, c in enumerate(candid):
                        # for each candidate entity of current cell
                        if not row_feature_only:
                            col_ent_sim_score = self._extract_col_feature(c[0],
                                                                          i, j,
                                                                          tab.row_size,
                                                                          output_tab)
                        else:
                            col_ent_sim_score = 0.0

                        info = {
                            "col_ctx_score": col_ent_sim_score,
                            "str_sim": c[1],
                            "popularity": c[2],
                            "entity": c[0]
                        }

                        # extract row ent_sim features
                        if (i, j, c[0]) in property_feature_cache:
                            property_info = property_feature_cache[(i, j, c[0])]
                        else:
                            property_info = self._extract_row_feature(c[0], tab,
                                                                      output_tab, i, j,
                                                                      tf_property_item_weights,
                                                                      tf_lexical_item_weights, property_mask)
                            property_feature_cache[(i, j, c[0])] = property_info
                        info["row_ctx_score"] = property_info["score"]
                        # info["final"] = alpha * info["col_ctx_score"] + \
                        #                 beta * info["row_ctx_score"] + \
                        #                 gamma * info["popularity"] + \
                        #                 (1 - alpha - beta - gamma) * info["str_sim"]
                        info["final"] = alpha * info["col_ctx_score"] + \
                                        beta * info["row_ctx_score"] + \
                                        (1 - alpha - beta) * info["str_sim"]
                        candid_info.append(info)
                    candid_entities_info[(i, j)] = candid_info

            has_changed = output_tab.reassign_pred(candid_entities_info)

        output_tab.set_property_feature_cache(property_feature_cache)
        output_tab.set_tf_property_weights(tf_property_item_weights, tf_lexical_item_weights)

    def gen_revisit_pred_entities(self, tab, output_tab, alpha, beta, tab_pred_type):
        revisit_predict_entities = copy.deepcopy(output_tab.predict_entities)
        for j in range(tab.col_size):
            if j > 0:
                continue
            best_type = tab_pred_type[j]["best_type"]
            for i in range(tab.row_size):
                # If a top candidate for an entity does not have the best type, then we look through
                # the list of candidates before they were short listed to see if any of them have that type.
                if best_type not in self.entity_meta_info.track_parents(
                        revisit_predict_entities[(i, j)]["entity"],
                        self.config["k_level"]):
                    for c in output_tab.candid_list_before_shortlist[(i, j)]:
                        if best_type in self.entity_meta_info.track_parents(c[0], self.config["k_level"]):
                            revisit_predict_entities[(i, j)] = {"entity": c[0],
                                                                "str_sim": c[1],
                                                                "popularity": c[2],
                                                                "col_ctx_score": 0.0,
                                                                "row_ctx_score": 0.0,
                                                                "final": (1 - alpha - beta) * c[1]}
                            break
        return revisit_predict_entities

    def gen_property_outputs(self, out_tab):
        pre_property = dict()
        lexical_property_info = out_tab.tf_lexical_item_weights
        item_property_info = out_tab.tf_property_item_weights

        for x in lexical_property_info:
            col_pre_property = dict()
            for p in lexical_property_info[x]:
                col_pre_property[p] = lexical_property_info[x][p]
            for p in item_property_info[x]:
                if p not in col_pre_property:
                    col_pre_property[p] = item_property_info[x][p]
                else:
                    if item_property_info[x][p] > col_pre_property[p]:
                        col_pre_property[p] = item_property_info[x][p]
            list_info = []
            for p in col_pre_property:
                list_info.append(
                    {
                        "property_name": self.entity_meta_info.get_item_name(p),
                        "property": p,
                        "confidence": col_pre_property[p]
                    }
                )
            pre_property[x] = sorted(list_info, key=lambda x: x["confidence"], reverse=True)
        return pre_property

    def detect_single_table(self,
                            tab: InputTable,
                            keep_N=10,
                            alpha=0.4,
                            beta=0.3,
                            gamma=0.1,
                            max_iter=10,
                            topk=20,
                            init_prune_topk=10,
                            min_final_diff=0.01,
                            min_property_threshold=0.7,
                            row_feature_only=True,
                            map_ids=False):

        start = datetime.now()

        # init a output table
        output_tab = OutputTable(tab, map_ids, self.mapper)

        # content analysis
        # main_col_idx = tab.get_main_column_idx()

        t1 = datetime.now()
        # generate candidates
        # topk is the initial candidate size
        # keep_N is the shortlisted candidate size
        output_tab.gen_candidates(self.candid_gen, topk, keep_N)
        t2 = datetime.now()
        self.time_counter['gen_candid'] += (t2 - t1).total_seconds()

        if self.config['kb_store'] != 'RAM':
            ent_set = output_tab.gen_ent_set()
            self.entity_meta_info.retrieve_candid_kb_info(ent_set)
            self.property_info.retrieve_candid_kb_info(ent_set)
        t3 = datetime.now()
        self.time_counter['kb_access'] += (t3 - t2).total_seconds()

        # ICA
        self.ica_predict(tab, output_tab, init_prune_topk,
                         max_iter, alpha, beta, gamma,
                         min_property_threshold=min_property_threshold,
                         row_feature_only=row_feature_only)

        # generate property outputs
        pre_property_coarse = self.gen_property_outputs(output_tab)
        output_tab.set_pre_property(pre_property_coarse)

        # resort prior if top2 diff is smaller than min_final_diff
        output_tab.resort_final_prior(min_final_diff)

        # vote type
        tab_pred_type = self.wikidata.wikidata_ent_type_map.pred_type(tab,
                                                                      output_tab)
        output_tab.set_tab_pred_type(tab_pred_type)

        # gen revisit predict entities
        revisit_predict_entities = self.gen_revisit_pred_entities(tab, output_tab, alpha, beta, tab_pred_type)

        # set predictions
        output_tab.set_revisit_pred(revisit_predict_entities)
        output_tab.add_ent_title(self.entity_meta_info)

        # set main_col_idx
        # output_tab.set_main_col_idx(main_col_idx=main_col_idx)

        if self.config['kb_store'] != 'RAM':
            # Free KB cache
            self.entity_meta_info.free()
            self.property_info.free()
        end = datetime.now()
        self.time_counter["model_run"] += (end-start).total_seconds() - (t3-t1).total_seconds()
        return output_tab
