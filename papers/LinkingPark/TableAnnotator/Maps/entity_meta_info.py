# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Set, List, Any, Tuple
import jsonlines
from tqdm import tqdm
from TableAnnotator.Config import config
from copy import deepcopy
from TableAnnotator.Config import shortname
from TableAnnotator.Maps.kb_store import KBStore
from GlobalConfig import global_config
import time


class EntityMetaInfo(object):
    def __init__(self, params):
        self.schema_ent_meta_info = EntityMetaInfo.load_data(params["schema_entity_meta_fn"])
        if params["kb_store"] == "RAM":
            self.ent_meta_info = EntityMetaInfo.load_data(params["entity_meta_fn"])
        else:
            # self.ent_meta_info = EntityMetaInfo.load_data(params["schema_entity_meta_fn"])
            self.ent_meta_info = dict()
            params = {
                "kb_store": params["kb_store"],
                "host": global_config.ent_meta_redis_host,
                "port": global_config.ent_meta_redis_port,
                "db_path": global_config.rocksdb_meta_info_path
            }
            self.kb_store_interface = KBStore(params=params)

    def retrieve_candid_kb_info(self, ent_set: Set[str]):
        t1 = time.time()
        for ent_id in ent_set:
            output = self.kb_store_interface.get_obj(ent_id)
            if output:
                self.ent_meta_info[ent_id] = output
                self.ent_meta_info[ent_id][shortname.TYPES] = \
                    set(self.ent_meta_info[ent_id][shortname.TYPES])
        t2 = time.time()
        # print('kb access for meta info {}s'.format(t2-t1))

    @staticmethod
    def load_data(fn: str) -> Dict[str, Dict]:
        with jsonlines.open(fn, mode="r") as fp:
            entity_meta_info = dict()
            n = 0
            for line in tqdm(fp):
                entity_meta_info[line["id"]] = line
                entity_meta_info[line["id"]][shortname.TYPES] = \
                    set(entity_meta_info[line["id"]][shortname.TYPES])
                n += 1
                if config.debug and n > 10000:
                    break
            return entity_meta_info

    def get_item_name(self, item_id: str) -> str:
        if item_id in self.schema_ent_meta_info:
            # if schema node, directly access RAM
            data = self.schema_ent_meta_info[item_id]
            return data[shortname.ENT_NAME].replace(" ", "_")
        if item_id in self.ent_meta_info:
            data = self.ent_meta_info[item_id]
            return data[shortname.ENT_NAME].replace(" ", "_")
        return "NIL"

    def get_entity_types(self, item_id: str) -> Set[str]:
        if item_id in self.schema_ent_meta_info:
            # if schema node, directly access RAM
            data = self.schema_ent_meta_info[item_id]
            return data[shortname.TYPES]
        if item_id in self.ent_meta_info:
            data = self.ent_meta_info[item_id]
            return data[shortname.TYPES]
        return set()

    def get_entity_rank_types(self, item_id: str) -> Dict[str, List[str]]:
        if item_id in self.schema_ent_meta_info:
            # if schema node, directly access RAM
            data = self.schema_ent_meta_info[item_id]
            return data[shortname.TYPES_RANK]
        if item_id in self.ent_meta_info:
            data = self.ent_meta_info[item_id]
            return data[shortname.TYPES_RANK]
        return {
            shortname.INSTANCE_OF: [],
            shortname.SUBCLASS_OF: []
        }

    def track_parents(self, item_id: str, level: int) -> Set[str]:
        """
        :param item_id: a Q number
        :param level: the number of levels above the current level that we want to go to
        :return: types: a set that contains all the parent types (instance of) of the node up to the number
                 levels specified. i.e: If Belgium is a federal system, it's also form of government because federal
                 system is an instance of form of government.
        """
        types = deepcopy(self.get_entity_types(item_id))
        pre_level_types = types
        for i in range(level):
            new_level_types = set()
            for t in pre_level_types:
                parents = self.get_entity_types(t)
                if len(parents) == 0:
                    continue
                for p in parents:
                    if p not in types:
                        new_level_types.add(p)
            pre_level_types = new_level_types
            types |= pre_level_types
        return types

    def track_parents_level_ins_sub(self, item_id: str, level: int) -> Tuple[Set[str], Dict[str, int]]:
        """
        :param item_id: a Q number
        :param level: the number of levels above the current level that we want to go to
        :return: types: a set that contains all the parent types (instance of) of the node up to the number
                 levels specified. i.e: If Belgium is a federal system, it's also form of government because federal
                 system is an instance of form of government.
        """
        types = deepcopy(set(self.get_entity_rank_types(item_id)[shortname.INSTANCE_OF]))

        pre_level_types = types
        type_level = {}
        for t in types:
            type_level[t] = 0
        for i in range(level):
            new_level_types = set()
            for t in pre_level_types:
                parents = self.get_entity_rank_types(t)[shortname.SUBCLASS_OF]
                if len(parents) == 0:
                    continue
                for p in parents:
                    if p not in types:
                        type_level[p] = i + 1
                        new_level_types.add(p)
            pre_level_types = new_level_types
            types |= pre_level_types
        return types, type_level

    def is_child(self, t1, t2):
        # t1 is a child of t2
        return t2 in self.get_entity_types(t1)

    def free(self):
        del self.ent_meta_info
        # gc.collect()
        self.ent_meta_info = dict()