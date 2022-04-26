# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any, Set, Tuple, List
import jsonlines
from tqdm import tqdm
from TableAnnotator.Config import config
from TableAnnotator.Config import shortname
from TableAnnotator.Maps.kb_store import KBStore
from GlobalConfig import global_config as global_config
import time


class KBPropertyVal(object):
    def __init__(self,
                 params):
        self.feature = params["ent_feature"]
        if params["kb_store"] == "RAM":
            self.property_info = self.load_data(params["property_value_fn"])
        else:
            self.property_info = dict()
            params = {
                "feature":  params["ent_feature"],
                "kb_store": params["kb_store"],
                "host": global_config.property_redis_host,
                "port": global_config.property_redis_port,
                "db_path": global_config.rocksdb_property_val_info_path
            }
            self.kb_store_interface = KBStore(params=params)

    def retrieve_candid_kb_info(self, ent_set: Set[str]):
        t1 = time.time()
        for ent_id in ent_set:
            output = self.kb_store_interface.get_obj(ent_id)
            if output:
                self.property_info[ent_id] = self.extract_kb_feature(output)
        t2 = time.time()
        # print('kb access for property info {}s'.format(t2-t1))

    def extract_kb_feature(self, ent_obj):
        if self.feature == 'type':
            ent_obj["kb_feature"] = set(ent_obj[shortname.TYPES])
        else:
            ent_obj["kb_feature"] = set(ent_obj[shortname.TYPES] + ent_obj[shortname.PROPERTIES])
        del ent_obj[shortname.TYPES]
        del ent_obj[shortname.PROPERTIES]
        return ent_obj

    def load_data(self, fn: str) -> Dict[str, Dict[str, Any]]:
        with jsonlines.open(fn, mode="r") as fp:
            property_info = dict()
            n = 0
            for line in tqdm(fp):
                property_info[line['id']] = self.extract_kb_feature(line)
                # if self.feature == 'type':
                #     property_info[line['id']]["kb_feature"] = set(line[shortname.TYPES])
                # else:
                #     property_info[line['id']]["kb_feature"] = set(line[shortname.TYPES]+
                #                                                   line[shortname.PROPERTIES])
                # del property_info[line['id']][shortname.TYPES]
                # del property_info[line['id']][shortname.PROPERTIES]
                n += 1
                if n > 10000 and config.debug:
                    break
            return property_info

    def query_kb_feature(self, item_id: str) -> Set[str]:
        """
                given an entity_id, return all properties + 2 level types of it
                :param item_id:
                :return:
                """
        if item_id in self.property_info:
            return self.property_info[item_id]['kb_feature']
        return set()

    def get_property_values(self, item_id: str) -> Dict[str, List[Dict[str, str]]]:
        if item_id in self.property_info:
            return self.property_info[item_id][shortname.PROPERTIES_VALUES]
        return {}

    def free(self):
        del self.property_info
        # gc.collect()
        self.property_info = dict()

