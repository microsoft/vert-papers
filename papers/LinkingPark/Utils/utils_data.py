# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import json

wikidata_prefix = "http://www.wikidata.org/entity/"
property_prefix = "http://www.wikidata.org/prop/direct/"


def pd_read_csv(fn):
    df = pd.read_csv(fn, header=0)
    return df.values.tolist()


def read_csv(fn):
    """

    :param fn: file path
    :return: list of rows
    """
    table = list()
    with open(fn, encoding="utf-8", mode="r") as fp:
        # skip header
        header = fp.readline().split(',')
        for line in fp:
            cells = line.strip().split(',')
            assert len(header) == len(cells)
            table.append(cells)
    return table


def transpose_table(row_table):
    column_table = list()
    rows = len(row_table)
    cols = len(row_table[0])
    for i in range(cols):
        column_table.append([row_table[j][i] for j in range(rows)])
    return column_table


def read_cea_target(target_fn):
    targets = dict()
    with open(target_fn, encoding="utf-8", mode="r") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id, row_id, col_id = words[0], int(words[1]), int(words[2])
            if tab_id not in targets:
                targets[tab_id] = dict()

            if col_id not in targets[tab_id]:
                targets[tab_id][col_id] = []

            targets[tab_id][col_id].append(row_id)
        return targets


def read_cta_target(target_fn):
    targets = dict()
    with open(target_fn, encoding="utf-8", mode="r") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id, col_id = words[0], int(words[1])
            if tab_id not in targets:
                targets[tab_id] = []
            targets[tab_id].append(col_id)
        return targets


def read_cpa_target(target_fn):
    targets = dict()
    with open(target_fn, encoding="utf-8", mode="r") as fp:
        for line in fp:
            line = line.replace("\"", "")
            words = line.strip().split(',')
            tab_id, src_id, trg_id = words[0], int(words[1]), int(words[2])
            if tab_id not in targets:
                targets[tab_id] = []
            targets[tab_id].append(trg_id)
        return targets


def write_CEA_result(re_fn, col_entities, cea_targets):
    with open(re_fn, encoding="utf-8", mode="w") as fp:
        for tab_id in cea_targets:
            for col_id in cea_targets[tab_id]:
                if tab_id in col_entities and col_id in col_entities[tab_id]:
                    entities = col_entities[tab_id][col_id]
                    for row_id in cea_targets[tab_id][col_id]:
                        # skip header line
                        if entities[row_id-1] != "NIL" and entities[row_id-1] != None:
                            # fp.write("\"{}\",\"{}\",\"{}\",\"{}\"\n".format(tab_id,
                            #                                               col_id,
                            #                                               row_id,
                            #                                               wikidata_prefix+entities[row_id-1]))
                            fp.write("\"{}\",\"{}\",\"{}\",\"{}\"\n".format(tab_id,
                                                                          row_id,
                                                                          col_id,
                                                                          wikidata_prefix+entities[row_id-1]))


def write_CTA_result(re_fn, column_types, cta_targets):
    with open(re_fn, encoding="utf-8", mode="w") as fp:
        for tab_id in cta_targets:
            for col_id in cta_targets[tab_id]:
                if (tab_id in column_types) and (col_id in column_types[tab_id]):
                    best_type = column_types[tab_id][col_id]
                    if best_type != "":
                        fp.write("\"{}\",\"{}\",\"{}\"\n".format(tab_id,
                                                                 col_id,
                                                                 wikidata_prefix+best_type))


def write_CPA_result(re_fn, column_properties, cpa_targets):
    with open(re_fn, encoding="utf-8", mode="w") as fp:
        for tab_id in cpa_targets:
            for col_id in cpa_targets[tab_id]:
                if (tab_id in column_properties) and (col_id in column_properties[tab_id]):
                    property = column_properties[tab_id][col_id]
                    fp.write("\"{}\",\"0\",\"{}\",\"{}\"\n".format(tab_id,
                                                             col_id,
                                                             property_prefix+property))


def load_cache_result(log_fn):
    col_entities = dict()
    col_types = dict()
    col_properties = dict()
    with open(log_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            out_tab = json.loads(line.strip())
            tab_id = out_tab["tab_id"]
            predict_entities = out_tab["predict_entities"]
            revisit_predict_entities = out_tab["revisit_predict_entities"]
            tab_pred_type = out_tab["tab_pred_type"]
            if "fine_properties" in out_tab:
                properties = out_tab["fine_properties"]
            else:
                properties = out_tab["coarse_properties"]
            row_size = out_tab["row_size"]
            col_size = out_tab["col_size"]
            if tab_id not in col_entities:
                col_entities[tab_id] = dict()
            if tab_id not in col_types:
                col_types[tab_id] = dict()
            if tab_id not in col_properties:
                col_properties[tab_id] = dict()
            for j in range(col_size):
                if j == 0:
                    entities = []
                    for i in range(row_size):
                        wikidata_id = revisit_predict_entities[str((i, j))]["entity"]
                        entities.append(wikidata_id)
                else:
                    entities = []
                    for i in range(row_size):
                        wikidata_id = predict_entities[str((i, j))]["entity"]
                        entities.append(wikidata_id)
                col_entities[tab_id][j] = entities
                col_types[tab_id][j] = tab_pred_type[str(j)]["best_type"]
                if str(j) in properties and len(properties[str(j)]) > 0:
                    col_properties[tab_id][j] = properties[str(j)][0]["property"]
    return col_entities, col_types, col_properties


def load_cache_result_for_analysis(log_fn):
    log = dict()
    with open(log_fn, encoding='utf-8', mode='r') as fp:
        for line in fp:
            json_obj = json.loads(line.strip())
            tab_id = json_obj["tab_id"]
            log[tab_id] = json_obj
    return log
