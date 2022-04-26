"""
The code is adapted from the official SemTab evaluator: https://github.com/sem-tab-challenge/aicrowd-evaluator
"""

import json
import os
from GlobalConfig import global_config

_TABLE_CATEGORIES = {
        'ALL': ([''], []),
        'CTRL_WIKI': (['WIKI'], ['NOISE2']),
        'CTRL_DBP': (['CTRL', 'DBP'], ['NOISE2']),
        'CTRL_NOISE2': (['CTRL', 'NOISE2'], []),
        'TOUGH_T2D': (['T2D'], ['NOISE2']),
        'TOUGH_HOMO': (['HOMO'], ['SORTED', 'NOISE2']),
        'TOUGH_MISC': (['MISC'], ['NOISE2']),
        'TOUGH_MISSP': (['MISSP'], ['NOISE1', 'NOISE2']),
        'TOUGH_SORTED': (['SORTED'], ['NOISE2']),
        'TOUGH_NOISE1': (['NOISE1'], []),
        'TOUGH_NOISE2': (['TOUGH', 'NOISE2'], [])
    }


def _is_table_in_cat(x, whitelist, blacklist):
    b = True
    for i in whitelist:
        if not (b and (i in x)):
            return False
    for e in blacklist:
        if not (b and (e not in x)):
            return False
    return True


def get_tables_categories():
    table_map = json.load(open(os.path.join(global_config.tough_table_benchmark_dir, "gt/filename_map.json"), 'r'))
    categories = {}
    for cat in _TABLE_CATEGORIES:
        categories[cat] = [fake_id for fake_id, tab_id in table_map.items()
                           if _is_table_in_cat(tab_id,  *_TABLE_CATEGORIES[cat])]
    return categories
