# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import string
from TableAnnotator.Util.utils import check_cell_filtering
from TableAnnotator.Config import config


class WikidataCandidateTypoMap(object):
    def __init__(self,
                 alias_map_fn):
        if config.debug:
            self.alias_map = dict()
        else:
            print("using RAM alias map")
            self.alias_map = WikidataCandidateTypoMap.load_alias_map(alias_map_fn)

    @staticmethod
    def load_alias_map(alias_map_fn):
        with open(alias_map_fn, mode="rb") as fp:
            alias_map = pickle.load(fp)
            return alias_map

    def edits1(self, word):
        "All edits that are one edit away from `word`."

        letters = ' abcdefghijklmnopqrstuvwxyz'+string.punctuation+'0123456789'+'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def edit_mention(self, mention):
        edit_str_set = self.edits1(mention)
        valid_edit_str_set = [x for x in edit_str_set if x in self.alias_map]
        return valid_edit_str_set

    def gen_candid_spell_corrector(self, cell_text, strict_match_first=False, min_edit_len=10):
        if check_cell_filtering(cell_text):
            return []

        if strict_match_first or len(str(cell_text)) <= min_edit_len:
            if cell_text in self.alias_map:
                entities = [(x, cell_text) for x in self.alias_map[cell_text]]
            else:
                entities = []
            if len(entities) > 0:
                return entities
        valid_edit_str_set = self.edit_mention(cell_text)
        result = []
        for mention in valid_edit_str_set:
            entities = [(x, mention) for x in self.alias_map[mention]]
            result += entities
        return result
