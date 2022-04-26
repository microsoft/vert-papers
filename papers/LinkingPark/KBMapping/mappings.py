# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import csv
import os


class KBMapper(object):

    DEFAULT_LANGUAGE = 'en'

    LANG_FIELD = 'language'
    WIKIPEDIA_FIELD = 'wikipedia_id'
    TITLE_FIELD = 'title'
    SATORI_FIELD = 'satori_id'
    RANK_FIELD = 'static_rank'
    WIKIDATA_FIELD = 'wikidata_id'
    FREEBASE_FIELD = 'freebase_id'
    TYPE_FIELD = 'type_struct'

    field_names = [LANG_FIELD, WIKIPEDIA_FIELD, TITLE_FIELD, SATORI_FIELD, RANK_FIELD, WIKIDATA_FIELD,
                   FREEBASE_FIELD, TYPE_FIELD]

    wikidata_satori_pairs = dict()

    def retrieve_tsv_mapping(self, filename, languages=None, limit=-1):

        if languages is None:
            languages = [self.DEFAULT_LANGUAGE]

        with open(filename, mode="r", encoding="utf-8") as fp:

            lines = []

            reader = csv.DictReader(fp, fieldnames=self.field_names, dialect='excel-tab')

            for row in reader:
                if row[self.LANG_FIELD] in languages:
                    lines.append(row)

                if limit is not -1 and len(lines) == limit:
                    break

            return lines

    def get_wikidata_satori_pairs(self, map_table):
        pairs = dict()

        for entry in map_table:
            wikidata_id = entry.get(self.WIKIDATA_FIELD)
            satori_id = entry.get(self.SATORI_FIELD)
            if wikidata_id is not None and satori_id is not None:
                pairs[wikidata_id] = satori_id

        return pairs

    def dump_pairs_as_json(self, pairs, key_name, value_name, filepath):

        filename = os.path.join(filepath, f'{key_name}_to_{value_name}.json')

        with open(filename, 'w', encoding="utf-8") as fp:
            json.dump(pairs, fp)

    # Process table and persist pairs for wikidata and satori
    def process_ids_table(self, filename):

        output_dir = os.path.dirname(filename)

        map_table = self.retrieve_tsv_mapping(filename)
        pairs = self.get_wikidata_satori_pairs(map_table)
        self.dump_pairs_as_json(pairs, 'wikidata', 'satori', output_dir)

    def retrieve_pairs(self, filename):

        with open(filename, 'r', encoding="utf-8") as fp:
            return json.load(fp)

    def init_pairs(self, filename):
        self.wikidata_satori_pairs = self.retrieve_pairs(filename)

    def query(self, source_id):

        if not len(self.wikidata_satori_pairs) == 0:
            return self.wikidata_satori_pairs.get(source_id)
        else:
            return None
