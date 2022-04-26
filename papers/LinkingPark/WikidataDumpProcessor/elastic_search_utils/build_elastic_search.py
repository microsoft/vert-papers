# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from elasticsearch import client
from tqdm import tqdm
import pickle
import argparse
from TableAnnotator.Config.config_utils import process_relative_path_config


class ElasticSearchBuilder:

    def __init__(self, index_name):
        self.index_name = index_name
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=120)
        self.indexES = client.IndicesClient(self.es)
        if self.es.ping():
            print("Connect elasticsearch successfully!")
        else:
            print("Connect failed...")
        if not self.es.indices.exists(self.index_name):
            self.es.indices.create(index=self.index_name, ignore=400, body=None)
            print("create index successfully")
        else:
            print("index already exists.")

    def bulk_index_data(self, alias2qids_fn):
        def generator(alias2qids):
            cid = 0
            for alias, qids in tqdm(alias2qids.items()):
                cid += 1
                yield {
                    '_op_type': 'index',
                    '_id': cid,
                    '_index': self.index_name,
                    '_source': {
                        'title': alias,
                        'qid': ";".join(qids),
                    }
                }

        with open(alias2qids_fn, mode="rb") as fp:
            alias2qids = pickle.load(fp)

        for success, info in parallel_bulk(client=self.es,
                                           actions=generator(alias2qids),
                                           thread_count=5,
                                           request_timeout=60):
            if not success:
                print("Doc failed: {}".format(info))
        print("index data done.")

    def query_data(self,
                   mention,
                   top_k=50,
                   score_func='token'):

        if score_func == 'token':
            dsl = {
                "query": {
                    # single feature
                    "match": {"title": mention}
                }
            }
        else:
            dsl = {
                "query": {
                    # feature ensemble: title^2 + title.ngram
                    "multi_match": {
                        "query": mention,
                        "fields": ["title^2", "title.ngram"],
                        "type": "most_fields"
                    }
                }
            }

        res = self.es.search(index=self.index_name,
                             body=dsl,
                             size=top_k,
                             explain=False)

        hits = res["hits"]["hits"]
        if len(hits) == 0:
            return []
        ans = []
        for hit in hits:
            # qlist = hit["_source"]["qid"].split(";")
            qstr = hit["_source"]["qid"]
            htitle = hit["_source"]["title"]
            ans.append((qstr, hit['_score'], htitle))
        return ans

    def delete_indices(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])

    def generate_cands(self,
                       query_fn,
                       output_fn,
                       top_k=50,
                       score_func='token'):
        with open(query_fn, mode="rb") as fp:
            mentions = pickle.load(fp)

        ans = {}
        pbar = tqdm(total=len(mentions))
        for mention in mentions:
            if mention in ans:
                continue
            try:
                res = self.query_data(mention,
                                      top_k=top_k,
                                      score_func=score_func)
            except:
                res = []
            ans[mention] = res
            pbar.update(1)
        pbar.close()

        with open(output_fn, mode="wb") as fp:
            pickle.dump(ans, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", type=str, default="wikidata_keep_disambiguation")
    parser.add_argument("--alias2qids_fn", type=str, default="$BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_keep_disambiguation.pkl")
    args = process_relative_path_config(parser.parse_args())
    builder = ElasticSearchBuilder(args.index_name)
    builder.bulk_index_data(args.alias2qids_fn)
    print('done.')
