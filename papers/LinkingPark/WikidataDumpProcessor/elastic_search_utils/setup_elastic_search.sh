# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

index_name=$1

# create index
curl -XPUT "localhost:9200/${index_name}?pretty" -H 'Content-Type: application/json' -d'
{
    "settings" : {
        "number_of_shards" : 5,
        "number_of_replicas" : 1,
        "analysis" : {
            "analyzer" : {
                "my_analyzer" : {
                    "tokenizer" : "my_tokenizer"
                }
            },
            "tokenizer" : {
                "my_tokenizer" : {
                    "type" : "ngram",
                    "min_gram" : 3,
                    "max_gram" : 3,
                    "token_chars" : ["letter", "digit", "punctuation"]
                }
            }
        }
    }
}
'

# add mapping to index: text field use standard tokenizer; ngram field uses 3 char-level gram

curl -XPUT "localhost:9200/${index_name}/_mappings?pretty" -H 'Content-Type: application/json' -d'
{
    "properties" : {
        "title": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
                "ngram": {
                    "type": "text",
                    "analyzer": "my_analyzer"
                }
            }
        },
        "qid": {
            "type": "keyword",
            "index": false
        }
    }
}
'