## Configuring LinkingPark for SemTab 2021

### Download LinkingPark resources

Download resources for running experiments over SemTab 2021 [Azure storage](https://kcpapers.blob.core.windows.net/lp-2022/LP_Data_semtab21.zip). 
Then unzip it and put it in the ${BASE_DATA_DIR} setup in the previous readme.

### Environment setup
```shell
export BASE_DATA_DIR_SEMTAB21="${BASE_DATA_DIR}/SemTab21-Data"
```

#### Elastic Search

##### Installation

```shell
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.8.1-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.8.1-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.8.1-linux-x86_64.tar.gz.sha512 
tar -xzf elasticsearch-7.8.1-linux-x86_64.tar.gz
cd elasticsearch-7.8.1/
```

##### Start elasticsearch server

```shell
./bin/elasticsearch
```

##### Create index and add mappings

```shell
# Index for alias mapping removing disambiguation pages
bash WikidataDumpProcessor/elastic_search_utils/setup_elastic_search.sh "wikidata_20210830_rm_disambiguation"
# Index for alias mapping keeping disambiguation pages
bash WikidataDumpProcessor/elastic_search_utils/setup_elastic_search.sh "wikidata_20210830_keep_disambiguation"
```

##### Build index

```
# Build index for alias mapping removing disambiguation pages
python WikidataDumpProcessor/elastic_search_utils/build_elastic_search.py --index_name "wikidata_20210830_rm_disambiguation" --alias2qids_fn $BASE_DATA_DIR_SEMTAB21/wikidata/merged_alias_map/alias_map_rm_disambiguation.pkl
# Build index for alias mapping keeping disambiguation pages
python WikidataDumpProcessor/elastic_search_utils/build_elastic_search.py --index_name "wikidata_20210830_keep_disambiguation" --alias2qids_fn $BASE_DATA_DIR_SEMTAB21/wikidata/merged_alias_map/alias_map_keep_disambiguation.pkl 
```

## Experiments

### SemTab 21 datasets and convert csv to json formats
```shell
# download data
bash Benchmark/prepare_data/download_semtab21_data.sh
# convert csv to json
python Benchmark/prepare_data/semtab2021/convert_2T_table2json.py
python Benchmark/prepare_data/semtab2021/convert_AG_table2json.py
python Benchmark/prepare_data/semtab2021/convert_BioDivTab_table2json.py
python Benchmark/prepare_data/semtab2021/convert_BioTab_table2json.py
```

### Evaluation over the generated submission files
```shell
bash run_semtab21_eval.sh
```

### On-the-fly evaluation

### Start the redis service

- Entity meta-information (e.g., entity name, entity types, etc.) store
```shell
cd "${BASE_DATA_DIR_SEMTAB21}/Redis/Compress/EntityMetaInfo"
redis-server redis.windows.conf --port 7001
```

- Entity properties value information store
```shell
cd "${BASE_DATA_DIR_SEMTAB21}/Redis/Compress/PropertyValInfo"
redis-server redis.windows.conf --port 7002
```

- Entity Label information store
```shell
cd "${BASE_DATA_DIR_SEMTAB21}/Redis/EntityLabelInfo"
redis-server redis.windows.conf --port 7000
```

#### configs 


- Global config: /GlobalConfig/global_config.py

Set data store parameters
```python
ent_meta_redis_host, ent_meta_redis_port = "localhost", 7001
property_redis_host, property_redis_port = "localhost", 7002
ent_label_redis_host, ent_label_redis_port = "localhost", 7000
use_rocks_db = False
```

- Table annotator config: /TableAnnotator/Config/config.py
```python
# candidate generation
allow_int = True
keep_cell_filtering = False

kb_store = "redis" # other choices: "RAM" or "rocksdb"
candid_gen_method = "online" # other choices: "online". For quick experiments, choose "offline"
```


#### Run evaluation
```shell
bash run_semtab21.sh
```

#### Run multi-subject extension algorithm
```shell
bash run_multi_subject_biotab.sh
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
