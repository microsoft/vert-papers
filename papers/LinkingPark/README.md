# LinkingPark: An Automatic Semantic Table Interpretation System

This repository contains the open-sourced official implementation of LinkingPark, as described in the paper:

[LinkingPark: An Automatic Semantic Table Interpretation System](https://__) (2022).

If you find this repo helpful, please cite the following version of the paper (under submission, full information to be added upon acceptance):
```tex
@inproceedings{lp_system_chen_2022,
  author    = {Shuang Chen, Alperen Karaoglu, Carina Negreanu, Tingting Ma, Jin-Ge Yao, Jack Williams, Feng Jiang, Andy Gordon, Chin-Yew Lin},
  title     = {LinkingPark: An Automatic Semantic Table Interpretation System},
  year      = {2022},
}
```
or the software release itself:
```tex
@software{shuang_chen_2022_6496662,
  author       = {Shuang Chen, Alperen Karaoglu, Carina Negreanu, Börje F. Karlsson, Tingting Ma, Jin-Ge Yao, Jack Williams, Feng Jiang, Andry Gordon,               Chin-Yew Lin},
  title        = {{LinkingPark: Automatic Semantic Table Interpretation Software}},
  month        = apr,
  year         = 2022,
  note         = {{https://github.com/microsoft/vert-papers/tree/master/papers/LinkingPark}},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.6496662},
  url          = {https://doi.org/10.5281/zenodo.6496662}
}
```

A previous version of LinkingPark, which won 2nd place in the SemTab 2020 competition is described in:
```tex
@inproceedings{lp_v0_chen_2020,
  author    = {Shuang Chen, Alperen Karaoglu, Carina Negreanu, Tingting Ma, Jin-Ge Yao, Jack Williams, Andy Gordon, Chin-Yew Lin},
  title     = {LinkingPark: An integrated approach for Semantic Table Interpretation},
  year      = {2020},
  booktitle = {Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (SemTab 2020) at ISWC 2020},
}
```

For any questions/comments, please feel free to open GitHub issues.


## The LinkingPark system

LinkingPark is an automatic semantic annotation system for tabular data to knowledge graph matching. The system is designed as a modular framework which can handle Cell-Entity Annotation (CEA), Column-Type Annotation (CTA), and Columns-Property Annotation (CPA). LinkingPark has a number of desirable properties,
including a stand-alone architecture, flexibility for multilingual support, no dependence on labeled data, etc. 

The LP release in this repository utilizes [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) as its default backing knowledge base. For more details, please refer to the papers above.

## Configuring LinkingPark

### Download LinkingPark resources

Redis/RockDB dump files, and other necessary files to run the LinkingPark system/demo can be downloaded from [Azure storage](https://kcpapers.blob.core.windows.net/lp-2022/LP_Data.zip). Please extract the downloaded file to a local directory to be used in the system setup below. 

### Environment setup
```shell
export BASE_DATA_DIR="{path to the downloaded data}"
```

### Python environment
```
conda create --name lp_env python=3.9
conda activate lp_env
pip install -r requirements.txt
```

### Installing dependency services

LinkingPark leverages [RocksDB](http://rocksdb.org/) and [Redis](https://redis.io/) for data store, and [Elastic Search](https://www.elastic.co/) to conduct fuzzy matching.

#### RocksDB

According to the [guide](https://python-rocksdb.readthedocs.io/en/latest/installation.html).

```shell
apt-get install build-essential libsnappy-dev zlib1g-dev libbz2-dev libgflags-dev
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
mkdir build && cd build
cmake ..
make

export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:`pwd`/../include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`
export LIBRARY_PATH=${LIBRARY_PATH}:`pwd`

pip install git+git://github.com/twmht/python-rocksdb.git#egg=python-rocksdb
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
bash WikidataDumpProcessor/elastic_search_utils/setup_elastic_search.sh "wikidata_rm_disambiguation"
# Index for alias mapping keeping disambiguation pages
bash WikidataDumpProcessor/elastic_search_utils/setup_elastic_search.sh "wikidata_keep_disambiguation"
```

##### Build index

```
# Build index for alias mapping removing disambiguation pages
python WikidataDumpProcessor/elastic_search_utils/build_elastic_search.py --index_name "wikidata_rm_disambiguation" --alias2qids_fn $BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_rm_disambiguation.pkl
# Build index for alias mapping keeping disambiguation pages
python WikidataDumpProcessor/elastic_search_utils/build_elastic_search.py --index_name "wikidata_keep_disambiguation" --alias2qids_fn $BASE_DATA_DIR/wikidata/merged_alias_map/alias_map_keep_disambiguation.pkl 
```

#### Redis
```
wget https://download.redis.io/releases/redis-6.2.5.tar.gz
tar xzf redis-6.2.5.tar.gz
cd redis-6.2.5
make
```

## Experiments

### Download [SemTab 2020 datasets](https://zenodo.org/record/4282879#.Yme2nYVBxmM) and [Tough Tables datasets](https://zenodo.org/record/4246370#.Yme2xYVBxmM) and convert csv to json formats
```shell
# download data
bash Benchmark/prepare_data/download_semtab20_tt_data.sh
# convert csv to json
python Benchmark/prepare_data/convert_semtab20_csv_to_jsonl.py
python Benchmark/prepare_data/convert_tt_csv_to_jsonl.py
```

### Evaluation over the generated submission files
```shell
# full model
round=4 # switch to 1, 2, 3 if you want to evaluate other rounds results
sed -i 's/\r//g' run_eval.sh
bash run_eval.sh round${round}_results ${round}
```

|               | CEA-F1 (AG-Round4)   | CEA-Pr (AG-Round4) | CTA-F1 (AG-Round4)   | CTA-Pr (AG-Round4) | CPA-F1 (AG-Round4)   | CPA-Pr (AG-Round4) | CEA-F1 (TT)   | CEA-Pr (TT) | CTA-F1 (TT)   | CTA-Pr (TT) |
| :---          |    :----:            |               ---: |                 ---: | :---               |                 ---: | :---               |          ---: | :---        |          ---: | :---        |
| LinkingPark v1|           0.985      |       0.985        | 0.953                | 0.953              | 0.985                | 0.988              |    0.810      |   0.811     |   0.686       |   0.687     |
| LinkingPark v2|           0.988      |       0.988        | 0.972                | 0.972              | 0.995                | 0.995              |    0.908      |   0.908     |   0.784       |   0.784     |                                                     



#### Ablation over different features
```shell
# full model
sed -i 's/\r//g' run_eval.sh
# without row feature
bash run_eval.sh rounnd4_ablation_wo_row 4
# without col feature
bash run_eval.sh round4_ablation_wo_col 4
# without popularity feature
bash run_eval.sh round4_ablation_wo_popularity 4
# without lexical feature
bash run_eval.sh round4_ablation_wo_lexical 4
```
|                                   | CEA-F1 (AG-Round4)   | CEA-Pr (AG-Round4) | CEA-F1 (TT)   | CEA-Pr (TT) |
| :---                              |    :----:            |               ---: |          ---: | :---        |
| LinkingPark v2 (full)             |           0.988      |       0.988        | 0.908         | 0.908       |
| LinkingPark v2 (w/o row score)    |           0.948      |       0.948        | 0.738         | 0.738       |
| LinkingPark v2 (w/o col. score)   |           0.987      |       0.987        | 0.896         | 0.896       |
| LinkingPark v2 (w/o lexical)      |           0.986      |       0.986        | 0.898         | 0.898       |
| LinkingPark v2 (w/o popularity)   |           0.988      |       0.988        | 0.883         | 0.883       |


#### Ablation over different candidate generation method
```shell
sed -i 's/\r//g' run_eval.sh
bash run_eval.sh reproduce_round4_ds_only 4
```
|                                   | CEA-F1 (AG-Round4)   | CEA-Pr (AG-Round4) | CEA-F1 (TT)   | CEA-Pr (TT) |
| :---                              |    :----:            |               ---: |          ---: | :---        |
| LinkingPark v2 (DS + FS)          |           0.988      |       0.988        | 0.908         | 0.908       |
| LinkingPark v2 (DS-only)          |           0.985      |       0.989        | 0.809         | 0.897       |

### On-the-fly evaluation

#### configs 


- Global config: /GlobalConfig/global_config.py

Set data store parameters
```python
rocksdb_meta_info_path = "${BASE_DATA_DIR}/RocksDB/Compress/EntityMetaInfo/data.db"
rocksdb_property_val_info_path = "${BASE_DATA_DIR}/RocksDB/Compress/PropertyValueInfo/data.db"
ent_meta_redis_host, ent_meta_redis_port = "localhost", 6390
property_redis_host, property_redis_port = "localhost", 6400
ent_label_redis_host, ent_label_redis_port = "localhost", 6382
use_rocks_db = True
```

#### Offline candidate generation
```shell
# DS + FS
#python Benchmark/candid_gen/all_in_one.py --round 4 --method _ds_es
# DS only
python Benchmark/candid_gen/all_in_one.py --round 4 --method _ds_only --dictionary_only True
```

- Table annotator config: /TableAnnotator/Config/config.py
```python
# candidate generation
allow_int = True
keep_cell_filtering = False

kb_store = "redis" # other choices: "RAM" or "rocksdb"
candid_gen_method = "offline" # other choices: "online". For quick experiments, choose "offline"
data_round=4 # evaluation round
candid="alias_map_ds_only" # candidate generation method; DS only
# candid="alias_map_ds_es"       # candidate generation method; DS + FS
candid_map_fn = os.path.join(kb_dir, f"Round{data_round}", f"all_in_one/{candid}.pkl") # used if candid_gen_method == "offline"
```


#### Run evaluation
```shell
sed -i 's/\r//g' run.sh
bash run.sh exp_name 4
```



## Start RESTful API

### Start the redis service

- Entity meta-information (e.g., entity name, entity types, etc.) store 
```shell
cd "${BASE_DATA_DIR}/Redis/Compress/EntityMetaInfo"
redis-server redis.windows.conf --port 6390
```

- Entity properties value information store
```shell
cd "${BASE_DATA_DIR}/Redis/Compress/PropertyValInfo"
redis-server redis.windows.conf --port 6400
```

- Entity Label information store
```shell
cd "${BASE_DATA_DIR}/Redis/EntityLabelInfo"
redis-server redis.windows.conf --port 6382
```

### Start Elasticsearch service
```shell
/path_to_elasticsearch_installation/elasticsearch-7.14.1/bin/elasticsearch
```

### Start the demo
```
python lp_service.py --port 6009
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
