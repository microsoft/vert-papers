# CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition

This repository contains the open-sourced official implementation of the paper:

[CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition](https://arxiv.org/abs/1904.02141) (NAACL-HLT 2019).  
_Yuying Zhu, Guoxin Wang, Börje F. Karlsson_

If you find this repo helpful, please cite either of the following versions of the paper:

```tex
@article{DBLP:journals/corr/abs-1904-02141,
  author    = {Yuying Zhu and Guoxin Wang and B{\"{o}}rje F. Karlsson},
  title     = {{CAN-NER:} Convolutional Attention Network for Chinese Named Entity Recognition},
  journal   = {CoRR},
  volume    = {abs/1904.02141},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.02141},
}
```

```tex
@inproceedings{DBLP:conf/naacl/ZhuW19,
  author    = {Yuying Zhu and Guoxin Wang and Börje F. Karlsson},
  title     = {{CAN-NER:} Convolutional Attention Network for Chinese Named Entity Recognition},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2019},
  pages     = {3384--3393},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
}
```

For any questions/comments, please feel free to open GitHub issues.


## 🎥 Overview

In this paper, we investigate a **Convolutional Attention Network (CAN)** for Chinese NER, which consists of a character-based convolutional neural network (CNN) with local-attention layer
and a gated recurrent unit (GRU) with global self-attention layer to capture the information from adjacent characters and sentence contexts.
CAN-NER **does not** depend on any external resources like lexicons and employing small-size char embeddings makes CAN-NER more practical for real systems scenarios. 

<img src="images/model_structure.png" width = "400" height = "600" alt="model" align=center />


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Arguments
All command arguments are list as following:

```
usage: train.py [-h] [--model_name MODEL_NAME]
                [--data_name {weibo,MSRA,onto4}]
                [--train_data_path TRAIN_DATA_PATH]
                [--test_data_path TEST_DATA_PATH]
                [--dev_data_path DEV_DATA_PATH]
                [--pretrained_embed_path PRETRAINED_EMBED_PATH]
                [--result_folder RESULT_FOLDER] [--seed SEED]
                [--batch_size BATCH_SIZE] [--epoch EPOCH] [--lr LR]
                [--dropout DROPOUT] [--hidden_dim HIDDEN_DIM]
                [--window_size WINDOW_SIZE] [--is_parallel]

CAN-NER Model

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        give the model a name.
  --data_name {weibo,MSRA,onto4}
                        name for dataset.
  --train_data_path TRAIN_DATA_PATH
                        file path for train set.
  --test_data_path TEST_DATA_PATH
                        file path for test set.
  --dev_data_path DEV_DATA_PATH
                        file path for dev set.
  --pretrained_embed_path PRETRAINED_EMBED_PATH
                        path for embedding.
  --result_folder RESULT_FOLDER
                        folder path for save models and results.
  --seed SEED           seed for everything
  --batch_size BATCH_SIZE
  --epoch EPOCH         epoch number
  --lr LR               learning rate
  --dropout DROPOUT     dropout rate
  --hidden_dim HIDDEN_DIM
                        hidden dimension
  --window_size WINDOW_SIZE
                        window size for acnn
  --is_parallel         whether to use multiple gpu
```

## Train

To train the model(s) in the paper, run this command:

```train
python train.py
--model_name <demo_model> \
--data_name MSRA \
--train_data_path <path_to_data> \
--test_data_path <path_to_data> \
--dev_data_path <path_to_data> \
--pretrained_embed_path <path_to_data> \
--result_folder <folder_name>
```

For hyperparameter configuration, we adjust them according to the performance on the described development sets for Chinese NER. We set the character embedding size, hidden sizes of CNN and BiGRU to 300 dims. After comparing experimental results with different CNN window sizes, we set the window size as 5. Adadelta is used for optimization, with an initial learning rate of 0.005. 


## 🍯 Datasets

We use the following four widely-used benchmark datasets in our experiments:

- OntoNotes 4: [(Weischedel et al., 2011)](https://catalog.ldc.upenn.edu/LDC2011T03);
- MSRA Chinese NER, from SIGHAN Bakeoff 2006: [Levow et al. 2006](https://www.aclweb.org/anthology/W06-0115/);
- Weibo corpus: [(Peng and Dredze (2015))](https://www.aclweb.org/anthology/D15-1064.pdf), which is extracted from the [Sina Weibo](https://www.weibo.com/) social network site;
- Chinese Resume corpus: [(Zhang and Yang, 2018)](https://arxiv.org/abs/1805.02023) collected from [Sina Finance](https://finance.sina.com.cn/stock/), for more domain variety.


## 📋 Results

Our model achieves the following performance results on the four datasets:

* Chinese Resume dataset

| Model                                                                    | P          | R          | F1         | 
| ------------------------------------------------------------------------ | ---------- | ---------- | ---------- |
| [Zhang and Yang [2018]1](https://www.aclweb.org/anthology/P18-1144/)     | 94.53      | 94.29      | 94.41      | 
| [Zhang and Yang [2018]2](https://www.aclweb.org/anthology/P18-1144/)     | 94.07      | 94.42      | 94.24      | 
| [Zhang and Yang [2018]3](https://www.aclweb.org/anthology/P18-1144/)     | 94.81      | 94.11      | 94.46      | 
| Baseline                                                                 | 93.71      | 93.74      | 93.73      | 
| Baseline + CNN                                                           | 94.36      | **94.85**  | 94.60      | 
| **CAN-NER**                                                              | **95.05**  | **94.82*   | **94.94**  | 

1 represents the char-based LSTM model, 2 indicates the word-based LSTM model, and 3 is their Lattice model.

* Weibo dataset

| Model                                                                             | NE         | NM         | Overall    | 
| --------------------------------------------------------------------------------- | ---------- | ---------- | ---------- |
| [Peng and Dredze [2015]](https://www.aclweb.org/anthology/D15-1064/)              | 51.96      | 61.05      | 56.05      | 
| [Peng and Dredze [2016]](https://www.aclweb.org/anthology/P16-2025/)              | 55.28      | **62.97**  | 58.99      | 
| [He and Sun [2017]a](https://www.aclweb.org/anthology/E17-2113/)                  | 50.60      | 59.32      | 54.82      | 
| [He and Sun [2017]b](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14484) | 54.50      | 62.17      | 58.23      | 
| [Cao et al. [2018]](https://www.aclweb.org/anthology/P18-1144/)                   | 54.34      | 57.35      | 58.70      | 
| [Zhang and Yang [2018]](https://www.aclweb.org/anthology/P18-1144/)               | 53.04      | 62.25      | 58.79      | 
| Baseline                                                                          | 49.02      | 58.80      | 53.80      | 
| Baseline + CNN                                                                    | 53.86      | 58.05      | 55.91      | 
| **CAN-NER**                                                                       | **55.38**  | **62.98**  | **59.31**  | 

* OntoNotes 4 dataset

| Model                                                                                      | P          | R          | F1         | 
| ------------------------------------------------------------------------------------------ | ---------- | ---------- | ---------- |
| [Yang et al. [2016]](https://arxiv.org/abs/1708.07279)                                     | 65.59      | 71.84      | 68.57      | 
| [Yang et al. [2016]](https://arxiv.org/abs/1708.07279)1                                    | 72.98      | **80.15**  | **76.40**  | 
| [Che et al. [2013]](https://www.aclweb.org/anthology/N13-1006/)1                           | **77.71**  | 72.51      | 75.02      | 
| [Wang et al. [2013]](https://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/viewPaper/6346)1 | 76.43      | 72.32      | 74.32      | 
| [Zhang and Yang [2018]](https://www.aclweb.org/anthology/P18-1144/)2                       | 76.35      | 71.56      | 73.88      | 
| [Zhang and Yang [2018]](https://www.aclweb.org/anthology/P18-1144/)3                       | 74.36      | 69.43      | 71.81      | 
| Baseline                                                                                   | 70.67      | 71.64      | 71.15      | 
| Baseline + CNN                                                                             | 72.69      | 71.51      | 72.10      | 
| **CAN-NER**                                                                                | *75.05*    | *72.29*    | *73.64*    | 

1 denotes models with external labeled data for semi-supervised learning. 2 denotes models using external lexicon data. 3 is the char-based model in that paper.

* MSRA dataset

| Model                                                                                      | P          | R          | F1         | 
| ------------------------------------------------------------------------------------------ | ---------- | ---------- | ---------- |
| [Chen et al. [2006]](https://www.aclweb.org/anthology/W06-0130/)                           | 91.22      | 81.71      | 86.20      | 
| [Zhang et al. [2006]](https://www.aclweb.org/anthology/W06-0126/)1                         | 92.20      | 90.18      | 91.18      | 
| [Zhou et al. [2013]](http://www.ejournal.org.cn/Jweb_cje/CN/abstract/abstract7635.shtml)   | 91.86      | 88.75      | 90.28      | 
| [Lu et al. [2016]](https://www.aclweb.org/anthology/L16-1138/)                             | -          | -          | 87.94      | 
| [Dong et al. [2016]](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_20)       | 91.28      | 90.62      | 90.95      | 
| [Cao et al. [2018]](https://www.aclweb.org/anthology/P18-1144/)                            | 91.30      | 89.58      | 90.64      | 
| [Yang et al. [2018]](https://link.springer.com/chapter/10.1007/978-3-319-99495-6_16)       | 92.04      | 91.31      | 91.67      | 
| [Zhang and Yang [2018]](https://www.aclweb.org/anthology/P18-1144/)2                       | **93.57**  | **92.79**  | **93.18**  | 
| Baseline                                                                                   | 92.54      | 88.20      | 90.32      | 
| Baseline + CNN                                                                             | 92.57      | 92.11      | 92.34      | 
| **CAN-NER**                                                                                | **93.53**  | *92.42*    | **92.97**  | 

1 denotes models with external labeled data for semi-supervised learning. 2 denotes models using external lexicon data.


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
