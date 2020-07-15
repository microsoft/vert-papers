# UniTrans: Unifying Model Transfer and Data Transfer for Cross-Lingual Named Entity Recognition with Unlabeled Data

This repository contains the open-sourced official implementation of the paper:

[UniTrans: Unifying Model Transfer and Data Transfer for Cross-Lingual Named Entity Recognition with Unlabeled Data](https://www.microsoft.com/en-us/research/publication/unitrans-unifying-model-transfer-and-data-transfer-for-cross-lingual-named-entity-recognition-with-unlabeled-data/) (IJCAI 2020).  
_Qianhui Wu, Zijia Lin, Börje Karlsson, Biqing Huang and Jian-Guang Lou_

If you find this repo helpful, please cite the following paper:

```tex
@inproceedings{wu2020unitrans,
    title={UniTrans: Unifying Model Transfer and Data Transfer for Cross-Lingual Named Entity Recognition with Unlabeled Data},
    author={Qianhui Wu and Zijia Lin and B{\"{o}}rje F. Karlsson and Biqing Huang and Jian-Guang Lou},
    year={2020},
    booktitle={29th International Joint Conference on Artificial Intelligence (IJCAI 2020)},
    url={https://www.microsoft.com/en-us/research/publication/unitrans-unifying-model-transfer-and-data-transfer-for-cross-lingual-named-entity-recognition-with-unlabeled-data/},
}
```

For any questions/comments, please feel free to open GitHub issues.


## 🎥 Overview

In this paper, we propose a novel approach for cross-lingual NER termed UniTrans, which uniﬁes both model-transfer and data-transfer approaches based on their complementarity, and further leverages beneﬁcial information from unlabeled target-language data via knowledge distillation.
We also propose a voting scheme to generate pseudo hard labels for part of words in the unlabeled target-language data, so as to enhance knowledge distillation with supervision from both hard and soft labels. We evaluate the proposed UniTrans on benchmark datasets for four target languages.
Experimental results show that UniTrans achieves new state-of-the-art performance for all target languages.
We also extend UniTrans with teacher ensembling, which leads to further performance gains.

![image](https://cdn.nlark.com/yuque/0/2020/png/104214/1591779090520-ae3107ed-97a3-4d35-8173-40aa469141a7.png)


## 🎯 Quick Start

### Requirements

- python 3.7
- pytorch 1.0.0
- [HuggingFace PyTorch Pretrained mBERT](https://github.com/huggingface/pytorch-transformers.git) (2019.10.27)
- [MUSE](https://github.com/facebookresearch/MUSE.git) for translation: (2019.12.17)

Other pip package show in `requirements.txt`.

```bash
pip3 install -r requirements.txt
```

The code may work on other python and pytorch version. However, all experiments were run in the above environment.


### Train and Evaluate

For _Linux_ machines,

```bash
# Generation tgt translated datasets from src dataset
bash scripts/run_translate.sh

# Fine-tune mBERT in src dataset
bash scripts/run_src.sh

# Train & Evaluate in tgt dataset
bash scripts/run_tgt.sh
```

For _Windows_ machines,

```cmd
<!--  Generation tgt translated datasets from src dataset  -->
call scripts\run_translate.bat

<!--  Fine-tune mBERT in src dataset  -->
call scripts\run_src.bat

<!--  Train & Evaluate in tgt dataset  -->
call scripts\run_tgt.bat
```

Before generating the translation dataset, you need to have the embeddings file and dictionaries following these steps:

1. Translate source-language labeled data.
2. Download [monolingual word embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html) `wiki.language.vec` to `data/embedding`.
3. Download [src-tgt test dictionaries](https://github.com/facebookresearch/MUSE) for evaluation (English to other languages) to `data/dict`.


## 🍯 Datasets

We use the following widely-used benchmark datasets for the experiments:

- CoNLL-2002 [Tjong Kim Sang, 2002](https://www.aclweb.org/anthology/W02-2024/) for Spanish [es] and Dutch [nl] NER;
- CoNLL-2003 [Tjong Kim Sang and De Meulder, 2003](https://www.aclweb.org/anthology/W03-0419/) for English [en] and German [de] NER;
- NoDaLiDa-2019 [Johansen, 2019](https://www.aclweb.org/anthology/W19-6123/) for Norwegian [no] NER (We use Bokm˚al rather than Nynorsk here, considering that it is used by 85–90% of the population of Norway.).

All datasets are annotated with 4 entity types: LOC, MISC, ORG, and PER. Each dataset is split into training, dev, and test sets.

All datasets are CoNLL-style and BIO tagging scheme.In this repo, we only publish a small data sample to validate the code. You can download them from their respective websites: [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner.tgz), [CoNLL-2002](http://www.cnts.ua.ac.be/conll2002/ner.tgz), and [NoDaLiDa-2019](https://github.com/ljos/navnkjenner).
And place them in the correct locations: `data/conll/ner/${language}/${train_type}.txt`.


## 📋 Results

We report the zero-resource cross-lingual NER results of the proposed UniTrans on the 4 target languages, alongside those reported by prior state-of-the-art methods and those of two re-implemented baseline methods, i.e., Model Transfer and Data Transfer.

|                                                                                  | es        | nl        | de        | no        | Average   |
| -------------------------------------------------------------------------------- | --------- | --------- | --------- | --------- | --------- |
| [Tackstrom _et_ _al_.[2012]](https://www.aclweb.org/anthology/N12-1052/)         | 59.30     | 58.40     | 40.40     | -         | -         |
| [Tsai _et_ _al_.[2016]](https://www.aclweb.org/anthology/K16-1022/)              | 60.55     | 61.56     | 48.12     | -         | -         |
| [Ni _et_ _al_.[2017]](https://www.aclweb.org/anthology/P17-1135/)                | 65.10     | 65.40     | 58.50     | -         | -         |
| [Mayhew _et_ _al_.[2017]](https://www.aclweb.org/anthology/D17-1269/)            | 65.95     | 66.50     | 59.11     | -         | -         |
| [Xie _et_ _al_.[2018]](https://www.aclweb.org/anthology/D18-1034/)               | 72.37     | 71.25     | 57.76     | -         | -         |
| [Jain _et_ _al_.[2019]](https://www.aclweb.org/anthology/D19-1100/)              | 73.5      | 69.9      | 61.5      | -         | -         |
| [Bari _et_ _al_.[2019]](https://arxiv.org/abs/1911.09812)                        | 75.93     | 74.61     | 65.24     | -         | -         |
| [Wu and Dredze [2019]](https://www.aclweb.org/anthology/D19-1077/)               | 74.96     | 77.57     | 69.56     | -         | -         |
| [Wu _et_ _al_.[2019]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-WuQ.5015.pdf) | 76.75     | 80.44     | 73.16     | -         | -         |
| Model Transfer (reimp.)                                                          | 76.34     | 80.61     | 72.39     | 78.47     | 76.95     |
| Data Transfer (reimp.)                                                           | 78.14     | 80.98     | 73.65     | 78.91     | 77.92     |
| **UniTrans**                                                                     | **79.31** | **82.90** | **74.82** | **81.17** | **79.55** |


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
