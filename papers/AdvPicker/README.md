# AdvPicker: Effectively Leveraging Unlabeled Data via Adversarial Discriminator for Cross-Lingual NER

This repository will contain the open-sourced official implementation of the paper:

[AdvPicker: Effectively Leveraging Unlabeled Data via Adversarial Discriminator for Cross-Lingual NER](https://aclanthology.org/2021.acl-long.61) (ACL-IJCNLP 2021).

_Weile Chen, Huiqiang Jiang, Qianhui Wu, Börje F. Karlsson, and Yi Guan_

If you find this repo helpful, please cite the following paper:

```bibtex
@inproceedings{chen2021advpicker,
    title="{A}dv{P}icker: {E}ffectively {L}everaging {U}nlabeled {D}ata via {A}dversarial {D}iscriminator for {C}ross-{L}ingual {NER}",
    author={Weile Chen and Huiqiang Jiang and Qianhui Wu and B{\"{o}}rje F. Karlsson and Yi Guan},
    year={2021},
    month={aug},
    booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
    publisher={Association for Computational Linguistics},
    url={https://aclanthology.org/2021.acl-long.61},
    doi={10.18653/v1/2021.acl-long.61},
    pages={743--753},
}
```

For any questions/comments, please feel free to open GitHub issues.

## 🎥 Overview

In this paper, we propose a novel adversarial approach (AdvPicker) to better leverage such data and further improve results in zero-shot cross-lingual NER task. We design an adversarial learning framework in which an encoder learns entity domain knowledge from labeled source-language data and better shared features are captured via adversarial training - where a discriminator selects less language-dependent target-language data via similarity to the source language. Experimental results on standard benchmark datasets well demonstrate that the proposed method beneﬁts strongly from this data selection process and outperforms existing state-of-the-art methods; without requiring any additional external resources (e.g., gazetteers or via machine translation).

![image](./images/framework-v5.png)

## 🎯 Quick Start

### Requirements

- python 3.6.9
- pytorch 1.9.0
- [HuggingFace Transformers 3.2.0](https://github.com/huggingface/transformers)

Other pip package show in `requirements.txt`.

```bash
pip3 install -r requirements.txt
```

The code may work on other python and pytorch version. However, all experiments were run in the above environment.

### Train and Evaluate

For _Linux_ machines,

```bash
bash scripts/run.sh
```

For _Windows_ machines,

```cmd
call scripts\run.bat
```

If you only want to predict in the trained model, you can get the model from [OneDrive](https://microsoftapc-my.sharepoint.com/:f:/g/personal/hjiang_microsoft_com/ErS3_cs5aQ5KjM-TvqqiM4UB-vEgemCwLFyQvcvz6_NChw?e=lobLbv).
To easier to use and download, we only put one seed model in that.
If you need all of models, you can download them from [OneDrive](https://microsoftapc-my.sharepoint.com/:f:/g/personal/hjiang_microsoft_com/Eloj7PaWT-VAqyP3D4B7GB8BvKIAJ-OlU1gs2wxUq_o2Fg?e=YZbd3H).
Put the `result` folder in `papers/AdvPicker` directory.

For _Linux_ machines,

```bash
bash scripts/run_pred.sh
```

For _Windows_ machines,

```cmd
call scripts\run_pred.bat
```

## 🍯 Datasets

We use the following widely-used benchmark datasets for the experiments:

- CoNLL-2002 [Tjong Kim Sang, 2002](https://www.aclweb.org/anthology/W02-2024/) for Spanish [es] and Dutch [nl] NER;
- CoNLL-2003 [Tjong Kim Sang and De Meulder, 2003](https://www.aclweb.org/anthology/W03-0419/) for English [en] and German [de] NER;

All datasets are annotated with 4 entity types: LOC, MISC, ORG, and PER. Each dataset is split into training, dev, and test sets.

All datasets are CoNLL-style and BIO tagging scheme.In this repo, we only publish a small data sample to validate the code. You can download them from their respective websites: [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner.tgz), [CoNLL-2002](http://www.cnts.ua.ac.be/conll2002/ner.tgz), and [NoDaLiDa-2019](https://github.com/ljos/navnkjenner).
And place them in the correct locations: `data/${language}/${train_type}.txt`.

## 📋 Results

We report the zero-resource cross-lingual NER results of the proposed UniTrans on the 3 target languages, alongside those reported by prior state-of-the-art methods and those of two re-implemented baseline methods, i.e., mBERT-ft and mBERT-TLADV.

|                                                                                   | es               | nl               | de               | Average          |
| --------------------------------------------------------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| [Tackstrom _et_ _al_.[2012]](https://www.aclweb.org/anthology/N12-1052/)          | 59.30            | 58.40            | 40.40            | 52.70            |
| [Tsai _et_ _al_.[2016]](https://www.aclweb.org/anthology/K16-1022/)               | 60.55            | 61.56            | 48.12            | 56.74            |
| [Ni _et_ _al_.[2017]](https://www.aclweb.org/anthology/P17-1135/)                 | 65.10            | 65.40            | 58.50            | 63.00            |
| [Mayhew _et_ _al_.[2017]](https://www.aclweb.org/anthology/D17-1269/)             | 65.95            | 66.50            | 59.11            | 61.57            |
| [Xie _et_ _al_.[2018]](https://www.aclweb.org/anthology/D18-1034/)                | 72.37            | 71.25            | 57.76            | 67.13            |
| [Jain _et_ _al_.[2019]](https://www.aclweb.org/anthology/D19-1100/)               | 73.5             | 69.9             | 61.5             | 68.30            |
| [Bari _et_ _al_.[2020]](https://arxiv.org/abs/1911.09812)                         | 75.93            | 74.61            | 65.24            | 71.93            |
| [Wu and Dredze [2019]](https://www.aclweb.org/anthology/D19-1077/)                | 74.96            | 77.57            | 69.56            | 73.57            |
| [Keung _et_ _al_. [2019]](https://www.aclweb.org/anthology/D19-1138)              | 74.3             | 77.6             | 71.9             | 74.60            |
| [Wu _et_ _al_.[2020a]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-WuQ.5015.pdf) | 76.75            | 80.44            | 73.16            | 76.78            |
| [Wu _et_ _al_.[2020b]\*](https://www.ijcai.org/proceedings/2020/543)              | 77.30 ± 0.78     | 81.20 ± 0.83     | 73.61 ± 0.39     | 77.37 ± 0.67     |
| mBERT-ft                                                                          | 75.12 ± 0.83     | 80.34 ± 0.27     | 72.59 ± 0.31     | 76.02 ± 0.47     |
| mBERT-TLADV                                                                       | 76.92 ± 0.62     | 80.62 ± 0.56     | 73.89 ± 0.56     | 77.14 ± 0.58     |
| **AdvPicker**                                                                     | **79.00** ± 0.21 | **82.90** ± 0.44 | **75.01** ± 0.50 | **78.97** ± 0.38 |

PS: \* denotes the version of the method without additional data.

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
