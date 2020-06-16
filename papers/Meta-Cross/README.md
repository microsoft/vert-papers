# Enhanced Meta-Learning for Cross-lingual Named Entity Recognition with Minimal Resources

This repository is the official implementation of

[Enhanced Meta-Learning for Cross-lingual Named Entity Recognition with Minimal Resources](https://www.microsoft.com/en-us/research/publication/enhanced-meta-learning-for-cross-lingual-named-entity-recognition-with-minimal-resources/) (AAAI 2020).  
_Qianhui Wu, Zijia Lin, Guoxin Wang, Hui Chen, Börje F. Karlsson, Biqing Huang, Chin-Yew Lin_

If you find this repo helpful, please cite the following:

```tex
@article{wu2019metacross,
    title={Enhanced Meta-Learning for Cross-lingual Named Entity Recognition with Minimal Resources},
    author={Wu, Qianhui and Lin, Zijia and Wang, Guoxin and Chen, Hui and Karlsson, Börje and Huang, Biqing and Lin, Chin-Yew},
    year={2020},
    month={February},
    journal={AAAI 2020},
    url={https://www.microsoft.com/en-us/research/publication/enhanced-meta-learning-for-cross-lingual-named-entity-recognition-with-minimal-resources/},
}
```

For any question, please feel free to post Github issues.

## 🎥 Overview

In this paper, we propose an **enhanced meta-learning algorithm for cross-lingual NER** with minimal resources, considering that the model could achieve better results after a few ﬁne-tuning steps over a very limited set of structurally/semantically similar examples from the source language.
To this end, we propose to construct multiple pseudoNER tasks for meta-training by computing sentence similarities.
Moreover, in order to improve the model’s capability to transfer across different languages, we present a masking scheme and augment the loss function with an additional maximum term during meta-training.
Experiments on ﬁve target languages show that the proposed approach leads to new state-of-the-art results with a relative F1-score improvement of up to 8.76%.
We also extend the approach to low-resource cross-lingual NER, and it also achieves state-of-the-art results.

## 🎯 Quick Start

### Requirements

- python 3.7
- pytorch 1.0.0
- pytorch_pretrained_bert

The code may work on other python and pytorch version.
However, we ran experiments in the above environment.

## Data Pre-processing

Prepare the preprocessed data to `data/${language}`.
We use the BIO tagging scheme and NER data in the following format:

```txt
Peter B-PER
Blackburn I-PER
BRUSSELS B-LOC
1996-08-22 O
```

We take English as the source language in our approach.
There is supposed to have two files: `data/en/train.txt` and `data/en/valid.txt`.
`train.txt` is used for model training and `valid.txt` is used for model selection.

In principle, we can transfer knowledge from the source language to any other target languages.
For instance, if you take Spanish as the target language, for `zero-shot`, there is supposed to have `data/es/test.txt` wihle for `k-shot`, there is supposed to have `data/es/train.txt` and `data/es/test.txt`.
`data/es/train.txt` is the k-shot training data for fine-tuning.

**Note: all the data should be with the same label types, for example, `[PER, LOC, ORG, MISC, O]` or `[PER, LOC, ORG, O]`.**  
**Before training the model, set the `LABEL_LIST` in `perprocessor.py`:**

```python
LABEL_LIST = [LABEL_1, LABEL_2, ..., LABEL_N, "X", "[CLS]", "[SEP]"]
# for example:
# LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
```

## Train and Evaluate

Just run `scripts/run.bat`(for _Windows_) or `scripts/run.sh`(For _Linux_).

The trained model and the results are saved to `models/result_dir`.

## 🍯 Datasets

We use the following widely-used benchmark datasets for experiments:

- CoNLL-2002 [Tjong Kim Sang, 2002](https://www.aclweb.org/anthology/W02-2024/) for Spanish [es] and Dutch [nl] NER;
- CoNLL-2003 [Tjong Kim Sang and De Meulder, 2003](https://www.aclweb.org/anthology/W03-0419/) for English [en] and German [de] NER;
- Europeana Newspapers French NER [Neudecker, 2016](https://www.aclweb.org/anthology/L16-1689/) for French [fr];
- MSRA Chinese NER [Levow et al. 2006](https://www.aclweb.org/anthology/W06-0115/) for Chinese [zh].

All datasets are annotated with 4 entity types: LOC, MISC, ORG, and PER. Each dataset is split into training, dev, and test sets.

All datasets are CoNLL-style and BIO tagging scheme.
In this repo, we only public some sample of this corpus.
You can download they from the websites [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner.tgz), [CoNLL-2002](http://www.cnts.ua.ac.be/conll2002/ner.tgz), [Europeana Newspapers French NER](https://github.com/EuropeanaNewspapers/ner-corpora).
And put they to the file path `data/${language}/${train_type}.txt`.

## 📋 Results

We reports the zero-resource cross-lingual NER results of the proposed UniTrans on the 5 target languages, alongside those reported by prior state-of-the-art methods.

|                                                                          | es        | nl        | de        | fr        | zh        | Average   |
| ------------------------------------------------------------------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| [Tackstrom _et_ _al_.[2012]](https://www.aclweb.org/anthology/N12-1052/) | 59.30     | 58.40     | 40.40     | -         | -         | -         |
| [Tsai _et_ _al_.[2016]](https://www.aclweb.org/anthology/K16-1022/)      | 60.55     | 61.56     | 48.12     | -         | -         | -         |
| [Ni _et_ _al_.[2017]](https://www.aclweb.org/anthology/P17-1135/)        | 65.10     | 65.40     | 58.50     | -         | -         | -         |
| [Mayhew _et_ _al_.[2017]](https://www.aclweb.org/anthology/D17-1269/)    | 65.95     | 66.50     | 59.11     | -         | -         | -         |
| [Xie _et_ _al_.[2018]](https://www.aclweb.org/anthology/D18-1034/)       | 72.37     | 71.25     | 57.76     | -         | -         | -         |
| [Wu and Dredze [2019]](https://www.aclweb.org/anthology/D19-1077/)       | 74.96     | 77.57     | 69.56     | -         | -         | -         |
| Base Model                                                               | 74.59     | 79.57     | 70.79     | 50.89     | 76.42     | 70.45     |
| **Meta-Cross**                                                           | **76.75** | **80.44** | **73.16** | **55.30** | **77.89** | **72.71** |

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
