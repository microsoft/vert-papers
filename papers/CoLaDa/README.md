# CoLaDa: A Collaborative Label Denoising Framework for Cross-lingual Named Entity Recognition

This repository contains the open-sourced official implementation of the paper

[CoLaDa: A Collaborative Label Denoising Framework for Cross-lingual Named Entity Recognition](https://arxiv.org/abs/2305.14913) (ACL 2023).  
_Tingting Ma, Qianhui Wu, Huiqiang Jiang, Börje F. Karlsson, Tiejun Zhao and Chin-Yew Lin_

If you find this repo helpful, please cite the following paper

```tex
@inproceedings{ma-etal-2023-colada,
    title = {CoLaDa: A Collaborative Label Denoising Framework for Cross-lingual Named Entity Recognition},
    author = {Tingting Ma and Qianhui Wu and Huiqiang Jiang and B{\"{o}}rje F. Karlsson and Tiejun Zhao and Chin-Yew Lin},
    booktitle = {Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (ACL 2023)},
    month = {jul},
    year = {2023},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2023.acl-long.330/},
    doi = {10.18653/v1/2023.acl-long.330},
    pages = {5995--6009},
}
```

For any questions/comments, please feel free to open GitHub issues or email the <a href="mailto:hittingtingma@gmail.com">fist author<a/> directly.

## Requirements
    
The code requires  
```
python >= 3.9.15  
torch == 1.7.1  
faiss >= 1.7.1
transformers == 4.21.1
tokenizers == 0.12.1
seqeval == 1.2.2
```

## Prepare dataset

The data for CoNLL benchmark need to putted in data/conll-lingual folder. The files are structured as follows: 

```
data/
    conll-lingual/
        en/
            train.txt
            dev.txt
            test.txt
            word-trans/
                trains.train.de.conll
                trains.train.es.conll
                trains.train.nl.conll
        de/
            train.txt
            dev.txt
            test.txt
```

You can prepare the translation data for CoNLL as [UniTrans](https://github.com/microsoft/vert-papers/tree/master/papers/UniTrans).

To prepare translation data for WikiAnn, you can follow the [CROP](https://github.com/YuweiYin/CROP):
1. prepare marker-inserted English data by running script ./m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py
2. translate the English data to target language using fairseq with the released m2m_checkpoint_insert_avg_41_60.pt checkpoint.
3. convert the inserted data in target language to CoNLL format using ./m2m/scripts/pattern/m2m/ner/prepare_ner_data_from_pattern_sentence_step1.py

After all, deduplicate the data for KNN as utils/preprocess.py

## CoLaDa Training

train an NER model in source-language data 

```bash
bash scripts/srcmodel-train.sh
```

train the final model

```bash
bash scripts/colada-train.sh
```

You can change the target langauge by setting the 'tgt' parameter in the script.


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit httpscla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](httpsopensource.microsoft.comcodeofconduct).
For more information see the [Code of Conduct FAQ](httpsopensource.microsoft.comcodeofconductfaq) or
contact [opencode@microsoft.com](mailtoopencode@microsoft.com) with any additional questions or comments.
