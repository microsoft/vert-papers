This repository contains code, datasets, and links related to entity/knowledge papers from the **VERT** (**V**ersatile **E**ntity **R**ecognition & Disambiguation **T**oolkit) project, by the [Knowledge Computing (**KC**)](https://www.microsoft.com/en-us/research/group/knowledge-computing/) group at Microsoft Research Asia (MSRA).

Our group is hiring both research interns and full-time employees! If you are interest, please take a look at:
* [Internship opportunities in KC](https://www.microsoft.com/en-us/research/uploads/prod/2020/12/kc_intern_job_description_2020.pdf) (PDF);
* [Researcher or RSDE positions](https://careers.microsoft.com/professionals/us/en/search-results?rk=l-c-research) and select "China" on the left-side "Country/Region" menu.

# News:

* 2023-May: Two papers accepted by ACL'23, including [MLKD OOD](https://github.com/microsoft/KC/tree/main/papers/MLKD_OOD), and [CoLaDa](https://github.com/microsoft/vert-papers/tree/master/papers/CoLaDa).
* 2022-Aug: The [**Recognizers-Text** project](https://github.com/microsoft/Recognizers-Text) reached over **5 million** package downloads (across NuGet/npm/PyPI)! 
* 2022-May: **Tiara** (ReTraCk v2), KC's new knowledge base question answering (KBQA) system, has reached **#1** in all [Generalizable Question Answering (GrailQA)](https://dki-lab.github.io/GrailQA/) evaluation categories including Overall, Compositional Generalization, and Zero-Shot.  
* 2022-Apr: We have now open-sourced the latest version of the [**LinkingPark** system](https://github.com/microsoft/vert-papers/tree/master/papers/LinkingPark) for automatic semantic table interpretation. This new version includes improved performance, stability, flexibility, and overall results. Contributions and collaboration are very welcome!
* 2022-Mar: The [**Recognizers-Text** project](https://github.com/microsoft/Recognizers-Text) reached over **4 million** package downloads (across NuGet/npm/PyPI)! 
* 2021-Jul: The [**Recognizers-Text** project](https://github.com/microsoft/Recognizers-Text) reached over **3 million** package downloads (across NuGet/npm/PyPI)! 
* 2021-May: [**ReTraCk**](https://github.com/microsoft/KC/tree/master/papers/ReTraCk) has reached **\#1** in the [Generalizable Question Answering (GrailQA) leaderboard](https://dki-lab.github.io/GrailQA/) for knowledge base QA (KBQA).
* 2020-Dec: The [**Recognizers-Text** project](https://github.com/microsoft/Recognizers-Text) reached over **2 million** package downloads (across NuGet/npm/PyPI)! 
* 2020-Nov: The [**LinkingPark**](https://www.microsoft.com/en-us/research/publication/linkingpark-an-integrated-approach-for-semantic-table-interpretation/) system, developed in partnership between the Knowledge Computing group at MSRA and our collaborators in MSR Cambridge, has gotten 2nd place in the [SemTab 2020 challenge (Semantic Web Challenge on Tabular Data to Knowledge Graph Matching)](https://www.cs.ox.ac.uk/isg/challenges/sem-tab/2020/results.html)!

# Recent Papers:

* [Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text](https://aclanthology.org/2023.acl-long.403/), *Qianhui Wu, Huiqiang Jiang, Haonan Yin, Börje Karlsson, Chin-Yew Lin*, ACL 2023. <br>Repository: **https://github.com/microsoft/KC/tree/main/papers/MLKD_OOD**
* [ColaDa: A Collaborative Label Denoising Framework for Cross-lingual Named Entity Recognition](https://aclanthology.org/2023.acl-long.330/), *Tingting Ma, Qianhui Wu, Huiqiang Jiang, Börje Karlsson, Tiejun Zhao, Chin-Yew Lin*, ACL 2023. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/CoLaDa**
* [TIARA: Multi-grained Retrieval for Robust Question Answering over Large Knowledge Bases](https://arxiv.org/abs/2210.12925), *Yiheng Shu, Zhiwei Yu, Yuhan Li, Börje F. Karlsson, Tingting Ma, Yuzhong Qu, Chin-Yew Lin*, EMNLP 2022, 2022. <br>Repository: **https://github.com/microsoft/KC/tree/master/papers/TIARA**
* [LinkingPark: An Automatic Semantic Table Interpretation System](https://www.sciencedirect.com/science/article/abs/pii/S1570826822000233), *Shuang Chen, Alperen Karaoglu, Carina Negreanu, Tingting Ma, Jin-Ge Yao, Jack Williams, Feng Jiang, Andy Gordon, Chin-Yew Lin*, Journal of Web Semantics, 2022. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/LinkingPark**
* [Rows from Many Sources: Enriching row completions from Wikidata with a pre-trained Language Model](https://arxiv.org/abs/2204.07014), *Carina Negreanu, Alperen Karaoglu, Jack Williams, Shuang Chen, Daniel Fabian, Andrew Gordon, Chin-Yew Lin*, Wiki Workshop 2022.
* [On the Effectiveness of Sentence Encoding for Intent Detection Meta-Learning](https://arxiv.org/abs/X), *Tingting Ma, Qianhui Wu, Zhiwei Yu, Tiejun Zhao, Chin-Yew Lin*, NAACL 2022. <br>Repository: **https://github.com/microsoft/KC/tree/master/papers/IDML**
* [Decomposed Meta-Learning for Few-Shot Named Entity Recognition](https://arxiv.org/abs/2204.05751), *Tingting Ma, Huiqiang Jiang, Qianhui Wu, Tiejun Zhao, Chin-Yew Lin*, Findings of the ACL 2022. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/DecomposedMetaNER**
* [AdvPicker: Effectively Leveraging Unlabeled Data via Adversarial Discriminator for Cross-Lingual NER](https://arxiv.org/abs/2106.02300), *Weile Chen, Huiqiang Jiang, Qianhui Wu, Börje F. Karlsson, Yi Guan*, ACL 2021. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/AdvPicker**
* [ReTraCk: A Flexible and Efficient Framework for Knowledge Base Question Answering](https://aclanthology.org/2021.acl-demo.39/), *Shuang Chen, Qian Liu, Zhiwei Yu, Chin-Yew Lin, Jian-Guang Lou, Feng Jiang*, ACL 2021. (demo paper) <br>Repository: **https://github.com/microsoft/KC/tree/master/papers/ReTraCk**
* [BoningKnife: Joint Entity Mention Detection and Typing for Nested NER via prior Boundary Knowledge](https://arxiv.org/abs/2107.09429), *Huiqiang Jiang, Guoxin Wang, Weile Chen, Chengxi Zhang, Börje F. Karlsson*, arXiv:2107.09429 - 2020/2021.
* [LinkingPark: An integrated approach for Semantic Table Interpretation](http://ceur-ws.org/Vol-2775/paper7.pdf), *Shuang Chen, Alperen Karaoglu, Carina Negreanu, Tingting Ma, Jin-Ge Yao, Jack Williams, Andy Gordon, Chin-Yew Lin*, Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (SemTab 2020) at ISWC 2020. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/LinkingPark**
* [UniTrans: Unifying Model Transfer and Data Transfer for Cross-Lingual Named Entity Recognition with Unlabeled Data](https://www.ijcai.org/Proceedings/2020/543), *Qianhui Wu, Zijia Lin, Börje F. Karlsson, Biqing Huang, Jian-Guang Lou*, IJCAI 2020. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/UniTrans**
* [Single-/Multi-Source Cross-Lingual NER via Teacher-Student Learning on Unlabeled Data in Target Language](https://arxiv.org/abs/2004.12440), *Qianhui Wu, Zijia Lin, Börje F. Karlsson, Jian-Guang Lou, Biqing Huang*, ACL 2020. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/SingleMulti-TS**
* [Enhanced Meta-Learning for Cross-lingual Named Entity Recognition with Minimal Resources](https://arxiv.org/abs/1911.06161), *Qianhui Wu, Zijia Lin, Guoxin Wang, Hui Chen, Börje F. Karlsson, Biqing Huang, Chin-Yew Lin*, AAAI 2020. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/Meta-Cross**
* [Improving Entity Linking by Modeling Latent Entity Type Information](https://arxiv.org/abs/2001.01447), *Shuang Chen, Jinpeng Wang, Feng Jiang, Chin-Yew Lin*, AAAI 2020.
* [Exploring Word Representations on Time Expression Recognition](https://www.microsoft.com/en-us/research/publication/exploring-word-representations-on-time-expression-recognition), *Sanxing Chen, Guoxin Wang, Börje Karlsson*, Technical Report - Microsoft Research Asia, 2019.
* [Towards Improving Neural Named Entity Recognition with Gazetteers](https://www.aclweb.org/anthology/P19-1524/), *Tianyu Liu, Jin-Ge Yao, Chin-Yew Lin*, ACL 2019. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/SubTagger**
* [CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition](https://arxiv.org/abs/1904.02141), *Yuying Zhu, Guoxin Wang, Börje F. Karlsson*, NAACL-HLT 2019. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/CAN-NER**
* [GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition](https://arxiv.org/abs/1907.05611), *Hui Chen, Zijia Lin, Guiguang Ding, Jian-Guang Lou, Yusen Zhang, Börje F. Karlsson*, AAAI 2019. <br>Repository: **https://github.com/microsoft/vert-papers/tree/master/papers/GRN-NER**

# Related Projects:

* **[microsoft/Recognizers-Text](https://github.com/microsoft/Recognizers-Text)** - Open-source library that provides recognition and normalization/resolution of **numbers**, **units**, **date/time**, and **sequences** (e.g., phone numbers, URLs) expressed in multiple languages;
* [**Knowledge Computing (KC)** on GitHub](https://github.com/microsoft/KC) - Open-source repository including code and datasets for other projects by the [Knowledge Computing group at MSRA](https://www.microsoft.com/en-us/research/group/knowledge-computing/). 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
