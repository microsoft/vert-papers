# GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition

This repository contains the open-sourced official implementation of the paper:

[GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition](https://arxiv.org/abs/1907.05611) (AAAI 2019).  
_Hui Chen, Zijia Lin, Guiguang Ding, Jian-Guang Lou, Yusen Zhang, and Börje F. Karlsson_

If you find this repo helpful, please cite either of the following versions of the paper:
```tex
@inproceedings{DBLP:conf/aaai/ChenLDLZK19,
  author    = {Hui Chen and Zijia Lin and Guiguang Ding and Jian-Guang Lou and Yusen Zhang and B{\"{o}}rje F. Karlsson},
  title     = {{GRN:} Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition},
  booktitle = {The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI} 2019},
  pages     = {6236--6243},
  publisher = {{AAAI} Press},
  year      = {2019},
  url       = {https://doi.org/10.1609/aaai.v33i01.33016236},
  doi       = {10.1609/aaai.v33i01.33016236},
}
```

## Requirements and Installation
We recommended the following dependencies.

* [PyTorch](http://pytorch.org/) 0.4
* Python 3.6
* torchvision
* Numpy

## Training and Test

After datasets are prepared, run `train.py`:

```bash
python train.py
```

Test:

```bash
python test.py
```

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
