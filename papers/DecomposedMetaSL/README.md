# Decomposed Meta-Learning for Few-Shot Sequence Labeling

This repository contains the open-sourced official implementation of the paper:
[Decomposed Meta-Learning for Few-Shot Sequence Labeling](https://ieeexplore.ieee.org/abstract/document/10458261) (TASLP).


_Tingting Ma, Qianhui Wu, Huiqiang Jiang, Jieru Lin, Börje F. Karlsson, Tiejun Zhao, and Chin-Yew Lin_

If you find this repo helpful, please cite the following paper

```bibtex
@ARTICLE{ma-etal-2024-decomposedmetasl,
  author={Ma, Tingting and Wu, Qianhui and Jiang, Huiqiang and Lin, Jieru and Karlsson, Börje F. and Zhao, Tiejun and Lin, Chin-Yew},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Decomposed Meta-Learning for Few-Shot Sequence Labeling}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TASLP.2024.3372879}
}
```

For any questions/comments, please feel free to open GitHub issues.

### Requirements

- python 3.9.17
- pytorch 1.9.1+cu111
- [HuggingFace Transformers 4.10.0](https://github.com/huggingface/transformers)

### Input data format

train/dev/test_N_K_id.jsonl:
Each line contains the following fields:

1. `target_classes`: A list of types (e.g., "event-other," "person-scholar").
2. `query_idx`: A list of indexes corresponding to query sentences for the i-th instance in train/dev/test.txt.
3. `support_idx`: A list of indexes corresponding to support sentences for the i-th instance in train/dev/test.txt.

### Train and Evaluate

For _Seperate_ model,

```bash
bash scripts/train_ment.sh
bash scripts/train_type.sh
bash scripts/eval_sep.sh
```

For _Joint_ model,

```bash
bash scripts/train_joint.sh
bash scripts/eval_joint.sh
```

For _POS tagging_ task, run

```bash
bash scripts/train_pos.sh
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