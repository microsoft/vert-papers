import argparse
import logging
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler, TensorDataset

from seqeval.metrics import f1_score, precision_score, recall_score
from utils import ViterbiDecoder, get_labels, mkdir, torch_device


# evaluate
def evaluate_viterbi(labels, pad_token_label_id, eval_dataset, device):
    eval_batch_size = 32
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation with Viterbi Decoding *****")
    logger.info("  Num examples = %d" % len(eval_dataset))
    logger.info("  Batch size = %d" % eval_batch_size)

    viterbi_decoder = ViterbiDecoder(labels, pad_token_label_id, device)
    out_label_ids = None
    pred_label_list = []

    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        logits = batch[-1]

        # decode with viterbi
        log_probs = torch.nn.functional.log_softmax(
            logits.detach(), dim=-1
        )  # batch_size x max_seq_len x n_labels

        pred_labels = viterbi_decoder.forward(log_probs, batch[1], batch[3])
        pred_label_list.extend(pred_labels)

        if out_label_ids is None:
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            out_label_ids = np.append(
                out_label_ids, batch[3].detach().cpu().numpy(), axis=0
            )

    label_map = {i: label for i, label in enumerate(labels)}
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    if out_label_ids.shape[0] != len(pred_label_list):
        raise ValueError("Num of examples doesn't match!")

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
        if len(out_label_list[i]) != len(pred_label_list[i]):
            raise ValueError("Sequence length doesn't match!")

    results = {
        "precision": precision_score(out_label_list, pred_label_list) * 100,
        "recall": recall_score(out_label_list, pred_label_list) * 100,
        "f1": f1_score(out_label_list, pred_label_list) * 100,
    }

    for key in sorted(results.keys()):
        logger.info(f"{key} =  {results[key]}")

    return results, pred_label_list


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--result_path", default="result/", type=str)
parser.add_argument("--labels", default="./data/labels.txt", type=str)
parser.add_argument("--seeds", default="320,550,631,691,985", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--tgt_langs", default="es,de,nl", type=str)
args = parser.parse_args()
device = torch_device(args.gpu_id)
mkdir(args.result_path)
result_dir = os.path.join(args.result_path, "adv-data")
mkdir(result_dir)

labels = get_labels(args.labels)

logging.basicConfig(
    level=logging.INFO,
    filename=result_dir + "/log.txt",
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for lang in args.tgt_langs.split(","):
    logger.info(f"---------------{lang} Begin Overlap All------------")
    wrg = []
    for seed in args.seeds.split(","):
        dataset = torch.load(
            f"{args.result_path}/{seed}-{lang}-{args.mode}/wrg_data_ne.pth"
        )
        dataset_ = torch.load(
            f"{args.result_path}/{seed}-{lang}-{args.mode}/cor_data_ne.pth"
        )
        dataset = ConcatDataset([dataset, dataset_])

        for i in range(dataset.__len__()):
            wrg.append(tuple(dataset.__getitem__(i)[0].numpy().tolist()))

    wrg = list(set(wrg))
    logger.info(f"Len of WRG: {len(wrg)}")

    wrg_dict = {ii: [] for ii in wrg}

    for seed in args.seeds.split(","):
        dataset = torch.load(
            f"{args.result_path}/{seed}-{lang}-{args.mode}/wrg_data_ne.pth"
        )
        dataset_ = torch.load(
            f"{args.result_path}/{seed}-{lang}-{args.mode}/cor_data_ne.pth"
        )
        dataset = ConcatDataset([dataset, dataset_])

        for i in range(dataset.__len__()):
            temp = dataset.__getitem__(i)
            temp0 = tuple(temp[0].numpy().tolist())

            wrg_dict[temp0].append(temp)

    for i in wrg_dict:
        conf = []
        for sample in wrg_dict[i]:
            # 128 X 1
            max_log = torch.max(sample[-1], dim=-1)
            conf.append(torch.mean(max_log.values[: torch.sum(sample[1])]))
        max_con = conf.index(max(conf))
        wrg_dict[i] = wrg_dict[i][max_con]

    xl_data_ne = [[] for _ in range(9)]
    dataset = torch.load(
        f"{args.result_path}/{args.seeds.split(',')[0]}-{lang}-{args.mode}/wrg_data_ne.pth"
    )
    dataset_ = torch.load(
        f"{args.result_path}/{args.seeds.split(',')[0]}-{lang}-{args.mode}/cor_data_ne.pth"
    )
    dataset = ConcatDataset([dataset, dataset_])

    for i in range(dataset.__len__()):
        temp = tuple(dataset.__getitem__(i)[0].numpy().tolist())
        for j in range(9):
            xl_data_ne[j].append(wrg_dict[temp][j])

    eval_dataset1 = TensorDataset(*[torch.stack(ii) for ii in xl_data_ne])
    evaluate_viterbi(labels, CrossEntropyLoss().ignore_index, eval_dataset1, device)

    torch.save(xl_data_ne, f"{result_dir}/{lang}-ne-all.pth")
