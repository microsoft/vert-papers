# -*- coding: utf-8 -*-
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from model import BertForTokenClassificationKD
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import (
    evaluate_viterbi,
    get_labels,
    get_optimizer_grouped_parameters,
    mkdir,
    set_seed,
    torch_device,
    write_prediction,
)


parser = argparse.ArgumentParser()
parser.add_argument("--tgt_lang", default="es", type=str)
parser.add_argument("--eval_langs", default="de", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--lr_lm", default=6e-5, type=float)
parser.add_argument("--num_epoches", default=10, type=int)
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--do_predict", default=False, type=bool)

parser.add_argument(
    "--weight_decay", default=0.01, type=float, help="AdamW weight decay."
)
parser.add_argument(
    "--warmup_ratio", default=0.05, type=float, help="Linear warmup over warmup_ratio."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer."
)
parser.add_argument(
    "--freeze_bottom_layer",
    default=3,
    type=int,
    help="the freeze bottom layer number during fine-tuning.",
)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
parser.add_argument("--seed", type=int, default=320)
parser.add_argument("--result_path", default="result/", type=str)

parser.add_argument("--model_type", default="bert", type=str)
parser.add_argument(
    "--model_name_or_path", default="bert-base-multilingual-cased", type=str
)
parser.add_argument("--model_path", default="bert-base-multilingual-cased", type=str)
parser.add_argument("--data_dir", default="./data/", type=str)
parser.add_argument("--labels", default="./data/labels.txt", type=str)
parser.add_argument("--use_overlap", default=True, type=bool)

parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--overwrite_cache",
    action="store_true",
    help="Overwrite the cached training and evaluation sets",
)

args = parser.parse_args()
mkdir(args.result_path)
args.device = torch_device(args.gpu_id)
args.n_gpu = 1
args.output_path = "{}{}-{}-kd-en/".format(args.result_path, args.seed, args.tgt_lang)
mkdir(args.output_path)
set_seed(args.seed)
writer = SummaryWriter(args.output_path + "run")

logging.basicConfig(
    level=logging.INFO,
    filename=args.output_path + "log.txt",
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassificationKD, BertTokenizer),
    "unicoder": (
        XLMRobertaConfig,
        XLMRobertaForTokenClassification,
        XLMRobertaTokenizer,
    ),
}


config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

# load English data
tokenizer = tokenizer_class.from_pretrained(
    args.model_path
    + ("/sentencepiece.bpe.model" if args.model_type == "unicoder" else ""),
    do_lower_case=args.do_lower_case,
)
labels = get_labels(args.labels)


if args.use_overlap:
    train_dataset = torch.load(
        f"{args.result_path}/adv-data/{args.tgt_lang}-ne-all.pth"
    )
    train_dataset = TensorDataset(*[torch.stack(ii) for ii in train_dataset])
else:
    train_dataset = torch.load(
        f"{args.result_path}/{args.seed}-{args.tgt_lang}-train/wrg_data_ne.pth"
    )
    train_dataset_cor = torch.load(
        f"{args.result_path}/{args.seed}-{args.tgt_lang}-train/cor_data_ne.pth"
    )
    train_dataset = ConcatDataset([train_dataset, train_dataset_cor])

num = len(train_dataset[0])

xl_data_ne = [[] for _ in range(9)]
for i in range(train_dataset.__len__()):
    temp = train_dataset.__getitem__(i)
    for j in range(9):
        xl_data_ne[j].append(temp[j])
train_dataset = xl_data_ne

confidence = torch.stack(train_dataset[7])
confidence = (1 - torch.abs(confidence[:, 1] - 0.5)) * -1
sorted_idx, indices = torch.sort(confidence, 0)

train_dataset_ = [[] for _ in range(9)]
for i in indices[: int(len(indices) * (8 / 10))]:
    for j in range(9):
        train_dataset_[j].append(train_dataset[j][i])
train_dataset = train_dataset_

train_dataset = TensorDataset(*[torch.stack(ii) for ii in train_dataset])
ner_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


# model moudles
config = config_class.from_pretrained(args.model_path, num_labels=len(labels))
config.output_hidden_states = True
config.output_attentions = False
ner_lm = model_class.from_pretrained(args.model_path, from_tf=False, config=config).to(
    args.device
)

# optimizers
t_total = args.num_epoches * ner_train_loader.dataset.__len__() / args.batch_size
no_grad = ["embeddings"] + [
    "layer." + str(layer_i) + "."
    for layer_i in range(12)
    if layer_i < args.freeze_bottom_layer
]

opt_params = get_optimizer_grouped_parameters(args, ner_lm, no_grad=no_grad)
optimizer1 = AdamW(
    opt_params, lr=args.lr_lm, eps=args.adam_epsilon, weight_decay=args.weight_decay
)
scheduler1 = get_linear_schedule_with_warmup(
    optimizer1,
    num_warmup_steps=int(t_total * args.warmup_ratio),
    num_training_steps=t_total,
)
logger.info("Start training.")

global_step = 0
best_f1, best_value = {ll: 0 for ll in args.eval_langs.split()}, {
    ll: {} for ll in args.eval_langs.split()
}

if args.do_predict:
    ner_lm = torch.load(f"{args.output_path}lm_model.pth").to(args.device)
    args.num_epoches = 1

for epoch in range(args.num_epoches):
    if not args.do_predict:
        for i, ner_data in enumerate(ner_train_loader):
            ner_data = [item.to(args.device) for item in ner_data]

            # train lm on ner data
            inputs = {
                "input_ids": ner_data[0],
                "attention_mask": ner_data[1],
                "token_type_ids": ner_data[2]
                if args.model_type in ["bert", "xlnet"]
                else None,
                "labels": ner_data[3],
                "src_probs": ner_data[-1],
            }
            loss_ner, _, _ = ner_lm(**inputs)

            loss_ner.backward()
            optimizer1.step()
            scheduler1.step()
            ner_lm.zero_grad()
            logger.info(f"epoch: {epoch}, batch: {i}, loss: {loss_ner.item()}")

    for ll in args.eval_langs.split():
        args.data_dir = "./data/" + ll
        r_tgt_dev, pred_dev = evaluate_viterbi(
            args,
            ner_lm,
            tokenizer,
            labels,
            CrossEntropyLoss().ignore_index,
            "dev",
            prefix="",
            add_eval=True,
        )
        r_tgt_test, pred_test = evaluate_viterbi(
            args,
            ner_lm,
            tokenizer,
            labels,
            CrossEntropyLoss().ignore_index,
            "test",
            prefix="",
            add_eval=True,
        )
        if r_tgt_dev["f1"] > best_f1[ll]:
            best_f1[ll] = r_tgt_dev["f1"]
            best_value[ll] = r_tgt_test
            torch.save(ner_lm, f"{args.output_path}lm_model.pth")

        write_prediction(
            pred_dev, f"{args.output_path}adv-{ll}-dev.txt", f"{args.data_dir}/dev.txt"
        )
        write_prediction(
            pred_test,
            f"{args.output_path}adv-{ll}-test.txt",
            f"{args.data_dir}/test.txt",
        )

for ll in args.eval_langs.split():
    logger.info(
        f"{ll} best f1/p/r: {best_value[ll]['f1']}/{best_value[ll]['precision']}/{best_value[ll]['recall']}"
    )
