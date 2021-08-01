# apply gradient update on discriminator and generator
# using different batch of data

import argparse
import logging
from itertools import cycle

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import LanguageDiscriminatorTokenLevel
from transformers import (
    AdamW,
    BertConfig,
    BertForTokenClassification,
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
    load_and_cache_examples,
    mkdir,
    non_en_dataset,
    set_seed,
    torch_device,
    write_prediction,
)


parser = argparse.ArgumentParser()
parser.add_argument("--tgt_langs", default="es de nl", type=str)
parser.add_argument("--eval_langs", default="de es nl", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--lr_lm", default=6e-5, type=float)
parser.add_argument("--lr_d", default=5e-3, type=float)
parser.add_argument("--lr_gen", default=6e-7, type=float)
parser.add_argument("--num_epoches", default=10, type=int)
parser.add_argument("--disc_hidden_size", default=500, type=int)
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
args.result_path = "{}{}-{}-{}-{}/".format(
    args.result_path,
    args.disc_hidden_size,
    args.lr_d,
    args.seed,
    args.tgt_langs.replace(" ", "_"),
)
set_seed(args.seed)
mkdir(args.result_path)
writer = SummaryWriter(args.result_path + "run")

logging.basicConfig(
    level=logging.INFO,
    filename=args.result_path + "log.txt",
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "unicoder": (
        XLMRobertaConfig,
        XLMRobertaForTokenClassification,
        XLMRobertaTokenizer,
    ),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

# data loader
tokenizer = tokenizer_class.from_pretrained(
    args.model_path
    + ("/sentencepiece.bpe.model" if args.model_type == "unicoder" else ""),
    do_lower_case=args.do_lower_case,
)
labels = get_labels(args.labels)

# load English data
en_train_dataset = load_and_cache_examples(
    args,
    tokenizer,
    labels,
    CrossEntropyLoss().ignore_index,
    mode="train",
    data_dir=args.data_dir + "en",
)
ner_train_loader = DataLoader(
    en_train_dataset, batch_size=args.batch_size, shuffle=True
)
en_loader_d = DataLoader(en_train_dataset, batch_size=args.batch_size, shuffle=True)
en_loader_gen = DataLoader(en_train_dataset, batch_size=args.batch_size, shuffle=True)

# load tgt data
nonen_dataset = non_en_dataset(
    args,
    ner_train_loader.dataset.__len__(),
    tokenizer,
    labels,
    CrossEntropyLoss().ignore_index,
    args.tgt_langs,
)
non_en_loader_d = DataLoader(nonen_dataset, batch_size=args.batch_size, shuffle=True)
non_en_loader_gen = DataLoader(nonen_dataset, batch_size=args.batch_size, shuffle=True)

logger.info(
    f"Source language dataset size: {en_train_dataset.__len__()}, Target language dataset size: {nonen_dataset.__len__()}"
)

# model moudles
config = config_class.from_pretrained(args.model_path, num_labels=len(labels))
config.output_hidden_states = True
config.output_attentions = False
ner_lm = model_class.from_pretrained(args.model_path, from_tf=False, config=config).to(
    args.device
)
discri1 = LanguageDiscriminatorTokenLevel(config.hidden_size, args.disc_hidden_size).to(
    args.device
)

# optimizers
t_total = args.num_epoches * ner_train_loader.dataset.__len__() / args.batch_size
no_grad = ["embeddings"] + [
    "layer.{}.".format(layer_i)
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
optimizer2 = AdamW(discri1.parameters(), lr=args.lr_d, eps=args.adam_epsilon)
scheduler2 = get_linear_schedule_with_warmup(
    optimizer2,
    num_warmup_steps=int(t_total * args.warmup_ratio),
    num_training_steps=t_total,
)
optimizer3 = AdamW(
    ner_lm.bert.parameters(),
    lr=args.lr_gen,
    eps=args.adam_epsilon,
    weight_decay=args.weight_decay,
)
scheduler3 = get_linear_schedule_with_warmup(
    optimizer3,
    num_warmup_steps=int(t_total * args.warmup_ratio),
    num_training_steps=t_total,
)

logger.info("Start training.")

best_f1, best_value = {ll: 0 for ll in args.eval_langs.split()}, {
    ll: {} for ll in args.eval_langs.split()
}
global_step = 0
if args.do_predict:
    ner_lm = torch.load(f"{args.result_path}lm_model.pth").to(args.device)
    discri1 = torch.load(f"{args.result_path}discriminator.pth").to(args.device)
    args.num_epoches = 1

for epoch in range(args.num_epoches):
    if not args.do_predict:
        for i, data in enumerate(
            zip(
                cycle(non_en_loader_d),
                cycle(non_en_loader_gen),
                ner_train_loader,
                en_loader_d,
                en_loader_gen,
            )
        ):
            ne_d, ne_gen, ner_data, en_d, en_gen = [
                [jj.to(args.device) for jj in ii] for ii in data
            ]

            # train lm on ner data
            ner_lm.train()
            discri1.train()
            inputs = {
                "input_ids": ner_data[0],
                "attention_mask": ner_data[1],
                "token_type_ids": ner_data[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
                "labels": ner_data[3],
            }

            loss_ner, _, _ = ner_lm(**inputs)
            loss_ner.backward(retain_graph=True)

            optimizer1.step()
            scheduler1.step()
            ner_lm.zero_grad()
            discri1.zero_grad()
            del inputs, ner_data, loss_ner

            # train discriminator
            inputs = {
                "input_ids": ne_d[0],
                "attention_mask": ne_d[1],
                "token_type_ids": ne_d[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
            }

            _, nonen_embedding = ner_lm(**inputs)
            loss_d, _, _ = discri1(nonen_embedding[-1], ne_d[1])

            inputs = {
                "input_ids": en_d[0],
                "attention_mask": en_d[1],
                "token_type_ids": en_d[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
            }

            _, en_embedding = ner_lm(**inputs)
            loss_d_, _, _ = discri1(en_embedding[-1], en_d[1], fake=False)

            loss_d += loss_d_

            loss_d.backward(retain_graph=True)
            optimizer2.step()
            scheduler2.step()
            ner_lm.zero_grad()
            discri1.zero_grad()

            del loss_d, inputs, en_embedding, nonen_embedding, en_d, ne_d, loss_d_

            # train generator
            inputs = {
                "input_ids": ne_gen[0],
                "attention_mask": ne_gen[1],
                "token_type_ids": ne_gen[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
            }

            _, nonen_embedding = ner_lm(**inputs)
            loss_gen_, _, _ = discri1(nonen_embedding[-1], ne_gen[1], fake=False)

            inputs = {
                "input_ids": en_gen[0],
                "attention_mask": en_gen[1],
                "token_type_ids": en_gen[2]
                if args.model_type in ["bert", "unicoder"]
                else None,
            }

            _, en_embedding = ner_lm(**inputs)
            loss_gen, _, _ = discri1(en_embedding[-1], en_gen[1])

            loss_gen += loss_gen_

            loss_gen.backward()
            optimizer3.step()
            scheduler3.step()
            ner_lm.zero_grad()
            discri1.zero_grad()
            del (
                inputs,
                en_gen,
                ne_gen,
                en_embedding,
                nonen_embedding,
                loss_gen,
                loss_gen_,
            )

    keys, abb_keys = ["precision", "recall", "f1"], ["p", "r", "f1"]
    for ll in args.eval_langs.split():
        args.data_dir = "./data/{}/".format(ll)
        r_tgt_dev, pred_dev = evaluate_viterbi(
            args, ner_lm, tokenizer, labels, CrossEntropyLoss().ignore_index, "dev"
        )
        r_tgt_test, pred_test = evaluate_viterbi(
            args, ner_lm, tokenizer, labels, CrossEntropyLoss().ignore_index, "test"
        )

        if r_tgt_dev["f1"] > best_f1[ll]:
            best_f1[ll] = r_tgt_dev["f1"]
            best_value[ll] = r_tgt_test

            write_prediction(
                pred_dev,
                "{}adv-{}-dev.txt".format(args.result_path, ll),
                "{}/dev.txt".format(args.data_dir),
            )
            write_prediction(
                pred_test,
                "{}adv-{}-test.txt".format(args.result_path, ll),
                "{}/dev.txt".format(args.data_dir),
            )
            torch.save(ner_lm, "{}lm-{}_model.pth".format(args.result_path, ll))
            torch.save(discri1, "{}{}-discriminator.pth".format(args.result_path, ll))
            torch.save(ner_lm, f"{args.result_path}lm_model.pth")
            torch.save(discri1, f"{args.result_path}discriminator.pth")

        writer.add_scalars(
            f"{ll}-performance",
            {
                **{f"{jj}_dev": r_tgt_dev[ii] for ii, jj in zip(keys, abb_keys)},
                **{f"{jj}_test": r_tgt_test[ii] for ii, jj in zip(keys, abb_keys)},
            },
            epoch,
        )
        writer.add_scalars(
            f"{ll}-loss",
            {"loss_test": r_tgt_test["loss"], "loss_dev": r_tgt_dev["loss"]},
            epoch,
        )

for ll in args.eval_langs.split():
    logger.info(
        f"{ll} best f1/p/r: {best_value[ll]['f1']}/{best_value[ll]['precision']}/{best_value[ll]['recall']}"
    )
