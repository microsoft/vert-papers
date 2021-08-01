import argparse
import logging
from itertools import cycle

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, BertTokenizer, XLMRobertaTokenizer, get_linear_schedule_with_warmup
from utils import get_labels, load_and_cache_examples, mkdir, set_seed, torch_device


parser = argparse.ArgumentParser()
parser.add_argument("--tgt_lang", default="de", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--lr_d", default=5e-3, type=float)
parser.add_argument("--gpu_id", default=0, type=int)

parser.add_argument(
    "--weight_decay", default=0.01, type=float, help="AdamW weight decay."
)
parser.add_argument(
    "--warmup_ratio", default=0.05, type=float, help="Linear warmup over warmup_ratio."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer."
)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--seed", type=int, default=320)
parser.add_argument("--result_path", default="result/", type=str)

parser.add_argument("--model_type", default="bert", type=str)
parser.add_argument(
    "--model_name_or_path", default="bert-base-multilingual-cased", type=str
)
parser.add_argument("--model_path", default="bert-base-multilingual-cased", type=str)
parser.add_argument("--pth_dir", default="result/500-0.005-320-es_de_nl/", type=str)
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

parser.add_argument("--threshold", default=0.1, type=float)
parser.add_argument("--train_type", default="train", type=str)
args = parser.parse_args()
mkdir(args.result_path)
args.device = torch_device(args.gpu_id)
args.n_gpu = 1
args.result_path = "{}{}-{}-{}/".format(
    args.result_path, args.seed, args.tgt_lang, args.train_type
)
set_seed(args.seed)
mkdir(args.result_path)

ner_lm2 = torch.load("{}/lm-{}_model.pth".format(args.pth_dir, args.tgt_lang)).to(
    args.device
)
discri2 = torch.load("{}/discriminator.pth".format(args.pth_dir)).to(args.device)
ner_lm = torch.load("{}/lm_model.pth".format(args.pth_dir)).to(args.device)

mkdir(args.result_path)
writer = SummaryWriter(args.result_path + "run")

logging.basicConfig(
    level=logging.INFO,
    filename=args.result_path + "log.txt",
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

args.labels = get_labels(args.labels)


# model definition
MODEL_CLASSES = {"bert": (BertTokenizer), "unicoder": (XLMRobertaTokenizer)}
tokenizer_class = MODEL_CLASSES[args.model_type]
args.tokenizer = tokenizer_class.from_pretrained(
    args.model_path
    + ("/sentencepiece.bpe.model" if args.model_type == "unicoder" else ""),
    do_lower_case=args.do_lower_case,
)

en_train_dataset = load_and_cache_examples(
    args,
    args.tokenizer,
    args.labels,
    CrossEntropyLoss().ignore_index,
    mode=args.train_type,
    data_dir=args.data_dir + "en",
)
en_loader_lang = DataLoader(en_train_dataset, batch_size=args.batch_size, shuffle=True)


non_en_dataset = load_and_cache_examples(
    args,
    args.tokenizer,
    args.labels,
    CrossEntropyLoss().ignore_index,
    mode=args.train_type,
    data_dir=args.data_dir + args.tgt_lang,
)
non_en_loader_lang = DataLoader(
    non_en_dataset, batch_size=args.batch_size, shuffle=True
)

t_total = 3 * en_loader_lang.dataset.__len__() / args.batch_size
optimizer4 = AdamW(
    discri2.parameters(),
    lr=args.lr_d,
    eps=args.adam_epsilon,
    weight_decay=args.weight_decay,
)
scheduler4 = get_linear_schedule_with_warmup(
    optimizer4,
    num_warmup_steps=int(t_total * args.warmup_ratio),
    num_training_steps=t_total,
)
new_inputs = []
for i, data in enumerate(zip(non_en_loader_lang, cycle(en_loader_lang))):
    ne_lang, en_lang = [[jj.to(args.device) for jj in ii] for ii in data]

    # Test Lang_Dis True
    inputs = {
        "input_ids": ne_lang[0],
        "attention_mask": ne_lang[1],
        "token_type_ids": ne_lang[2]
        if args.model_type in ["bert", "unicoder"]
        else None,
    }
    logtis, nonen_embedding = ner_lm(**inputs)
    loss_lang_, confidences, cor_lang = discri2(nonen_embedding[-1], ne_lang[1])
    confidences = [
        confidences[seq_num][: int(torch.sum(ne_lang[1][seq_num]))]
        for seq_num in range(len(ne_lang[0]))
    ]
    confidences = torch.stack([torch.mean(seq, 0) for seq in confidences], dim=0)

    lang_type = torch.tensor([0] * len(ne_lang[0]))
    random_seed = torch.tensor([args.seed] * len(ne_lang[0]))
    adv = torch.tensor([1] * len(ne_lang[0]))

    logtis, _ = ner_lm2(**inputs)

    confidences = confidences.detach().cpu()
    logtis = logtis.detach().cpu()

    tmp = [
        *[ii.cpu() for ii in ne_lang],
        lang_type,
        random_seed,
        adv,
        confidences,
        logtis,
    ]
    new_inputs = (
        [torch.cat((ii, jj)) for ii, jj in zip(new_inputs, tmp)] if new_inputs else tmp
    )

    # Test Lang_Dis False
    B = len(en_lang[0])
    inputs = {
        "input_ids": en_lang[0],
        "attention_mask": en_lang[1],
        "token_type_ids": en_lang[2]
        if args.model_type in ["bert", "unicoder"]
        else None,
    }
    logtis, en_embedding = ner_lm(**inputs)
    loss_lang, confidences, cor_lang_ = discri2(
        en_embedding[-1], en_lang[1], fake=False
    )
    confidences = [
        confidences[seq_num][: int(torch.sum(en_lang[1][seq_num]))]
        for seq_num in range(B)
    ]
    confidences = torch.stack([torch.mean(seq, 0) for seq in confidences], dim=0)

    lang_type = torch.tensor([1] * B)
    random_seed = torch.tensor([args.seed] * B)
    adv = torch.tensor([1] * B)
    confidences = confidences.detach().cpu()
    logtis = logtis.detach().cpu()
    tmp = [
        *[ii.cpu() for ii in en_lang],
        lang_type,
        random_seed,
        adv,
        confidences,
        logtis,
    ]
    new_inputs = [torch.cat((ii, jj)) for ii, jj in zip(new_inputs, tmp)]

    ner_lm.zero_grad()
    discri2.zero_grad()

    cor_lang += cor_lang_
    logger.info(
        "lang_epoch: {0} \t loss_lang: {1} \t cor_lang: {2} ".format(
            0, loss_lang.item(), cor_lang / 2
        )
    )


threshold = args.threshold
sp_data_en = [[] for _ in range(9)]
sp_data_ne = [[] for _ in range(9)]
xl_data_en = [[] for _ in range(9)]
xl_data_ne = [[] for _ in range(9)]
cor_data_en = [[] for _ in range(9)]
cor_data_ne = [[] for _ in range(9)]

for data_index in range(new_inputs[0].size(0)):
    if (new_inputs[7][data_index][0].item() < 0.5 + threshold) and (
        new_inputs[7][data_index][0].item() > 0.5 - threshold
    ):
        d = xl_data_en if new_inputs[4][data_index] == 1 else xl_data_ne
        for i in range(9):
            d[i].append(new_inputs[i][data_index])
        continue

    if ((new_inputs[7][data_index][0] > 0.5) and (new_inputs[4][data_index] == 1)) or (
        (new_inputs[7][data_index][1] > 0.5) and (new_inputs[4][data_index] == 0)
    ):
        d = sp_data_en if new_inputs[4][data_index] == 1 else sp_data_ne
        for i in range(9):
            d[i].append(new_inputs[i][data_index])
        continue

    d = cor_data_en if new_inputs[4][data_index] == 1 else cor_data_ne
    for i in range(9):
        d[i].append(new_inputs[i][data_index])


sp_data_en = TensorDataset(*[torch.stack(ii) for ii in sp_data_en])
sp_data_ne = TensorDataset(*[torch.stack(ii) for ii in sp_data_ne])
xl_data_en = TensorDataset(*[torch.stack(ii) for ii in xl_data_en])
xl_data_ne = TensorDataset(*[torch.stack(ii) for ii in xl_data_ne])
cor_data_en = TensorDataset(*[torch.stack(ii) for ii in cor_data_en])
cor_data_ne = TensorDataset(*[torch.stack(ii) for ii in cor_data_ne])

wrg_data_en = ConcatDataset([sp_data_en, xl_data_en])
wrg_data_ne = ConcatDataset([sp_data_ne, xl_data_ne])

torch.save(xl_data_en, args.result_path + "xl_data_en.pth")
torch.save(xl_data_ne, args.result_path + "xl_data_ne.pth")
torch.save(cor_data_en, args.result_path + "cor_data_en.pth")
torch.save(cor_data_ne, args.result_path + "cor_data_ne.pth")
torch.save(wrg_data_en, args.result_path + "wrg_data_en.pth")
torch.save(wrg_data_ne, args.result_path + "wrg_data_ne.pth")
