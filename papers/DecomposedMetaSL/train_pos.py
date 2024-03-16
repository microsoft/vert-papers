import torch
import numpy as np
import json
import argparse
import os
import random
from util.pos_loader import get_seq_loader
from trainer.pos_trainer import POSTrainer
from util.span_encoder import BERTSpanEncoder
from model.pos_model import SeqProtoCls
from util.log_utils import write_pos_pred_json, save_json, set_seed

def add_args():
    def str2bool(arg):
        if arg.lower() == "true":
            return True
        return False

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str,
                        help='train / test / typetest')
    parser.add_argument('--train_eval_mode', default='test', type=str, choices=['test', 'typetest'])
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--output_dir', default=None,
                        help='output dir')
    parser.add_argument('--log_dir', default=None,
                        help='log dir')
    parser.add_argument('--root', default=None, type=str,
                        help='data root dir')
    parser.add_argument('--train', default='train.txt',
                        help='train file')
    parser.add_argument('--val', default='dev.txt',
                        help='val file')
    parser.add_argument('--test', default='test.txt',
                        help='test file')
    parser.add_argument('--ep_dir', default=None, type=str)
    parser.add_argument('--ep_train', default=None, type=str)
    parser.add_argument('--ep_val', default=None, type=str)
    parser.add_argument('--ep_test', default=None, type=str)

    parser.add_argument('--N', default=None, type=int,
                        help='N way')
    parser.add_argument('--K', default=None, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=None, type=int,
                        help='Num of query per class')

    parser.add_argument('--encoder_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--word_encode_choice', default='first', type=str)

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--last_n_layer', default=-4, type=int)
    parser.add_argument('--max_loss', default=0, type=float)

    parser.add_argument('--dot', type=str2bool, default=False,
                        help='use dot instead of L2 distance for knn')
    parser.add_argument("--normalize", default='none', type=str, choices=['none', 'l2'])
    parser.add_argument("--temperature", default=None, type=float)

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout rate')

    parser.add_argument('--log_steps', default=2000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--val_steps', default=2000, type=int,
                        help='val after training how many iters')

    parser.add_argument('--train_batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--eval_batch_size', default=1, type=int,
                        help='batch size')

    parser.add_argument('--train_iter', default=-1, type=int,
                        help='num of iters in training')
    parser.add_argument('--dev_iter', default=-1, type=int,
                        help='num of iters in training')
    parser.add_argument('--test_iter', default=-1, type=int,
                        help='num of iters in training')

    parser.add_argument("--max_grad_norm", default=None, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="lr rate")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")
    parser.add_argument("--bert_learning_rate", default=5e-5, type=float, help="lr rate")
    parser.add_argument("--bert_weight_decay", default=1e-5, type=float, help="weight decay")

    parser.add_argument("--warmup_step", default=0, type=int)

    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use nvidia apex fp16')

    parser.add_argument('--use_maml', default=False, type=str2bool)
    parser.add_argument('--warmup_prop_inner', default=0, type=float)
    parser.add_argument('--train_inner_lr', default=0, type=float)
    parser.add_argument('--train_inner_steps', default=0, type=int)
    parser.add_argument('--eval_inner_lr', default=0, type=float)
    parser.add_argument('--eval_inner_steps', default=0, type=int)
    opt = parser.parse_args()
    return opt

def main(opt):
    set_seed(opt.seed)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if opt.mode == "train":
        output_dir = opt.output_dir
        print("Output dir is ", output_dir)
        log_dir = os.path.join(
            output_dir,
            "logs",
        )
        opt.log_dir = log_dir
        save_json(opt.__dict__, os.path.join(opt.output_dir, "train_setting.txt"))
    else:
        save_json(opt.__dict__, os.path.join(opt.output_dir, "test_setting.txt"))

    print('loading model and tokenizer...')
    word_encoder = BERTSpanEncoder(opt.encoder_name_or_path, opt.max_length, last_n_layer=opt.last_n_layer,
                                   word_encode_choice=opt.word_encode_choice, drop_p=opt.dropout)
    print('loading data')
    if opt.mode == "train":
        train_loader = get_seq_loader(os.path.join(opt.root, opt.train), "train", word_encoder, batch_size=opt.train_batch_size, max_length=opt.max_length,
                                  shuffle=True,
                                  debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None)
    else:
        train_loader = get_seq_loader(os.path.join(opt.root, opt.train), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  shuffle=False,
                                  debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None)
    dev_loader = get_seq_loader(os.path.join(opt.root, opt.val), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  shuffle=False,
                                  debug_file=os.path.join(opt.ep_dir, opt.ep_val) if opt.ep_val else None)
    test_loader = get_seq_loader(os.path.join(opt.root, opt.test), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  shuffle=False,
                                  debug_file=os.path.join(opt.ep_dir, opt.ep_test) if opt.ep_test else None)
    
    print("max_length: {}".format(opt.max_length))
    print('mode: {}'.format(opt.mode))
    
    print("{}-way-{}-shot Token MAML-Proto Few-Shot NER".format(opt.N, opt.K))
    print("this mode can only used for maml training !!!!!!!!!!!!!!!!!")
    model = SeqProtoCls(word_encoder, opt.max_loss, dot=opt.dot, normalize=opt.normalize,
                    temperature=opt.temperature)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    trainer = POSTrainer()

    if opt.mode == 'train':
        print("==================start training==================")
        trainer.train(model, opt, device, train_loader, dev_loader, dev_pred_fn=os.path.join(opt.output_dir, "dev_pred.json"))
        test_loss, test_acc, test_logs = trainer.eval(model, device, test_loader, eval_iter=opt.test_iter, load_ckpt=os.path.join(opt.output_dir, "model.pth.tar"),
                                    update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_mode=opt.train_eval_mode)

        print('[TEST] loss: {0:2.6f} | [POS] acc: {1:3.4f}'\
                         .format(test_loss, test_acc) + '\r')        
    else:
        test_loss, test_acc, test_logs = trainer.eval(model, device, test_loader, load_ckpt=opt.load_ckpt,
                                                             eval_iter=opt.test_iter, update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_mode=opt.mode)

        print('[TEST] loss: {0:2.6f} | [POS] acc: {1:3.4f}'\
                         .format(test_loss, test_acc) + '\r')        
    name = "test_metrics.json"
    if opt.mode != "test":
        name = f"{opt.mode}_test_metrics.json"
    with open(os.path.join(opt.output_dir, name), mode="w", encoding="utf-8") as fp:
        res_mp = {"test_acc": test_acc}
        json.dump(res_mp, fp)

    write_pos_pred_json(test_loader.dataset.samples, test_logs, os.path.join(opt.output_dir, "test_pred.json"))
    return



if __name__ == "__main__":
    opt = add_args()
    main(opt)