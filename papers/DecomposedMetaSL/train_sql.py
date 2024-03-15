import torch
import numpy as np
import argparse
import os
from util.seq_loader import get_seq_loader as get_ment_loader
from trainer.ment_trainer import MentTrainer
from util.log_utils import eval_ment_log, write_ep_ment_log_json, save_json, set_seed
from util.span_encoder import BERTSpanEncoder
from model.ment_model import MentSeqtagger

def add_args():
    def str2bool(arg):
        if arg.lower() == "true":
            return True
        return False
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str,
                        help='train / test')
    parser.add_argument("--use_episode", default=False, type=str2bool)
    parser.add_argument("--bio", default=False, type=str2bool)
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
    parser.add_argument('--encoder_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--word_encode_choice', default='first', type=str)

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--last_n_layer', default=-4, type=int)
    parser.add_argument('--schema', default='IO', type=str, choices=['IO', 'BIO', 'BIOES'])
    parser.add_argument('--use_crf', type=str2bool, default=False)
    parser.add_argument('--max_loss', default=0, type=float)

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout rate')

    parser.add_argument('--log_steps', default=2000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--val_steps', default=2000, type=int,
                        help='val after training how many iters')

    parser.add_argument('--train_batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--eval_batch_size', default=16, type=int,
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

    parser.add_argument('--ep_dir', default=None, type=str)
    parser.add_argument('--ep_train', default=None, type=str)
    parser.add_argument('--ep_val', default=None, type=str)
    parser.add_argument('--ep_test', default=None, type=str)
    parser.add_argument('--eval_all_after_train', default=False, type=str2bool)
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
        train_loader = get_ment_loader(os.path.join(opt.root, opt.train), "train", word_encoder, batch_size=opt.train_batch_size, max_length=opt.max_length,
                                  schema=opt.schema,
                                  shuffle=True,
                                  bio=opt.bio, debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None)
    else:
        train_loader = get_ment_loader(os.path.join(opt.root, opt.train), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  schema=opt.schema,
                                  shuffle=False,
                                  bio=opt.bio, debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None)
    dev_loader = get_ment_loader(os.path.join(opt.root, opt.val), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  schema=opt.schema,
                                  shuffle=False,
                                  bio=opt.bio, debug_file=os.path.join(opt.ep_dir, opt.ep_val) if opt.ep_val else None)
    test_loader = get_ment_loader(os.path.join(opt.root, opt.test), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                  schema=opt.schema,
                                  shuffle=False,
                                  bio=opt.bio, debug_file=os.path.join(opt.ep_dir, opt.ep_test) if opt.ep_test else None)
    
    print("max_length: {}".format(opt.max_length))
    print('mode: {}'.format(opt.mode))
    
    print("mention detection sequence labeling with max loss {}".format(opt.max_loss))
    model = MentSeqtagger(word_encoder, len(test_loader.dataset.ment_label2tag), test_loader.dataset.ment_tag2label, opt.schema, opt.use_crf, opt.max_loss)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    trainer = MentTrainer()
    assert opt.eval_batch_size == 1

    if opt.mode == 'train':
        print("==================start training==================")
        trainer.train(model, opt, device, train_loader, dev_loader, eval_log_fn=os.path.join(opt.output_dir, "dev_ment_log.txt"))
        if opt.eval_all_after_train:
            train_loader = get_ment_loader(os.path.join(opt.root, opt.train), "test", word_encoder, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                                    schema=opt.schema,
                                    shuffle=False,
                                    bio=opt.bio, debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None)
            load_ckpt = os.path.join(opt.output_dir, "model.pth.tar")

            _, dev_p, dev_r, dev_f1, _, _, _, dev_logs = trainer.eval(model, device, dev_loader, load_ckpt=load_ckpt, update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_iter=opt.dev_iter)
            print("Dev precison {:.5f} recall {:.5f} f1 {:.5f}".format(dev_p, dev_r, dev_f1))
            eval_ment_log(dev_loader.dataset.samples, dev_logs)
            write_ep_ment_log_json(dev_loader.dataset.samples, dev_logs, os.path.join(opt.output_dir, "dev_ment.json"))

            _, test_p, test_r, test_f1, _, _, _, test_logs = trainer.eval(model, device, test_loader, load_ckpt=load_ckpt, update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_iter=opt.test_iter)
            print("Test precison {:.5f} recall {:.5f} f1 {:.5f}".format(test_p, test_r, test_f1))
            eval_ment_log(test_loader.dataset.samples, test_logs)
            write_ep_ment_log_json(test_loader.dataset.samples, test_logs, os.path.join(opt.output_dir, "test_ment.json"))
    else:
        _, dev_p, dev_r, dev_f1, _, _, _, dev_logs = trainer.eval(model, device, dev_loader, load_ckpt=opt.load_ckpt, update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_iter=opt.dev_iter)
        print("Dev precison {:.5f} recall {:.5f} f1 {:.5f}".format(dev_p, dev_r, dev_f1))
        eval_ment_log(dev_loader.dataset.samples, dev_logs)
        write_ep_ment_log_json(dev_loader.dataset.samples, dev_logs, os.path.join(opt.output_dir, "dev_ment.json"))

        _, test_p, test_r, test_f1, _, _, _, test_logs = trainer.eval(model, device, test_loader, load_ckpt=opt.load_ckpt, update_iter=opt.eval_inner_steps, learning_rate=opt.eval_inner_lr, eval_iter=opt.test_iter)
        model.timer.avg()
        print("Test precison {:.5f} recall {:.5f} f1 {:.5f}".format(test_p, test_r, test_f1))
        eval_ment_log(test_loader.dataset.samples, test_logs)
        write_ep_ment_log_json(test_loader.dataset.samples, test_logs, os.path.join(opt.output_dir, "test_ment.json"))
    res_mp = {"dev_p": dev_p, "dev_r": dev_r, "dev_f1": dev_f1, "test_p": test_p, "test_r": test_r, "test_f1": test_f1}
    save_json(res_mp, os.path.join(opt.output_dir, "metrics.json"))
    return

if __name__ == "__main__":
    opt = add_args()
    main(opt)