import sys
import torch
import numpy as np
import json
import argparse
import os
import random
from util.joint_loader import get_joint_loader
from trainer.joint_trainer import JointTrainer
from util.span_encoder import BERTSpanEncoder
from util.log_utils import eval_ent_log, cal_episode_prf, set_seed, save_json
from model.joint_model import SelectedJointModel

def add_args():
    def str2bool(arg):
        if arg.lower() == "true":
            return True
        return False

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str,
                        help='train / test')
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
    parser.add_argument('--train_ment_fn', default=None,
                        help='train file')
    parser.add_argument('--val_ment_fn', default=None,
                        help='val file')
    parser.add_argument('--test_ment_fn', default=None,
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
    parser.add_argument('--span_encode_choice', default=None, type=str)

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--max_span_len', default=8, type=int)
    parser.add_argument('--max_neg_ratio', default=5, type=int)
    parser.add_argument('--last_n_layer', default=-4, type=int)


    parser.add_argument('--dot', type=str2bool, default=False,
                        help='use dot instead of L2 distance for knn')
    parser.add_argument("--normalize", default='none', type=str, choices=['none', 'l2'])
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--use_width", default=False, type=str2bool)
    parser.add_argument("--width_dim", default=20, type=int)
    parser.add_argument("--use_case", default=False, type=str2bool)
    parser.add_argument("--case_dim", default=20, type=int)

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
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="lr rate")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")
    parser.add_argument("--bert_learning_rate", default=5e-5, type=float, help="lr rate")
    parser.add_argument("--bert_weight_decay", default=1e-5, type=float, help="weight decay")

    parser.add_argument("--warmup_step", default=0, type=int)

    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use nvidia apex fp16')
    parser.add_argument('--use_focal', default=False, type=str2bool)
    parser.add_argument('--iou_thred', default=None, type=float)
    parser.add_argument('--use_att', default=False, type=str2bool)
    parser.add_argument('--att_hidden_dim', default=-1, type=int)
    parser.add_argument('--label_fn', default=None, type=str)
    parser.add_argument('--hou_eval_ep', default=False, type=str2bool)

    parser.add_argument('--use_maml', default=False, type=str2bool)
    parser.add_argument('--warmup_prop_inner', default=0, type=float)
    parser.add_argument('--train_inner_lr', default=0, type=float)
    parser.add_argument('--train_inner_steps', default=0, type=int)
    parser.add_argument('--eval_inner_lr', default=0, type=float)
    parser.add_argument('--eval_type_inner_steps', default=0, type=int)
    parser.add_argument('--eval_ment_inner_steps', default=0, type=int)
    parser.add_argument('--overlap', default=False, type=str2bool)
    parser.add_argument('--type_lam', default=1, type=float)
    parser.add_argument('--use_adapter', default=False, type=str2bool)
    parser.add_argument('--adapter_size', default=64, type=int)
    parser.add_argument('--type_threshold', default=-1, type=float)
    parser.add_argument('--use_oproto', default=False, type=str2bool)
    parser.add_argument('--bio', default=False, type=str2bool)
    parser.add_argument('--schema', default='IO', type=str, choices=['IO', 'BIO', 'BIOES'])
    parser.add_argument('--use_crf', type=str2bool, default=False)
    parser.add_argument('--max_loss', default=0, type=float)
    parser.add_argument('--adapter_layer_ids', default='9-10-11', type=str)
    opt = parser.parse_args()
    return opt


def main(opt):
    print("Joint Model pipeline {} way {} shot".format(opt.N, opt.K))
    set_seed(opt.seed)
    if opt.mode == "train":
        output_dir = opt.output_dir
        print("Output dir is ", output_dir)
        log_dir = os.path.join(
            output_dir,
            "logs",
        )
        opt.log_dir = log_dir
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        save_json(opt.__dict__, os.path.join(opt.output_dir, "train_setting.txt"))
    else:
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        save_json(opt.__dict__, os.path.join(opt.output_dir, "test_setting.txt"))
    print('loading model and tokenizer...')
    print("use adapter: ", opt.use_adapter)
    word_encoder = BERTSpanEncoder(opt.encoder_name_or_path, opt.max_length, last_n_layer=opt.last_n_layer,
                                   word_encode_choice=opt.word_encode_choice,
                                   span_encode_choice=opt.span_encode_choice,
                                   use_width=opt.use_width, width_dim=opt.width_dim, use_case=opt.use_case,
                                   case_dim=opt.case_dim,
                                   drop_p=opt.dropout, use_att=opt.use_att,
                                   att_hidden_dim=opt.att_hidden_dim, use_adapter=opt.use_adapter, adapter_size=opt.adapter_size,
                                   adapter_layer_ids=[int(x) for x in opt.adapter_layer_ids.split('-')])
    print('loading data')
    if opt.mode == "train":
        train_loader = get_joint_loader(os.path.join(opt.root, opt.train), "train", word_encoder, N=opt.N,
                                  K=opt.K,
                                  Q=opt.Q, batch_size=opt.train_batch_size, max_length=opt.max_length,
                                  shuffle=True,
                                  bio=opt.bio,
                                  schema=opt.schema,
                                  debug_file=os.path.join(opt.ep_dir, opt.ep_train) if opt.ep_train else None,
                                  query_file=opt.train_ment_fn,
                                  iou_thred=opt.iou_thred,
                                  use_oproto=opt.use_oproto,
                                  label_fn=opt.label_fn)
    print(os.path.join(opt.root, opt.val))
    dev_loader = get_joint_loader(os.path.join(opt.root, opt.val), "test", word_encoder, N=opt.N, K=opt.K,
                            Q=opt.Q, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                            shuffle=False,
                            bio=opt.bio,
                            schema=opt.schema,
                            debug_file=os.path.join(opt.ep_dir, opt.ep_val) if opt.ep_val else None,
                            query_file=opt.val_ment_fn,
                            iou_thred=opt.iou_thred,
                            use_oproto=opt.use_oproto,
                            label_fn=opt.label_fn)

    test_loader = get_joint_loader(os.path.join(opt.root, opt.test), "test", word_encoder, N=opt.N, K=opt.K,
                             Q=opt.Q, batch_size=opt.eval_batch_size, max_length=opt.max_length,
                             shuffle=False,
                             bio=opt.bio,
                            schema=opt.schema,
                             debug_file=os.path.join(opt.ep_dir, opt.ep_test) if opt.ep_test else None,
                             query_file=opt.test_ment_fn,
                             iou_thred=opt.iou_thred, hidden_query_label=False,
                             use_oproto=opt.use_oproto,
                             label_fn=opt.label_fn)

    print("max_length: {}".format(opt.max_length))
    print('mode: {}'.format(opt.mode))

    print("{}-way-{}-shot Proto Few-Shot NER".format(opt.N, opt.K))

    model = SelectedJointModel(word_encoder, num_tag=len(test_loader.dataset.ment_label2tag),
                    ment_label2idx=test_loader.dataset.ment_tag2label,
                    schema=opt.schema, use_crf=opt.use_crf, max_loss=opt.max_loss,
                    use_oproto=opt.use_oproto, dot=opt.dot, normalize=opt.normalize,
                    temperature=opt.temperature, use_focal=opt.use_focal, type_lam=opt.type_lam)

    num_params = sum(param.numel() for param in model.parameters())
    print("total parameter numbers: ", num_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    trainer = JointTrainer()
    assert opt.eval_batch_size == 1

    if opt.mode == 'train':
        trainer.train(model, opt, device, train_loader, dev_loader, dev_pred_fn=os.path.join(opt.output_dir, "dev_metrics.json"), dev_log_fn=os.path.join(opt.output_dir, "dev_log.txt"), load_ckpt=opt.load_ckpt)
        test_loss, test_ment_p, test_ment_r, test_ment_f1, test_p, test_r, test_f1, test_ment_logs, test_logs = trainer.eval(model, device, test_loader, eval_iter=opt.test_iter, load_ckpt=os.path.join(opt.output_dir, "model.pth.tar"),
                                    ment_update_iter=opt.eval_ment_inner_steps, 
                                    type_update_iter=opt.eval_type_inner_steps,
                                    learning_rate=opt.eval_inner_lr, 
                                    overlap=opt.overlap, 
                                    threshold=opt.type_threshold, 
                                    eval_mode="test-twostage")
        if opt.hou_eval_ep:
            test_p, test_r, test_f1 = cal_episode_prf(test_logs)
        print("Mention test precision {:.5f} recall {:.5f} f1 {:.5f}".format(test_ment_p, test_ment_r, test_ment_f1))
        print('[TEST] loss: {0:2.6f} | [Entity] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'\
                         .format(test_loss, test_p, test_r, test_f1) + '\r')
    else:

        _, dev_ment_p, dev_ment_r, dev_ment_f1, dev_p, dev_r, dev_f1, dev_ment_logs, dev_logs = trainer.eval(model, device, dev_loader, load_ckpt=opt.load_ckpt,
                                                             eval_iter=opt.dev_iter, 
                                                             ment_update_iter=opt.eval_ment_inner_steps, 
                                                             type_update_iter=opt.eval_type_inner_steps,                                                             
                                                             learning_rate=opt.eval_inner_lr, overlap=opt.overlap, threshold=opt.type_threshold, eval_mode=opt.mode)

        if opt.hou_eval_ep:
            dev_p, dev_r, dev_f1 = cal_episode_prf(dev_logs)
        print("Mention dev precision {:.5f} recall {:.5f} f1 {:.5f}".format(dev_ment_p, dev_ment_r, dev_ment_f1))
        print("Dev precison {:.5f} recall {:.5f} f1 {:.5f}".format(dev_p, dev_r, dev_f1))
        with open(os.path.join(opt.output_dir, "dev_metrics.json"), mode="w", encoding="utf-8") as fp:
            json.dump({"ment_p": dev_ment_p, "ment_r": dev_ment_r, "ment_f1": dev_ment_f1, "precision": dev_p, "recall": dev_r, "f1": dev_f1}, fp)

        test_loss, test_ment_p, test_ment_r, test_ment_f1, test_p, test_r, test_f1, test_ment_logs, test_logs = trainer.eval(model, device, test_loader, eval_iter=opt.test_iter, load_ckpt=opt.load_ckpt,
                                                            ment_update_iter=opt.eval_ment_inner_steps, 
                                                            type_update_iter=opt.eval_type_inner_steps, learning_rate=opt.eval_inner_lr, overlap=opt.overlap, threshold=opt.type_threshold, eval_mode=opt.mode)
        if opt.hou_eval_ep:
            test_p, test_r, test_f1 = cal_episode_prf(test_logs)
        print("Mention test precision {:.5f} recall {:.5f} f1 {:.5f}".format(test_ment_p, test_ment_r, test_ment_f1))
        print("Test precison {:.5f} recall {:.5f} f1 {:.5f}".format(test_p, test_r, test_f1))


    with open(os.path.join(opt.output_dir, "test_metrics.json"), mode="w", encoding="utf-8") as fp:
        json.dump({"ment_p": test_ment_p, "ment_r": test_ment_r, "ment_f1": test_ment_f1, "precision": test_p, "recall": test_r, "f1": test_f1}, fp)
    eval_ent_log(test_loader.dataset.samples, test_logs)
    return


if __name__ == "__main__":
    opt = add_args()
    main(opt)

