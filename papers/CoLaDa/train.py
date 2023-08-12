# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, json
from utils.options import add_args
from trainers.ner_trainer import SrcNERTrainer
from trainers.ts_trainer import TSTrainer
from utils.utils_ner import set_seed

def main():
    args = add_args()
    print(args)

    set_seed(args.seed)

    if args.mode == "ner":
        trainer = SrcNERTrainer(args)
        if args.do_train:
            trainer.train_ner()
        elif args.do_eval:
            trainer.load_model(args.ckpt_dir)
            test_sents, _ = trainer.processor.read_examples_from_file(args.test_file)
            trainer.eval_ner(trainer.eval_dataset, out_file=os.path.join(args.output_dir, "test_pred.txt"), res_file=os.path.join(args.output_dir, "test_res.json"), ori_sents=test_sents)
    elif args.mode == 'step1':
        trainer = TSTrainer(args)
        if args.do_train:
            trainer.train_filter_trans()
    elif args.mode == 'step2':
        trainer = TSTrainer(args)
        if args.do_train:
            trainer.kd_train()
    else:
        raise NotImplementedError
    return

if __name__ == "__main__":
    main()