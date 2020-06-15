import  torch, os, random, logging, json
import  numpy as np
from    pathlib import Path
from copy import deepcopy
import  argparse, time

from preprocessor import Corpus, Reprer, LABEL_LIST
from learner import Learner

#########################################################
# Train a model on the source language
#########################################################

def train_NOmeta(args):
    logger.info("********** Scheme: NO Meta Learning **********")

    # prepare dataset, here we take English as the source language.
    corpus_en_train = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language='en', mode='train',
                             load_data=False, support_size=-1, base_features=None, mask_rate=args.mask_rate,
                             compute_repr=False, shuffle=True)

    corpus_en_valid = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language='en', mode='valid',
                             load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                             compute_repr=False, shuffle=False)
    # build the model
    learner = Learner(args.bert_model, corpus_en_train.label_list, args.freeze_layer, logger, args.lr_meta, args.lr_inner,
                      args.warmup_prop_meta, args.warmup_prop_inner, args.max_meta_steps).to(device)


    t = time.time()
    best_en_valid_F1 = -1.0
    best_step = -1.0

    for step in range(args.max_meta_steps):

        batch_data = corpus_en_train.get_batch_NOmeta(batch_size=args.inner_size)
        loss = learner.forward_NOmeta(batch_data, lambda_max_loss=args.lambda_max_loss, lambda_mask_loss=args.lambda_mask_loss)

        if step % 20 == 0:
            logger.info('Step: {}/{}, loss = {:.6f}, time = {:.2f}s.'.format(step, args.max_meta_steps, loss, time.time() - t))

        if step % args.eval_every_meta_steps == 0:
            logger.info("********** Scheme: evaluate [en] - [valid] **********")
            F1_valid = learner.evaluate_NOmeta(corpus_en_valid, args.result_dir, logger)
            if F1_valid > best_en_valid_F1:
                logger.info("===> Best Valid F1: {}".format(F1_valid))
                logger.info("  Saving model...".format(F1_valid))
                learner.save_model(args.result_dir, 'en', args.max_seq_len)
                best_en_valid_F1 = F1_valid
                best_step = step
            else:
                logger.info("===> Valid F1: {}".format(F1_valid))

    logger.info('Best Valid F1: {}, Step: {}'.format(best_en_valid_F1, best_step))

def train_meta(args):
    logger.info("********** Scheme: Meta Learning **********")

    ## prepare dataset
    corpus_en_train = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language='en', mode='train',
                             load_data=True, support_size=args.support_size, base_features=None, mask_rate=args.mask_rate,
                             compute_repr=True, shuffle=True)

    corpus_en_valid = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language='en', mode='valid',
                             load_data=True, support_size=-1, base_features=None, mask_rate=-1.0,
                             compute_repr=False, shuffle=False)


    learner = Learner(args.bert_model, corpus_en_train.label_list, args.freeze_layer, logger, args.lr_meta, args.lr_inner,
                      args.warmup_prop_meta, args.warmup_prop_inner, args.max_meta_steps).to(device)

    t = time.time()
    best_en_valid_F1 = -1.0
    best_step = -1.0

    for step in range(args.max_meta_steps):
        progress = 1.0 * step / args.max_meta_steps

        batch_query, batch_support = corpus_en_train.get_batch_meta(batch_size=args.inner_size)#(batch_size=32)
        loss = learner.forward_meta(batch_query, batch_support, progress=progress, inner_steps=args.inner_steps,
                                    lambda_max_loss=args.lambda_max_loss, lambda_mask_loss=args.lambda_mask_loss)

        if step % 20 ==0:
            logger.info('Step: {}/{}, loss = {:.6f}, time = {:.2f}s.'.format(step, args.max_meta_steps, loss, time.time()-t))

        if step % args.eval_every_meta_steps == 0:
            logger.info("********** Scheme: evaluate [en] - [valid] **********")
            F1_valid = learner.evaluate_NOmeta(corpus_en_valid, args.result_dir, logger)
            if F1_valid > best_en_valid_F1:
                logger.info("===> Best Valid F1: {}".format(F1_valid))
                logger.info("  Saving model...".format(F1_valid))
                learner.save_model(args.result_dir, 'en', args.max_seq_len)
                best_en_valid_F1 = F1_valid
                best_step = step
            else:
                logger.info("===> Valid F1: {}".format(F1_valid))

    logger.info('Best Valid F1: {}, Step: {}'.format(best_en_valid_F1, best_step))


#########################################################
# Transfer the source-trained model to target languages
#########################################################

def zero_shot_NOmeta(args):
    res_filename = '{}/res-0shot-NOmeta-{}.json'.format(args.model_dir, '_'.join(args.test_langs))
    if os.path.exists(res_filename):
        assert False, 'Already evaluated.'

    logger.info("********** Scheme: 0-shot NO meta learning **********")

    # build the model
    learner = Learner(args.bert_model, LABEL_LIST, args.freeze_layer, logger, lr_meta=0, lr_inner=0,
                      warmup_prop_meta=-1, warmup_prop_inner=-1, max_meta_steps=-1,
                      model_dir=args.model_dir, gpu_no=args.gpu_device).to(device)

    languages = args.test_langs
    F1s = {lang: [] for lang in languages}
    for lang in languages:
        corpus_test = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='test',
                            load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                            compute_repr=False, shuffle=False)

        logger.info("********** Scheme: evaluate [{}] - [test] **********".format(lang))
        F1_test = learner.evaluate_NOmeta(corpus_test, args.result_dir, logger, lang=lang, mode='test')

        F1s[lang].append(F1_test)
        logger.info("===> Test F1: {}".format(F1_test))

    for lang in languages:
        logger.info('{} Test F1: {}'.format(lang, ', '.join([str(i) for i in F1s[lang]])))

    with Path(res_filename).open('w', encoding='utf-8') as fw:
        json.dump(F1s, fw, indent=4, sort_keys=True)

def zero_shot_meta(args):
    res_filename = '{}/res-0shot-ftLr_{}-ftSteps_{}-spSize_{}-maxLoss_{}-{}.json'.format(args.model_dir, args.lr_finetune,
                        args.max_ft_steps, args.support_size, args.lambda_max_loss, '_'.join(args.test_langs))
    if os.path.exists(res_filename):
        assert False, 'Already evaluated.'

    logger.info("********** Scheme: 0-shot with meta learning (separate support set) **********")

    ## prepare dataset
    reprer = Reprer(args.bert_model)
    corpus_en_train = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language='en', mode='train',
                            load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                            compute_repr=True, shuffle=False, reprer=reprer)

    learner = Learner(args.bert_model, LABEL_LIST, args.freeze_layer, logger, args.lr_meta,
                      args.lr_inner, args.warmup_prop_meta, args.warmup_prop_inner, args.max_meta_steps,
                      model_dir=args.model_dir, gpu_no=args.gpu_device).to(device)

    languages = args.test_langs
    F1s = {lang: [] for lang in languages}
    for lang in languages:
        corpus_test = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='test',
                            load_data=False, support_size=args.support_size, base_features=corpus_en_train.original_features,
                            mask_rate=-1.0, compute_repr=True, shuffle=False, reprer=reprer)

        logger.info("********** Scheme: evaluate [{}] - [test] - support on [en] **********".format(lang))
        F1_test = learner.evaluate_meta(corpus_test, args.result_dir, logger, lr=args.lr_finetune, steps=args.max_ft_steps,
                                        lambda_max_loss=args.lambda_max_loss, lambda_mask_loss=args.lambda_mask_loss,
                                        lang=lang, mode='test')

        F1s[lang].append(F1_test)
        logger.info("===> Test F1: {}".format(F1_test))

    for lang in languages:
        logger.info('{} Test F1: {}'.format(lang, ', '.join([str(i) for i in F1s[lang]])))

    with Path(res_filename).open('w', encoding='utf-8') as fw:
        json.dump(F1s, fw, indent=4, sort_keys=True)

def k_shot(args):
    # to define: k_shot, max_ft_steps, lr_finetune, lambda_max_loss
    res_filename = '{}/res-{}shot-ftLr_{}-ftSteps_{}-maxLoss_{}-{}.json'.format(args.model_dir, args.k_shot, args.lr_finetune,
                                    args.max_ft_steps, args.lambda_max_loss, '_'.join(args.test_langs))
    if os.path.exists(res_filename):
        assert False, 'Already evaluated.'

    logger.info("********** Scheme: {}-shot fine-tuning **********".format(args.k_shot))

    learner_pretrained = Learner(args.bert_model, LABEL_LIST, args.freeze_layer, logger, lr_meta=0, lr_inner=0,
                                 warmup_prop_meta=-1, warmup_prop_inner=-1, max_meta_steps=-1,
                                 model_dir=args.model_dir, gpu_no=args.gpu_device).to(device)

    languages = args.test_langs
    F1s = {lang: [] for lang in languages}

    for lang in languages:
        corpus_train = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='train',
                              load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                              compute_repr=False, shuffle=True, k_shot_prop=args.k_shot) # add k_shot_prop
        corpus_test = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='test',
                             load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                             compute_repr=False, shuffle=False)

        # build the model
        learner = deepcopy(learner_pretrained)

        t = time.time()
        for ft_step in range(args.max_ft_steps):
            data_batches = corpus_train.get_batches(args.inner_size, device="cuda", shuffle=True)

            for batch_data in data_batches:
                loss = learner.inner_update(batch_data, lr_curr=args.lr_finetune, inner_steps=1,
                                            lambda_max_loss=args.lambda_max_loss, lambda_mask_loss=args.lambda_mask_loss)

            if ft_step in [0, 4, 9, 14]:
                logger.info('Fine-tune Step: {}/{}, loss = {:8f}, time = {:2f}s.'.format(ft_step, args.max_ft_steps, loss, time.time() - t))
                logger.info("********** Scheme: evaluate [{}] - [test], Finetune step = {} **********".format(lang, ft_step))
                F1_test = learner.evaluate_NOmeta(corpus_test, args.result_dir, logger, lang=lang, mode='test')
                F1s[lang].append(F1_test)
                logger.info("===> Test F1: {}".format(F1_test))

    for i, lang in enumerate(languages):
        logger.info('{} Test F1: {}'.format(lang, ', '.join([str(i) for i in F1s[lang]])))

    with Path(res_filename).open('w', encoding='utf-8') as fw:
        json.dump(F1s, fw, indent=4, sort_keys=True)

def supervised_NOmeta(args):
    logger.info("********** Scheme: Supervised & NO Meta Learning **********")
    lang = args.test_langs[0]
    logger.info("language: {}".format(lang))

    # prepare dataset
    corpus_train = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='train',
                             load_data=False, support_size=-1, base_features=None, mask_rate=args.mask_rate,
                             compute_repr=False, shuffle=True)

    corpus_test = Corpus('bert-base-multilingual-cased', args.max_seq_len, logger, language=lang, mode='test',
                             load_data=False, support_size=-1, base_features=None, mask_rate=-1.0,
                             compute_repr=False, shuffle=False)
    # build the model
    learner = Learner(args.bert_model, corpus_train.label_list, args.freeze_layer, logger, args.lr_meta, args.lr_inner,
                      args.warmup_prop_meta, args.warmup_prop_inner, args.max_meta_steps).to(device)


    t = time.time()
    best_en_valid_F1 = -1.0
    best_step = -1.0

    for step in range(args.max_meta_steps):

        batch_data = corpus_train.get_batch_NOmeta(batch_size=args.inner_size)
        loss = learner.forward_NOmeta(batch_data, lambda_max_loss=args.lambda_max_loss, lambda_mask_loss=args.lambda_mask_loss)

        if step % 20 == 0:
            logger.info('Step: {}/{}, loss = {:.6f}, time = {:.2f}s.'.format(step, args.max_meta_steps, loss, time.time() - t))

        if step % args.eval_every_meta_steps == 0:
            logger.info("********** Scheme: evaluate [{}] - [test] **********".format(lang))
            F1_valid = learner.evaluate_NOmeta(corpus_test, args.result_dir, logger)
            if F1_valid > best_en_valid_F1:
                logger.info("===> Best Test F1: {}".format(F1_valid))
                logger.info("  Saving model...".format(F1_valid))
                learner.save_model(args.result_dir, 'en', args.max_seq_len)
                best_en_valid_F1 = F1_valid
                best_step = step
            else:
                logger.info("===> Test F1: {}".format(F1_valid))

    logger.info('Best Test F1: {}, Step: {}'.format(best_en_valid_F1, best_step))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--result_dir', type=str, help='where to save the result.', default='test')
    parser.add_argument('--test_langs', type=str, nargs='+', help='languages to test')

    parser.add_argument('--model_dir', type=str, help='dir name of a trained model', default='')

    # activate zero_shot only
    parser.add_argument('--zero_shot', action='store_true', help='if true, will run 0-shot procedure only.')
    # activate fine-tune only
    parser.add_argument('--k_shot', type=float, default=-1, help='size of k-shot data: k, if >0,  will run fine-ture procedure')
    parser.add_argument('--lr_finetune', type=float, help='finetune learning rate, used in [test_meta]. and [k_shot setting]', default=1e-5)
    parser.add_argument('--max_ft_steps', type=int, help='maximal steps token for fine-tune.', default=1) # ===>

    # activate mBERT only
    parser.add_argument('--no_meta_learning', action='store_true', help='if true, will run mBERT only.')
    parser.add_argument('--supervised', action='store_true', help='if true, will run mBERT only.')

    # meta-learning
    parser.add_argument('--inner_steps', type=int, help='every ** inner update for one meta-update', default=2) # ===>
    parser.add_argument('--inner_size', type=int, help='[number of tasks] for one meta-update', default=32)
    parser.add_argument('--support_size', type=int, help='support size (batch_size) for inner update', default=2)
    parser.add_argument('--lr_inner', type=float, help='inner loop learning rate', default=3e-5)
    parser.add_argument('--lr_meta', type=float, help='meta learning rate', default=3e-5)
    parser.add_argument('--max_meta_steps', type=int, help='maximal steps token for meta training.', default=3001)
    parser.add_argument('--eval_every_meta_steps', type=int, default=300)
    parser.add_argument('--warmup_prop_inner', type=int, help='warm up proportion for inner update', default=0.1)
    parser.add_argument('--warmup_prop_meta', type=int, help='warm up proportion for meta update', default=0.1)
    # parser.add_argument('--cross_meta_rate', type=float, help='when > 0, randomly flipping cross objective or normal objective', default=1.0)


    # training paramters
    parser.add_argument('--mask_rate', type=float, help='the probability to [mask] a token with a B/I-XXX label.', default=-1.0)
    parser.add_argument('--lambda_max_loss', type=float, help='the weight of the max-loss.', default=0.0)
    parser.add_argument('--lambda_mask_loss', type=float, help='the weight of the mask-loss.', default=0.0)


    # permanent params
    parser.add_argument('--freeze_layer', type=int, help='the layer of mBERT to be frozen', default=3)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--bert_model', type=str, default='bert-base-multilingual-cased', #required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--cache_dir', type=str, help='Where do you want to store the pre-trained models downloaded from s3',default='')
    # expt setting
    parser.add_argument('--seed', type=int, help='random seed to reproduce the result.', default=667)
    parser.add_argument('--gpu_device', type=int, help='GPU device num', default=0)

    args = parser.parse_args()


    # setup random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


    # set up GPU device
    device = torch.device('cuda')
    torch.cuda.set_device(args.gpu_device)

    # setup logger settings
    if args.zero_shot:
        assert args.model_dir != '' and os.path.exists(args.model_dir) and len(args.test_langs) > 0
        if args.no_meta_learning:
            fh = logging.FileHandler('{}/log-0shot-NOmeta-{}.txt'.format(args.model_dir, '_'.join(args.test_langs)))
        else:
            fh = logging.FileHandler('{}/log-0shot-ftLr_{}-ftSteps_{}-spSize_{}-maxLoss_{}-{}.txt'.format(
                args.model_dir, args.lr_finetune, args.max_ft_steps, args.support_size, args.lambda_max_loss, '_'.join(args.test_langs)))

        # dump args
        with Path('{}/args-0shot-ftLr_{}-ftSteps_{}-spSize_{}-maxLoss_{}-{}.json'.format(args.model_dir,
                args.lr_finetune, args.max_ft_steps, args.support_size, args.lambda_max_loss, '_'.join(args.test_langs))).open('w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
        args.result_dir = args.model_dir

    elif args.k_shot > 0:
        assert args.model_dir != '' and os.path.exists(args.model_dir) and len(args.test_langs) > 0
        fh = logging.FileHandler('{}/log-{}shot-ftLr_{}-ftSteps_{}-maxLoss_{}-{}.txt'.format(args.model_dir, args.k_shot, args.lr_finetune, args.max_ft_steps, args.lambda_max_loss, '_'.join(args.test_langs)))


        # dump args
        with Path('{}/args-{}shot-ftLr_{}-ftSteps_{}-maxLoss_{}-{}.json'.format(args.model_dir, args.k_shot, args.lr_finetune, args.max_ft_steps, args.lambda_max_loss, '_'.join(args.test_langs))).open('w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
        args.result_dir = args.model_dir
    elif args.supervised:
        assert args.model_dir == ''
        # top_dir = 'models\\result-{}'.format(args.expt_comment)
        top_dir = 'models'
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        if not os.path.exists('{}/{}-{}'.format(top_dir, args.result_dir, args.test_langs[0])):
            os.mkdir('{}/{}-{}'.format(top_dir, args.result_dir, args.test_langs[0]))
        elif args.result_dir != 'test':
            assert False, 'Existing result directory!'

        args.result_dir = '{}/{}-{}'.format(top_dir, args.result_dir, args.test_langs[0])

        fh = logging.FileHandler('{}/log-training.txt'.format(args.result_dir))

        # dump args
        with Path('{}/args.json'.format(args.result_dir)).open('w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
    else:
        assert args.model_dir == ''
        # top_dir = 'models\\result-{}'.format(args.expt_comment)
        top_dir = 'models'
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        if not os.path.exists('{}/{}'.format(top_dir, args.result_dir)):
            os.mkdir('{}/{}'.format(top_dir, args.result_dir))
        elif args.result_dir != 'test':
            assert False, 'Existing result directory!'

        args.result_dir = '{}/{}'.format(top_dir, args.result_dir)

        fh = logging.FileHandler('{}/log-training.txt'.format(args.result_dir))

        # dump args
        with Path('{}/args-train.json'.format(args.result_dir)).open('w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


    if args.zero_shot:
        if args.no_meta_learning:
            zero_shot_NOmeta(args)
        else:
            zero_shot_meta(args)
    elif args.k_shot > 0:
        k_shot(args)
    elif args.supervised:
        supervised_NOmeta(args)
    else:
        if args.no_meta_learning:
            train_NOmeta(args)
        else:
            train_meta(args)