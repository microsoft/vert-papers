# coding=utf-8
# checked

from __future__ import print_function

import math
import optparse
import os
import pickle
import time
from datetime import datetime
import numpy as np
from shutil import copyfile

from data_loader_conll import DataLoaderCoNLL
from data_processing import *
from enums import *
from evaluation import evaluating
from model import GRN_CRF
from utils import *

t = time.time()

optparser = optparse.OptionParser()
optparser.add_option(
    '--name', default='GRN',
    help='Model name'
)
optparser.add_option(
    "--train", default="datasets/english/conll2003/eng.train.train",
    help="Train set location"
)
optparser.add_option(
    "--dev", default="datasets/english/conll2003/eng.testa.dev",
    help="Dev set location"
)
optparser.add_option(
    "--test", default="datasets/english/conll2003/eng.testb.test",
    help="Test set location"
)
optparser.add_option(
    '--test_train', default='datasets/english/conll2003/eng.train.subset',
    help='Location of a subset of the train set for testing'
)
optparser.add_option(
    "--tag_scheme", choices=['iob2', 'iobes'], default="iobes",
    help="Tagging scheme (IOB2 or IOBES)"
)
optparser.add_option(
    "--word_lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "--word_threshold", default="3",
    type='int',
    help="Only words with the corresponding threshold larger than or equal to word_threshold will be preserved"
)
optparser.add_option(
    "--digits_to_zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "--char_dim", default="30",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
optparser.add_option(
    '--char_embed_dropout', default='0', type='int',
    help='To use dropout for char embedding or not'
)
optparser.add_option(
    "--char_lstm_dim", default="30",
    type='int', help="Char embedding LSTM hidden layer size"
)
optparser.add_option(
    "--char_lstm_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for char embedding"
)
optparser.add_option(
    '--char_cnn_win', default='3',
    type='int', help='CNN win-size for char embedding'
)
optparser.add_option(
    '--char_cnn_dim', default='30',
    type='int', help='Dimensionality of the output of CNN for char embedding, i.e., number of CNN kernels'
)
optparser.add_option(
    "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "--word_lstm_dim", default="200",
    type='int', help="Token embedding LSTM hidden layer size"
)
optparser.add_option(
    "--word_lstm_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "--pre_emb", default="resources/word_embeddings/english/glove.6B.100d.txt",
    help="Location of pre-trained word embeddings"
)
optparser.add_option(
    "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    '--use_gpu', default='1',
    type='int', help='whether or not to use gpu'
)
optparser.add_option(
    '--max_epoch', default='200',
    type='int', help='Maximal number of epochs'
)
optparser.add_option(
    '--mini_batch_size', default='10',
    type='int', help='Mini-batch size'
)
optparser.add_option(
    '--cuda', default='0',
    type='int', help='Index of the cuda device'
)
optparser.add_option(
    '--optimizer', choices=['SGD', 'AdaDelta', 'Adam'], default='SGD',
    help='Optimizer, selected from [SGD, AdaDelta, Adam]'
)
optparser.add_option(
    "--verbose", default="0",
    type='int', help="Verbose mode or not"
)
optparser.add_option(
    "--manual_seed", default="-1",
    type='int', help="Manual random seed"
)
optparser.add_option(
    "--grad_norm", default="5.0",
    type='float', help="Clip norm-value for the gradient of each mini-batch"
)
optparser.add_option(
    "--sgd_lr", default="0.02",
    type='float', help="Starting learning rate for SGD"
)
optparser.add_option(
    "--sgd_momentum", default="0.9",
    type='float', help="Momentum for SGD"
)
optparser.add_option(
    "--sgd_decay_weight", default="0.02",
    type='float', help="Decay weight for learning rate of SGD, i.e., lr = lr /(1+decay_weight * epoch)"
)
optparser.add_option(
    "--inception_mode", default="1",
    type='int', help="Inception-CNN mode. 0: no cnn; 1: full inception cnn, default; 2: only one win-size=3 cnn"
)
optparser.add_option(
    "--enable_context", default="1",
    type='int', help="Enable context or not (1/0)"
)

# parse and check all the parameters
opts = optparser.parse_args()[0]

manual_seed = opts.manual_seed
rand_seed = manual_seed if manual_seed >=0 else torch.initial_seed() % 4294967295   # 2^32-1
opts.rand_seed = rand_seed
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)

opts_str = "{0}".format(opts)

assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert os.path.isfile(opts.test_train)
assert opts.tag_scheme.lower() in ['iob2', 'iobes']
assert opts.word_threshold >= 0
assert opts.char_dim > 0
assert opts.char_mode.lower() in ['cnn', 'lstm']
assert opts.word_dim > 0
assert opts.word_lstm_dim > 0
assert opts.dropout >= 0.0 and opts.dropout < 1.0
assert opts.max_epoch > 0
assert opts.mini_batch_size > 0
assert opts.cuda >= 0
assert opts.optimizer.lower() in ['sgd', 'adadelta', 'adam']
assert opts.sgd_lr > 0
assert opts.sgd_momentum >= 0 and opts.sgd_momentum <= 1
assert opts.sgd_decay_weight >= 0
assert opts.inception_mode in [0, 1, 2]

name = opts.name
train_set_path = opts.train
dev_set_path = opts.dev
test_set_path = opts.test
train_subset_path = opts.test_train
label_schema = LabellingSchema.IOB2 if opts.tag_scheme.lower() == "iob2" else LabellingSchema.IOBES
word_to_lower = opts.word_lower == 1
word_frequency_threshold = opts.word_threshold
digits_to_zeros = opts.digits_to_zeros == 1
char_dim = opts.char_dim
char_mode = CharEmbeddingSchema.CNN if opts.char_mode.lower() == "cnn" else CharEmbeddingSchema.LSTM
char_lstm_dim = opts.char_lstm_dim
char_lstm_bidirect = opts.char_lstm_bidirect == 1
char_cnn_win = opts.char_cnn_win
char_cnn_dim = opts.char_cnn_dim
word_dim = opts.word_dim
word_lstm_dim = opts.word_lstm_dim
word_lstm_bidirect = opts.word_lstm_bidirect == 1
prebuilt_embed_path = '' if not os.path.isfile(opts.pre_emb) else opts.pre_emb
use_crf = opts.crf == 1
dropout = opts.dropout
reload = opts.reload == 1
use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()
max_epoch = opts.max_epoch
mini_batch_size = opts.mini_batch_size
optimizer_choice = OptimizationMethod.Adam if opts.optimizer.lower() == 'adam' else \
    (OptimizationMethod.AdaDelta if opts.optimizer.lower() == 'adadelta' else OptimizationMethod.SGDWithDecreasingLR)
verbose = opts.verbose == 1
grad_norm = opts.grad_norm
sgd_lr = opts.sgd_lr
sgd_momentum = opts.sgd_momentum
sgd_decay_weight = opts.sgd_decay_weight
char_embed_dropout = opts.char_embed_dropout == 1
inception_mode = opts.inception_mode
enable_context = opts.enable_context == 1

device_count = torch.cuda.device_count() if use_gpu else 0
assert (not char_mode == CharEmbeddingSchema.LSTM) or char_lstm_dim > 0
assert (not char_mode == CharEmbeddingSchema.CNN) or (char_cnn_win > 0 and char_cnn_dim > 0)
assert (not use_gpu) or opts.cuda < device_count

name = name if reload else "{0}{1}".format(name, datetime.now().strftime('%Y%m%d%H%M%S'))

device_name = "cuda:{0}".format(opts.cuda) if use_gpu else "cpu"
device = torch.device(device_name)
print("Using device {0}".format(device))

if use_gpu:
    torch.cuda.set_device(opts.cuda)

models_path = Constants.Models_Folder
logs_path = Constants.Logs_Folder
eval_path = Constants.Eval_Folder
eval_temp = Constants.Eval_Temp_Folder
eval_script = Constants.Eval_Script

mapping_file = os.path.join(models_path, "{0}.mappings.pkl".format(name))
model_name = os.path.join(models_path, "{0}.model".format(name))
model_dir = model_name[:-(len('.model'))]
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

assert (not reload) or os.path.exists(model_name)

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# create the mappings
if os.path.exists(mapping_file):
    mappings = pickle.load(open(mapping_file, "rb"))
    prebuilt_word_embedding = load_prebuilt_word_embedding(prebuilt_embed_path,
                                                           word_dim)
else:
    mappings, prebuilt_word_embedding = create_mapping_dataset_conll(
        [train_set_path],
        prebuilt_embed_path,
        word_dim,
        label_schema,
        word_to_lower,
        word_frequency_threshold,
        digits_to_zeros)
    pickle.dump(mappings, open(mapping_file, "wb"))
print('Loaded {0} pretrained embeddings.'.format(len(prebuilt_word_embedding)))

# create data loaders
train_set = DataLoaderCoNLL(train_set_path, mappings)
dev_set = DataLoaderCoNLL(dev_set_path, mappings)
test_set = DataLoaderCoNLL(test_set_path, mappings)
train_subset = DataLoaderCoNLL(train_subset_path, mappings)

print("{0} / {1} / {2} sentences in train / dev / test.".format(
    len(train_set), len(dev_set), len(test_set)))

tag_to_id = mappings["tag_to_id"]
word_to_id = mappings["word_to_id"]
char_to_id = mappings["char_to_id"]
id_to_tag = mappings["id_to_tag"]

if prebuilt_word_embedding is not None:
    word_embeds = np.random.uniform(-np.sqrt(6/word_dim), np.sqrt(6/word_dim), (len(word_to_id), word_dim)) # Kaiming_uniform
    for w in word_to_id.keys():
        if w in prebuilt_word_embedding.keys():
            word_embeds[word_to_id[w], :] = prebuilt_word_embedding[w]
        elif w.lower() in prebuilt_word_embedding.keys():
            word_embeds[word_to_id[w], :] = prebuilt_word_embedding[w.lower()]
else:
    word_embeds = None

print('word_to_id: {0}'.format(len(word_to_id)))

model = GRN_CRF(word_set_size=len(word_to_id),
                   tag_to_id=tag_to_id,
                   word_embedding_dim=word_dim,
                   word_lstm_dim=word_lstm_dim,
                   word_lstm_bidirect=word_lstm_bidirect,
                   pre_word_embeds=word_embeds,
                   char_embedding_dim=char_dim,
                   char_mode=char_mode,
                   char_lstm_dim=char_lstm_dim,
                   char_lstm_bidirect=char_lstm_bidirect,
                   char_cnn_win=char_cnn_win,
                   char_cnn_output=char_cnn_dim,
                   char_to_id=char_to_id,
                   use_gpu=use_gpu,
                   dropout=dropout,
                   use_crf=use_crf,
                   char_embed_dropout=char_embed_dropout,
                   inception_mode=inception_mode,
                   enable_context=enable_context)

if reload:
    last_saved_model = torch.load(model_name, map_location=device_name)
    model.load_state_dict(last_saved_model.state_dict())
    model.use_gpu = use_gpu
if use_gpu:
    model = model.to(device)

# Perf: Adam < AdaDelta < SGD
if optimizer_choice == OptimizationMethod.SGDWithDecreasingLR:
    learning_rate = sgd_lr
    learning_momentum = sgd_momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)
elif optimizer_choice == OptimizationMethod.Adam:
    optimizer = torch.optim.Adam(model.parameters())
elif optimizer_choice == OptimizationMethod.AdaDelta:
    optimizer = torch.optim.Adadelta(model.parameters())

best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
test_F_with_best_dev = -1.0

batch_count = math.ceil(len(train_set) / mini_batch_size)

test_time_costs = []
train_time_costs = []

last_model = ''
last_model_epoch = -1
model.train(True)
for epoch in range(max_epoch):
    epoch_start = time.time()
    train_indecies = np.random.permutation(len(train_set))
    full_logs = []
    if epoch == 0:
        if verbose:
            print(opts_str)
        full_logs.append(opts_str)

    train_time_cost = 0
    avg_loss = 0.0
    for batch_i in range(batch_count):
        start_idx = batch_i * mini_batch_size
        end_idx = min((batch_i + 1) * mini_batch_size, len(train_set))

        mini_batch_idx = train_indecies[start_idx:end_idx]

        sentence_masks, words, chars, tags, \
        sentence_char_lengths, sentence_char_position_map, str_words, unaligned_tags = \
            generate_mini_batch_input(train_set, mini_batch_idx, mappings, char_mode)

        if use_gpu:
            sentence_masks = sentence_masks.to(device)
            words = words.to(device)
            chars = chars.to(device)
            tags = tags.to(device)
            sentence_char_lengths = sentence_char_lengths.to(device)

        train_start_time = time.time()

        model.zero_grad()

        neg_log_likelihood = model.neg_log_likelihood(words, sentence_masks, tags, chars,
                                                      sentence_char_lengths, sentence_char_position_map, device)
        neg_log_likelihood.backward()
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        train_time_cost += time.time() - train_start_time

        loss = neg_log_likelihood.data.item()
        avg_loss += loss / batch_count

        log = "epoch {0} batch {1}/{2} loss {3}".format(epoch + 1, batch_i + 1, batch_count, loss)
        if verbose:
            print(log)
        full_logs.append(log)
    train_time_costs.append(train_time_cost)

    model.eval()
    with torch.no_grad():
        best_train_F, new_train_F, _, __ = evaluating(model, train_subset, best_train_F, name, mappings, char_mode, use_gpu,
                                                  device, mini_batch_size)
        best_dev_F, new_dev_F, save, __ = evaluating(model, dev_set, best_dev_F, name, mappings, char_mode, use_gpu, device, mini_batch_size)
        best_test_F, new_test_F, _, eval_test_time = evaluating(model, test_set, best_test_F, name, mappings, char_mode, use_gpu,
                                                device, mini_batch_size)
        test_time_costs.append(eval_test_time)

        if save:
            model_path = os.path.join(model_dir, 'epoch_{0}.model'.format(epoch))
            torch.save(model, model_path)
            if epoch + 1 <= 50 and last_model_epoch >= 0:
                os.remove(last_model)
            elif epoch + 1 <= 100 and last_model_epoch + 1 > 50:
                os.remove(last_model)
            elif epoch + 1 <= 150 and last_model_epoch + 1 > 100:
                os.remove(last_model)
            elif epoch + 1 <= 200 and last_model_epoch + 1 > 150:
                os.remove(last_model)

            last_model = model_path
            last_model_epoch = epoch
            test_F_with_best_dev = new_test_F

        log = "Epoch {0}: [avg_loss, best_train_f1, train_f1, best_test_f1, test_f1, best_dev_f1, dev_f1, f1_for_best_dev] = [{1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}, {6:.5f}, {7:.5f}, {8:.5f}]" \
            .format(epoch + 1, avg_loss, best_train_F, new_train_F, best_test_F, new_test_F, best_dev_F, new_dev_F,
                    test_F_with_best_dev)
        log1 = opts_str if epoch == 0 else ''
        log = ("{0}\n\n{1}".format(log1, log)) if epoch == 0 else log

        print(log)

        full_logs.append(log)
        full_logs.append('\n')

        with open(os.path.join(logs_path, "{0}.full.log".format(name)), "a") as fout:
            fout.write('\n'.join(full_logs))
            fout.flush()

        with open(os.path.join(logs_path, "{0}.important.log".format(name)), "a") as fout:
            fout.write(log)
            fout.write('\n')
            fout.flush()

    model.train(True)

    if optimizer_choice == OptimizationMethod.SGDWithDecreasingLR:
        adjust_learning_rate(optimizer, lr=learning_rate / (1 + sgd_decay_weight * (epoch + 1)))

    print('Epoch time cost: {0}, train {1}, test {2}'.format(time.time() - epoch_start, train_time_costs[-1], test_time_costs[-1]))

if len(last_model) > 0:
    copyfile(last_model, model_name)

time_cost_log = 'Time cost: {0}, whole train {1}, avg train {2}, avg test {3}'\
    .format(time.time() - t, sum(train_time_costs), sum(train_time_costs)/len(train_time_costs),
            sum(test_time_costs)/len(test_time_costs))
print(time_cost_log)

with open(os.path.join(logs_path, "{0}.full.log".format(name)), "a") as fout:
    fout.write(time_cost_log)
    fout.flush()

with open(os.path.join(logs_path, "{0}.important.log".format(name)), "a") as fout:
    fout.write(time_cost_log)
    fout.flush()
