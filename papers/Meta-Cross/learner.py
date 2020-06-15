from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import CONFIG_NAME, WEIGHTS_NAME, BertConfig #, BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from modeling import BertForTokenClassification_

from torch import nn
from copy import deepcopy

import  torch, os, numpy, json, time

class Learner(nn.Module):
    def __init__(self, bert_model, label_list, freeze_layer, logger, lr_meta, lr_inner,
                 warmup_prop_meta, warmup_prop_inner, max_meta_steps, model_dir='', cache_dir='', gpu_no=0):

        super(Learner, self).__init__()

        self.lr_meta = lr_meta
        self.lr_inner = lr_inner
        self.warmup_prop_meta = warmup_prop_meta
        self.warmup_prop_inner = warmup_prop_inner
        self.max_meta_steps = max_meta_steps

        self.bert_model = bert_model
        self.label_list = label_list

        num_labels = len(label_list) + 1

        ## load model
        if model_dir != '':
            logger.info('********** Loading saved model **********')
            output_config_file = os.path.join(model_dir, CONFIG_NAME)
            output_model_file = os.path.join(model_dir, 'en_{}'.format(WEIGHTS_NAME))
            config = BertConfig(output_config_file)
            self.model = BertForTokenClassification_(config, num_labels=num_labels)
            self.model.load_state_dict(torch.load(output_model_file, map_location="cuda:{}".format(gpu_no)))
        else:
            logger.info('********** Loading pre-trained model **********')
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            self.model = BertForTokenClassification_.from_pretrained(bert_model, cache_dir=cache_dir, num_labels=num_labels)

        ## layer freezing
        if freeze_layer == 0:
            no_grad_param_names = ['embeddings'] # layer.0
        else:
            no_grad_param_names = ['embeddings', 'pooler'] + ['layer.{}.'.format(i) for i in range(freeze_layer + 1)]
        logger.info("The frozen parameters are:")
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info("  {}".format(name))

        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=lr_meta,
                                       warmup=warmup_prop_meta, t_total=max_meta_steps)


    def get_optimizer_grouped_parameters(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters


    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params


    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)

    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            model_params[n].grad.data.add_(g.data) # accumulate


    def get_learning_rate(self, lr, progress, warmup, schedule='linear'):
        if schedule == 'linear':
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress - 1.) / (warmup - 1.), 0.)

        return lr


    def inner_update(self, data_support, lr_curr, inner_steps, lambda_max_loss, lambda_mask_loss):
        inner_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_inner)
        self.model.train()

        for i in range(inner_steps):
            inner_opt.param_groups[0]['lr'] = lr_curr
            inner_opt.param_groups[1]['lr'] = lr_curr

            inner_opt.zero_grad()
            loss = self.model.forward_wuqh(data_support['input_ids'], data_support['segment_ids'],
                                           data_support['input_mask'], data_support['label_ids'],
                                           lambda_max_loss=lambda_max_loss, lambda_mask_loss=lambda_mask_loss)

            loss.backward()
            inner_opt.step()

        return loss.item()


    def forward_meta(self, batch_query, batch_support, progress, inner_steps, lambda_max_loss, lambda_mask_loss): # for one task
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        meta_grad = []
        meta_loss = []

        task_num = len(batch_query)
        lr_inner = self.get_learning_rate(self.lr_inner, progress, self.warmup_prop_inner)

        # compute meta_grad of each task
        for task_id in range(task_num):
            self.inner_update(batch_support[task_id], lr_inner, inner_steps=inner_steps,
                              lambda_max_loss=lambda_max_loss, lambda_mask_loss=lambda_mask_loss)
            loss = self.model.forward_wuqh(batch_query[task_id]['input_ids'], batch_query[task_id]['segment_ids'],
                                           batch_query[task_id]['input_mask'], batch_query[task_id]['label_ids'],
                                           lambda_max_loss=lambda_max_loss, lambda_mask_loss=lambda_mask_loss)

            grad = torch.autograd.grad(loss, params)
            meta_grad.append(grad)
            meta_loss.append(loss.item())

            self.load_weights(names, weights)

        # accumulate grads of all tasks to param.grad
        self.opt.zero_grad()

        # similar to backward()
        for g in meta_grad:
            self.load_gradients(names, g)
        self.opt.step()

        ave_loss = numpy.mean(numpy.array(meta_loss))

        return ave_loss


    def forward_NOmeta(self, batch_data, lambda_max_loss, lambda_mask_loss): #, lambda_flag=-1.0):
        self.model.train()
        self.opt.zero_grad()

        loss = self.model.forward_wuqh(batch_data['input_ids'], batch_data['segment_ids'],
                                       batch_data['input_mask'], batch_data['label_ids'],
                                       lambda_max_loss=lambda_max_loss,
                                       lambda_mask_loss=lambda_mask_loss) #, lambda_flag=lambda_flag)
        loss.backward()
        self.opt.step()

        return loss.item()

    ##---------------------------------------- Evaluation --------------------------------------##

    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, 'w', encoding='utf-8') as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write('{} {} {}\n'.format(words[i][j], word, y_pred[i][j]))
            fw.write('\n')

    def evaluate_meta(self, corpus, result_dir, logger, lr, steps, lambda_max_loss, lambda_mask_loss, lang='en', mode='valid'):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=False)

        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        y_true = []
        y_pred = []
        words = []
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}

        t_tmp = time.time()
        for item_id in range(corpus.n_total):
            eval_query, eval_support = corpus.get_batch_meta(batch_size=1, shuffle=False)

            # train on support examples
            self.inner_update(eval_support[0], lr_curr=lr, inner_steps=steps,
                              lambda_max_loss=lambda_max_loss, lambda_mask_loss=lambda_mask_loss)

            # eval on pseudo query examples (test example)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(eval_query[0]['input_ids'], eval_query[0]['segment_ids'],
                                       eval_query[0]['input_mask'])  # batch_size x seq_len x target_size

            logits = torch.argmax(torch.nn.functional.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            input_ids = eval_query[0]['input_ids'].to('cpu').numpy()
            label_ids = eval_query[0]['label_ids'].to('cpu').numpy()
            input_mask = eval_query[0]['input_mask'].to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                temp_word = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X":
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[max(logits[i][j], 1)])
                            temp_word.append(tokenizer.ids_to_tokens[input_ids[i][j]])
                        else:
                            tmp = tokenizer.ids_to_tokens[input_ids[i][j]]
                            if len(tmp) > 2 and len(temp_word) > 0:
                                temp_word[-1] = temp_word[-1] + tmp[2:]
                    else:
                        temp_1.pop()
                        temp_2.pop()
                        temp_word.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                words.append(temp_word)

            self.load_weights(names, weights)
            if item_id % 50 == 0:
                logger.info('  To sentence {}/{}. Time: {}sec'.format(item_id, corpus.n_total, time.time() - t_tmp))

        tmp_fn = '{}/{}-{}_pred.txt'.format(result_dir, lang, mode)
        score_fn = '{}/{}-{}_score.txt'.format(result_dir, lang, mode)
        self.write_result(words, y_true, y_pred, tmp_fn)
        os.system('python %s < %s > %s' % ('conlleval.py', tmp_fn, score_fn))

        F1 = -1
        with open(score_fn, 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 1:
                    F1 = float(line.strip().split()[-1])
                logger.info(line.strip())

        return F1

    def evaluate_NOmeta(self, corpus, result_dir, logger, lang='en', mode='valid'):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=False)
        data_batches = corpus.get_batches(batch_size=64)
        self.model.eval()

        y_true = []
        y_pred = []
        words = []
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        for batch_id, data_batch in enumerate(data_batches):

            with torch.no_grad():
                logits = self.model(data_batch['input_ids'], data_batch['segment_ids'],
                               data_batch['input_mask'])  # batch_size x seq_len x target_size

            logits = torch.argmax(torch.nn.functional.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            input_ids = data_batch['input_ids'].to('cpu').numpy()
            label_ids = data_batch['label_ids'].to('cpu').numpy()
            input_mask = data_batch['input_mask'].to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                temp_word = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X":
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[max(logits[i][j], 1)])
                            temp_word.append(tokenizer.ids_to_tokens[input_ids[i][j]])
                        else:
                            tmp = tokenizer.ids_to_tokens[input_ids[i][j]]
                            if len(tmp) > 2 and len(temp_word) > 0:
                                temp_word[-1] = temp_word[-1] + tmp[2:]
                    else:
                        temp_1.pop() # pop [SEP]
                        temp_2.pop()
                        temp_word.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                words.append(temp_word)

        tmp_fn = '{}/{}-{}_pred.txt'.format(result_dir, lang, mode)
        score_fn = '{}/{}-{}_score.txt'.format(result_dir, lang, mode)
        self.write_result(words, y_true, y_pred, tmp_fn)
        os.system('python %s < %s > %s' % ('conlleval.py', tmp_fn, score_fn))

        F1 = -1
        with open(score_fn, 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 1:
                    F1 = float(line.strip().split()[-1])
                logger.info(line.strip())

        return F1

    def save_model(self, result_dir, fn_prefix, max_seq_len):
        # Save a trained model and the associated configuration
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        output_model_file = os.path.join(result_dir, '{}_{}'.format(fn_prefix, WEIGHTS_NAME))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(result_dir, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        model_config = {"bert_model": self.bert_model, "do_lower": False,
                        "max_seq_length": max_seq_len, "num_labels": len(self.label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(result_dir, "model_config.json"), "w", encoding='utf-8'))
