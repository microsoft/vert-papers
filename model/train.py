import argparse
import json
import os
import time
import util
import torch.optim as optim
import torch.nn as nn
import nner_with_seg_info as nner
from data_process import *
from metric import fmeasure_from_singlefile
from torch.utils import data
from util import MyDataset, print_para

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, tx, ty, id2tag, name='', base_path=None, is_test=False, type="test", is_weibo=False):
    model.eval()
    assert len(tx) == len(ty)
    with torch.no_grad():
        py = []
        for sentence in tx:
            # sentence = torch.tensor(sentence, dtype=torch.long, requires_grad=False).cuda()
            sentence = torch.tensor(sentence, dtype=torch.long, requires_grad=False).to(device)
            sentence = sentence.unsqueeze(0)
            mask = torch.ones((1, sentence.size(1))).byte().cuda()
            feats = model(sentence, mask)
            if isinstance(model, torch.nn.parallel.DataParallel):
                ret = model.module.crf.forward(feats, mask)
            else:
                ret = model.crf.forward(feats, mask)

            tags = [tag.item() for tag in ret[1][0]]
            py.append(tags)

        assert len(py) == len(ty)
        n = len(py)
        evalname = name + '.' + type + '.eval'
        fout = open(base_path + "/" + evalname, 'w', encoding='utf8')
        for i in range(n):
            pi = py[i]
            ti = ty[i]
            assert len(pi) == len(ti)
            k = len(pi)
            for j in range(k):
                idx = pi[j]
                if idx not in id2tag:
                    idx = 0
                fout.write(id2tag[idx] + ' ' + id2tag[ti[j]] + '\n')
            fout.write('\n')

        fout.close()
        print('eval ' + evalname)
        F = fmeasure_from_singlefile(base_path + "/" + evalname, "BMES", base_path=base_path, is_test=is_test, type=type, is_weibo=is_weibo)
        # 中间结果
        if is_test:
            os.remove(base_path + "/" + evalname)
        return F


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAN-NER Model')
    parser.add_argument('--model_name', type=str, default='weibo_model', help='give the model a name.')
    parser.add_argument('--data_name', type=str, default='weibo', choices=['weibo', 'MSRA', 'onto4'], help='name for dataset.')

    parser.add_argument('--train_data_path', type=str, default='', help='file path for train set.')
    parser.add_argument('--test_data_path', type=str, default='', help='file path for test set.')
    parser.add_argument('--dev_data_path', type=str, default=None, help='file path for dev set.')
    parser.add_argument('--pretrained_embed_path', type=str, default='../data/embeds', help='path for embedding.')
    parser.add_argument('--result_folder', type=str, default='', help='folder path for save models and results.')

    parser.add_argument('--seed', type=int, default=None, help='seed for everything')

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden dimension')
    parser.add_argument('--window_size', type=int, default=5, help='window size for acnn')

    parser.add_argument("--is_parallel", default=False, action='store_true', help="whether to use multiple gpu")

    args = parser.parse_args()

    if args.seed is not None:
        util.reset_seeds(args.seed)

    trainfile = args.train_data_path
    data_name = args.data_name
    model_name = args.model_name
    testfile = args.test_data_path
    result_folder = args.result_folder

    batch_size = args.batch_size
    epoch_num = args.epoch
    is_parallel = args.is_parallel

    base_path = os.path.join(result_folder, model_name + time.strftime("%Y_%m_%d_%I_%M_%S"), )
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    json.dump(args.__dict__, open(base_path + "/" + model_name + '.config', 'w', encoding='utf8'), indent=1)
    batch_log = open(base_path + '/batch.%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
    epoch_log = open(base_path + '/epoch.%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
    print(args)

    word2id, id2word, embeds_data = load_embed(args.pretrained_embed_path)

    train_data, tag2id = load_trainset_BIO(trainfile, word2id, data_name)
    trainX = [data[0] for data in train_data]
    trainY = [data[1] for data in train_data]

    testdata = load_testset_BIO(testfile, word2id, data_name)
    testX = [data[0] for data in testdata]
    testY = [data[1] for data in testdata]

    if args.dev_data_path is not None:
        dev_data = load_testset_BIO(args.dev_data_path, word2id, data_name)
        devX = [data[0] for data in dev_data]
        devY = [data[1] for data in dev_data]

    id2tag = {val: key for (key, val) in tag2id.items()}

    train_dataset = MyDataset(trainX, trainY)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=util.collate_fn)

    model = nner.CANNERModel(args, tag2id, args.dropout, pretrain_embed=embeds_data)
    # print_para(model)
    if is_parallel:
        model = nn.DataParallel(model, [0, 1])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    n = len(trainX)
    best_dev = -1
    best_test = -1
    best_epoch = -1
    for epoch in range(epoch_num):
        batch_cnt = 0
        epoch_loss = 0.0
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        for idx, (batch_x, batch_y, mask) in enumerate(train_loader):
            batch_cnt += 1
            optimizer.zero_grad()

            n1 = time.time()
            bfeats = model(batch_x, mask)
            if isinstance(model, torch.nn.parallel.DataParallel):
                loss = model.module.crf.batch_loss(bfeats, mask, batch_y)
            else:
                loss = model.crf.batch_loss(bfeats, mask, batch_y)
            epoch_loss += loss.item()
            n2 = time.time()
            print(epoch + 1, idx, n, loss.item(), n2 - n1)
            print(epoch + 1, idx, n, loss.item(), n2 - n1, file=batch_log)

            loss.backward()
            optimizer.step()

            del batch_x
            del batch_y
            del loss

        epoch_loss = epoch_loss / batch_cnt
        print(epoch + 1, epoch_loss, file=epoch_log)
        print(epoch + 1, epoch_loss)

        if args.dev_data_path is not None:
            if "weibo" in data_name:
                test_F = evaluate(model, testX, testY, id2tag, model_name, base_path, is_test=True, type="test", is_weibo=True)
                dev_F = evaluate(model, devX, devY, id2tag, model_name, base_path, is_test=True, type="dev", is_weibo=True)
            else:
                test_F = evaluate(model, testX, testY, id2tag, model_name, base_path, is_test=True, type="test")
                dev_F = evaluate(model, devX, devY, id2tag, model_name, base_path, is_test=True, type="dev")

            if dev_F > best_dev:
                best_dev = dev_F
                best_test = test_F
                best_epoch = epoch + 1
                model_name = base_path + "/" + model_name + '.' + str(epoch + 1)
                torch.save(model.state_dict(), model_name)

            print("epoch:%-s, dev:%-s, test:%-s, best epoch:%-s, best dev:%-s, best test:%-s" % (
                epoch + 1, dev_F, test_F, best_epoch, best_dev, best_test), file=open(base_path + "/F1.log", 'a'))
        else:
            evaluate(model, testX, testY, id2tag, model_name, base_path, is_test=True, type="test")


    torch.save(model.state_dict(), base_path + "/" + model_name)
