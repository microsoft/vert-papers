import json
import random
import torch
import torch.nn.init as init
from torch.utils import data
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_para(model):
    for name, module in model.named_children():
        print(name)
        for name, param in module.named_parameters():
            print(name, param.requires_grad)

def reset_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Token:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.length = end - start
        self.text = text 

def print_tensor(_tensor,path):
    shape = _tensor.size()
    # print("[", end="", file=open(path, "a"))
    # if len(shape)==3:
    #     for a in _tensor:
    #         for b in a:
    #             for c in b:
    #                 print(c.item(),",",end="",file=open(path,"a"))
    # elif len(shape)==2:
    #     for a in _tensor:
    #         for b in a:
    #             print(b.item(),",",end="",file=open(path,"a"))
    # print("]", file=open(path, "a"))
    with open(path, "a") as mf:
        mf.write("[")
        if len(shape) == 3:
            mf.write(", ".join([str(c.item()) for a in _tensor for b in a for c in b]))
        elif len(shape) == 2:
            mf.write(", ".join([str(b.item()) for a in _tensor for b in a]))
        elif len(shape) == 1:
            mf.write(", ".join([str(a.item()) for a in _tensor]))
        mf.write("]\n")

def init_weight_(weight):
    init.kaiming_uniform_(weight)

def init_embedding_(input_embedding):
    """
    Initialize embedding
    """
    init_weight_(input_embedding.weight)
    # init.normal_(input_embedding.weight, 0, 0.1)

def init_linear_(input_linear):
    """
    Initialize linear transformation
    """
    # init.normal_(input_linear.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    # # init_weight_(input_linear.weight)
    # if input_linear.bias is not None:
    #     # input_linear.bias.data.zero_()
    #     init.constant_(input_linear.bias, 0)
    # # weight_norm(input_linear)

    init_weight_(input_linear.weight)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm_(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        init_weight_(weight)

        weight = eval('input_lstm.weight_hh_l' + str(ind))
        init_weight_(weight)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            init_weight_(weight)

            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            init_weight_(weight)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_cnn_(input_cnn):
    """
    Initialize cnn
    """
    init_weight_(input_cnn.weight)
    if input_cnn.bias is not None:
        input_cnn.bias.data.zero_()
    # std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    # init.normal_(input_cnn.weight, mean=0, std=std)
    # init.constant_(input_cnn.bias, 0)
    # weight_norm(input_cnn, dim=2)
    # init_weight_(input_cnn.weight)
    # if input_cnn.bias is not None:
    #     input_cnn.bias.data.zero_()
    #
    # weight_norm(input_cnn)

class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        return X, Y

def collate_fn(seq_list):
    batch_x, batch_y = zip(*seq_list)
    # (batch, tokens, features)
    batch_lens = [len(item) for item in batch_x]
    max_tokens = max(batch_lens)
    batch_n = len(batch_lens)
    feature_num = len(batch_x[0][0])

    # mask = torch.ones((batch_n, max_tokens)).byte()
    mask = torch.ones((batch_n, max_tokens)).bool()
    batches = []
    batch_tags = []
    for i, d in enumerate(batch_x):
        dt = batch_x[i][:]
        l = len(dt)
        tags = batch_y[i][:]
        for j in range(max_tokens - l):
            dt.append([])
            mask[i, j + l] = 0
            tags.append(0)
            for k in range(feature_num):
                dt[j + l].append(0)
        batches.append(dt)
        batch_tags.append(tags)

    batch_data = torch.tensor(batches, dtype=torch.long)
    batch_tags = torch.tensor(batch_tags, dtype=torch.long)

    batch_data.to(device)
    batch_tags.to(device)
    mask.to(device)
    return batch_data, batch_tags, mask



