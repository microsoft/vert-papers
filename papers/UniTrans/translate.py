import numpy as np
from numpy import linalg as LA
import argparse, os
import torch


def nn(tgt_vec, eng_vec, O, bs):
    esp = torch.from_numpy(tgt_vec.astype('float32')).cuda()
    eng = torch.from_numpy(eng_vec.astype('float32')).cuda()
    O = torch.from_numpy(O.astype('float32')).cuda()

    indexes = []

    for i in range(0, len(eng_vec), bs):
        cossim = esp.mm(O).mm(eng[i:i + bs].t()).t()
        indexes.append(torch.max(cossim, 1)[1].cpu())

    return torch.cat(indexes).numpy()

def get_nn_avg_dist(esp, eng, O, k, bs):
    distance_esp = []
    distance_eng = []

    for i in range(0, esp.size(0), bs):
        cossim_esp = esp[i:i + bs].mm(O).mm(eng.t())
        tgt_dist, _ = torch.topk(cossim_esp, k)
        distance_esp.append(tgt_dist.mean(1))

    for i in range(0, eng.size(0), bs):
        cossim_eng = esp.mm(O).mm(eng[i:i + bs].t()).t()
        eng_dist, _ = torch.topk(cossim_eng, k)
        distance_eng.append(eng_dist.mean(1))

    return torch.cat(distance_esp), torch.cat(distance_eng)

def csls(tgt_vec, eng_vec, O, k, bs):
    esp = torch.from_numpy(tgt_vec.astype('float32')).cuda()
    eng = torch.from_numpy(eng_vec.astype('float32')).cuda()
    O = torch.from_numpy(O.astype('float32')).cuda()

    tgt_distance, eng_distance = get_nn_avg_dist(esp, eng, O, k, bs)

    all_scores = []
    indexes = []

    for i in range(0, len(eng_vec), bs):
        cossim = esp.mm(O).mm(eng[i:i + bs].t()).t()
        scores = cossim * 2 - eng_distance[i:i + bs].unsqueeze(1) - tgt_distance.unsqueeze(0)
        indexes.append(torch.max(scores, 1)[1].cpu())

    return torch.cat(indexes).numpy()

def load_embedding(path, vocab_size):
    word_vector = []
    word_dict = {}
    words = []

    num = 0

    for line in open(path, encoding='utf-8'):
        if num == 0:
            num += 1
            continue
        if num < vocab_size:
            word, vec = line.rstrip().split(' ', 1)
            word_dict[word] = len(word_dict)
            words.append(word)
            vec = np.array(vec.split(), dtype='float32')
            word_vector.append(vec)

            num += 1
            if num % 10000 ==0:
                print('To word #%d' % num)

    print(len(word_vector))

    return word_dict, normalize(np.vstack(word_vector)), words

def normalize(vectors):
    return vectors / LA.norm(vectors, axis=1).reshape((vectors.shape[0], 1))

def generate_dict(args):
    dict_path = os.path.join(args.data_dir, 'en2{}/muse/dict.txt'.format(args.tgt_lang))
    mapping_path = os.path.join(args.data_dir, 'en2{}/muse/best_mapping.pth'.format(args.tgt_lang))
    tgt_embed_path = os.path.join(args.embed_dir, "wiki.{}.vec".format(args.tgt_lang))
    src_embed_path = os.path.join(args.embed_dir, "wiki.en.vec")

    print('loading eng embedding...')
    eng_word, eng_vec, engs = load_embedding(src_embed_path, args.vocab_size)

    print('loading tgt embedding...')
    tgt_word, tgt_vec, esps = load_embedding(tgt_embed_path, args.vocab_size)

    O = torch.load(mapping_path)

    if args.distance == 'csls':
        indexes = csls(tgt_vec, eng_vec, O, args.k, args.batch_size)
    elif args.distance == 'nn':
        indexes = nn(tgt_vec, eng_vec, O, args.batch_size)

    output = open(dict_path, 'w', encoding='utf-8')

    for i in range(len(engs)):
        output.write(engs[i] + ' ' + esps[indexes[i]] + '\n')

    output.close()

    return dict_path



def get_lower_vocab(fn):
    vocab = {}
    with open(fn, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 0:
                continue

            word = line.split()[0]
            if word.lower() not in vocab:
                vocab[word.lower()] = np.zeros(3) # {word: [n_lower_case, n_capital, n_upper_case]}
            if word == word.lower():
                vocab[word.lower()][0] += 1
            elif word == word.capitalize():
                vocab[word.lower()][1] += 1
            else:
                vocab[word.lower()][2] += 1

    for k, v in vocab.items():
        vocab[k] = v / sum(v)

    print('The vocab size is %d' % len(vocab))
    return vocab

def adjust_case_with_ner_vocab(word, tgt_ner_vocab_lower):
    p_case = tgt_ner_vocab_lower[word]
    id_case = np.random.choice(3, p=p_case)

    if id_case == 1:  # 0: n_lower_case, 1: n_capital, 2: n_upper_case
        word = word.capitalize()
    elif id_case == 2:
        word = word.upper()
    else:
        word = word

    return word

def translate(dict_en_tgt_path, output_fn, args, tgt_ner_vocab_lower=None):
    # training_data = 'data/ner/conll/en/train.txt' # source training data file
    training_data = f"{args.data_dir}/en/train.txt"

    word_dict = {}
    for line in open(dict_en_tgt_path, encoding='utf-8'):
        src, tgt = line.rstrip('\n').split(' ', 1)
        word_dict[src] = tgt.lower()

    with open(output_fn, 'w', encoding='utf-8') as fw:

        for idx, line in enumerate(open(training_data, 'r', encoding='utf-8')):
            if len(line.strip()) == 0:
                fw.write(line)
            else:
                word, label = line.strip().split() # [word, label]

                if label[2:] == 'PER' or word.lower() not in word_dict:
                    fw.write(word + ' ' + label + '\n')

                else:
                    temp = word_dict[word.lower()]
                    if tgt_ner_vocab_lower is not None and temp in tgt_ner_vocab_lower:
                        word = adjust_case_with_ner_vocab(temp, tgt_ner_vocab_lower)

                    else:
                        if word.isupper():
                            word = temp.upper()
                        elif word[0].isupper():
                            word = temp[0].upper() + temp[1:]
                        else:
                            word = temp

                    fw.write(word + ' ' + label + '\n')
            if (idx + 1) % 10000 ==0:
                print('Translate tokens #%d' % (idx + 1))



def main(args):
    dict_path = generate_dict(args)  # according to loaded embeddings
    fn_ner_translated = os.path.join(args.data_dir, "en2{}/train.txt".format(args.tgt_lang))

    if args.tgt_lang == 'de': # adjust capitalization
        tgt_ner_vocab_lower = get_lower_vocab(os.path.join(args.data_dir, args.tgt_lang, "train.txt"))
    else:
        tgt_ner_vocab_lower = None

    translate(dict_path, fn_ner_translated, args, tgt_ner_vocab_lower=tgt_ner_vocab_lower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load muse')

    parser.add_argument('--tgt_lang', type=str, default='ny')
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--data_dir', type=str, default='data/XTREME')
    parser.add_argument('--embed_dir', type=str, default='../data/word_embedding/')

    # for generating dictionary
    parser.add_argument('--k', type=int, default=10, help='k in csls')
    parser.add_argument('--batch_size', type=int, default=5000, help='how many words to translate at once')
    parser.add_argument('--distance', type=str, default='csls', help='distance type, nn or csls')
    parser.add_argument('--vocab_size', type=int, default=100000, help='size of the vocab to load embeddings')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    main(args)