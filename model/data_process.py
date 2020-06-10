import numpy as np
import torch


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def load_trainset_BIO(name, word2id, corpusname='onto4'):
    with open(name, 'r', encoding='utf8') as f:
        datas = f.read().rstrip()
    sentences = datas.split('\n\n')

    rs = []
    sent = []
    tag2id = getTagIx(corpusname)[0]

    for r in sentences:
        lines = r.split('\n')
        for index, line in enumerate(lines):
            text = line.strip().split(" ")
            word = text[0][0]
            seg = text[0][1]
            tag = text[1]
            word = normalize_word(word)
            if word not in word2id:
                word2id[word] = 1
            if tag not in tag2id:
                print(tag)
                tag2id[tag] = len(tag2id)

            if index == len(lines) - 1:
                if seg == "0":
                    seg = 4
                else:
                    seg = 3
                sent.append([word2id[word], tag2id[tag], seg])
                ret = getDataFromSent_with_seg_test(sent)
                rs.append(ret)
                sent = []

            else:
                next_seg = lines[index + 1].split(" ")[0][1]
                if seg == "0":
                    if next_seg == "0":
                        seg = 4
                    else:
                        seg = 1
                else:
                    if next_seg == "0":
                        seg = 3
                    else:
                        seg = 2
                sent.append([word2id[word], tag2id[tag], seg])

    return rs, tag2id


def load_embed(filename):
    word2id, id2word = {}, {}

    embs = []
    file_names = [
                  "./data/embeds",
                  "./data/embedding/embeds",
                  "./data/embedding/word2vec.sgns.weibo.onlychar",
                  "./data/embedding/fasttext.cc.zh.300.vec.onlychar",
                  "./data/embedding/glove.6B.100d.english.txt",
                  "./data/embedding/glove.6B.300d.english.txt"]

    if filename in file_names:
        print("ADD UNK and PAD.....")
        word2id = {'<PAD>': 0, '<UNK>': 1}
        id2word = {0: '<PAD>', 1: '<UNK>'}
        # UNK not in vocab, we need to random a unk embedding
        # range(-1,1)
        if "glove.6B.100d.english.txt" in filename:
            # rand = (2 * np.random.random(100) - 1).astype(np.float32)
            scale = np.sqrt(6.0 / 100)
            UNK = np.random.uniform(-scale, scale, 100)
            PAD = np.random.uniform(-scale, scale, 100)
        else:
            scale = np.sqrt(6.0 / 300)
            UNK = np.random.uniform(-scale, scale, 300)
            PAD = np.random.uniform(-scale, scale, 300)

        embs.append(PAD)
        embs.append(UNK)
    else:
        print("ADD PAD.....")
        word2id = {'<PAD>': 0}
        id2word = {0: '<PAD>'}
        if filename == "./data/embedding/ctb.50d.vec":
            scale = np.sqrt(6.0 / 50)
            PAD = np.random.uniform(-scale, scale, 50)
        elif filename == "./data/embedding/word2vec.continuous.skipgram":
            scale = np.sqrt(6.0 / 50)
            PAD = np.random.uniform(-scale, scale, 100)
        else:
            scale = np.sqrt(6.0 / 300)
            PAD = np.random.uniform(-scale, scale, 300)
        embs.append(PAD)

    V, D = -1, -1
    for idx, line in enumerate(open(filename, encoding='utf8')):
        if idx != 0 and idx % 10000 == 0:
            print('{} embs loaded'.format(idx))

        line_split = line.strip().split()
        if idx == 0 and len(line_split) == 2:
            V, D = int(line_split[0]), int(line_split[1])
            continue
        if D == -1:
            D = len(line_split) - 1
        if len(line_split) != D + 1:
            continue

        if line_split[0] in word2id.keys():
            print('line {}: {} already in vocab'.format(idx + 1, line_split[0]))
            continue
        word2id[line_split[0]] = len(word2id)
        id2word[len(id2word)] = line_split[0]
        embs.append(np.array(list(map(float, line_split[1:])), dtype=np.float32))

    return word2id, id2word, np.array(embs, dtype=np.float32)


def load_testset_BIO(name, word2id, corpusname='onto4'):
    with open(name, 'r', encoding='utf8') as f:
        datas = f.read().rstrip()
    sentences = datas.split('\n\n')

    rs = []
    sent = []
    tag2id = getTagIx(corpusname)[0]

    for r in sentences:
        lines = r.split('\n')
        for index, line in enumerate(lines):
            text = line.strip().split(" ")
            word = text[0][0]
            seg = text[0][1]
            tag = text[1]
            word = normalize_word(word)
            if word not in word2id:
                word2id[word] = 1
            if tag not in tag2id:
                print(tag)
                tag2id[tag] = len(tag2id)

            if index == len(lines) - 1:
                if seg == "0":
                    seg = 4
                else:
                    seg = 3
                sent.append([word2id[word], tag2id[tag], seg])
                ret = getDataFromSent_with_seg_test(sent)
                rs.append(ret)
                sent = []

            else:
                next_seg = lines[index + 1].split(" ")[0][1]
                if seg == "0":
                    if next_seg == "0":
                        seg = 4
                    else:
                        seg = 1
                else:
                    if next_seg == "0":
                        seg = 3
                    else:
                        seg = 2
                sent.append([word2id[word], tag2id[tag], seg])

    return rs


def getDataFromSent_with_seg_test(sent):
    words = []
    tags = []
    # segs = []
    for i in sent:
        # segs.append([i[2]])
        words.append([i[0], i[2]])
        tags.append(i[1])
    # return words, tags, segs
    return words, tags


def getTagIx(name, ):
    tag_to_ix = {}

    def getTagDict(tags):
        if 'O' not in tags:
            tags.insert(0, "O")
        d = {}
        for t in tags:
            d[t] = len(d)
        return d

    if name == 'conll2003':
        tags = ["O", "B-PER", "M-PER", "S-PER", "E-PER",
                "B-ORG", "M-ORG", "S-ORG", "E-ORG",
                "B-LOC", "M-LOC", "S-LOC", "E-LOC",
                "B-MISC", "M-MISC", "S-MISC", "E-MISC", ]
        tag_to_ix = getTagDict(tags)
    elif name == 'onto4':
        tags = [
            "O", "B-PER", "M-PER", "S-PER", "E-PER",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
            "B-GPE", "M-GPE", "S-GPE", "E-GPE",
        ]
        tag_to_ix = getTagDict(tags)
    elif name == 'resume':
        tags = [
            "O", "B-CONT", "M-CONT", "S-CONT", "E-CONT",
            "B-EDU", "M-EDU", "S-EDU", "E-EDU",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
            "B-NAME", "M-NAME", "S-NAME", "E-NAME",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-PRO", "M-PRO", "S-PRO", "E-PRO",
            "B-RACE", "M-RACE", "S-RACE", "E-RACE",
            "B-TITLE", "M-TITLE", "S-TITLE", "E-TITLE",
        ]
        tag_to_ix = getTagDict(tags)
    elif name == 'weibo.nom':
        tags = [
            "B-PER", "M-PER", "S-PER", "E-PER",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
            "B-GPE", "M-GPE", "S-GPE", "E-GPE"
        ]
        wtags = [w + '.NOM' for w in tags]

        tag_to_ix = getTagDict(wtags)
    elif name == 'weibo.nam':
        tags = [
            "B-PER", "M-PER", "S-PER", "E-PER",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
            "B-GPE", "M-GPE", "S-GPE", "E-GPE"
        ]
        wtags = [w + '.NAM' for w in tags]

        tag_to_ix = getTagDict(wtags)
    elif name == 'weibo.all':
        tags = [
            "B-PER", "M-PER", "S-PER", "E-PER",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
            "B-GPE", "M-GPE", "S-GPE", "E-GPE"
        ]
        wtags = [w + '.NOM' for w in tags]
        wtags += [w + '.NAM' for w in tags]

        tag_to_ix = getTagDict(wtags)
    elif name == 'MSRA':
        tags = [
            "O", "B-PER", "M-PER", "S-PER", "E-PER",
            "B-ORG", "M-ORG", "S-ORG", "E-ORG",
            "B-LOC", "M-LOC", "S-LOC", "E-LOC",
        ]
        tag_to_ix = getTagDict(tags)

    ix_to_tag = {val: key for (key, val) in tag_to_ix.items()}
    return tag_to_ix, ix_to_tag




