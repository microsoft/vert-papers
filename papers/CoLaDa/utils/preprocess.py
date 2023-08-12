# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os

def read_examples_from_file(file_name):
    guid_index = 0
    sentences = []
    labels = []
    with open(file_name, encoding="utf-8") as f:
        cur_words = []
        cur_labels = []
        for line in f:
            line = line.strip().replace("\t", " ")
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if cur_words:
                    sentences.append(cur_words)
                    labels.append(cur_labels)
                    guid_index += 1
                    cur_words = []
                    cur_labels = []
            else:
                splits = line.split(" ")
                cur_words.append(splits[0])
                cur_labels.append(splits[-1].replace("\n", ""))
        if cur_words:
            sentences.append(cur_words)
            labels.append(cur_labels)
            guid_index += 1
    print("number of sentences : {}".format(len(sentences)))
    return sentences, labels

def convert_to_conll(sents, lbls, to_file):
    lines = []
    for words, tags in zip(sents, lbls):
        for w, t in zip(words, tags):
            lines.append("{}\t{}\n".format(w,t))
        lines.append("\n")
    with open(to_file, mode="w", encoding="utf-8") as fp:
        fp.writelines(lines)
    return



if __name__ == "__main__":
    dataset = "wikiann"
    transys = "m2m100"
    lang_list = ["ar", "hi", "zh"]
    # deduplicate the train data of english and translation for KNN
    ori_sents, ori_labels = read_examples_from_file(f"./data/{dataset}-lingual/en/train.txt")
    new_sents = []
    new_labels = []
    dup_ids = []
    for k in range(len(ori_sents)):
        if ori_sents[k] in new_sents:
            dup_ids.append(k)
            continue
        new_sents.append(ori_sents[k])
        new_labels.append(ori_labels[k])
    convert_to_conll(new_sents, new_labels, f"./data/{dataset}-lingual/en/dup-train.txt")
    print(len(dup_ids))

    for lang in lang_list:
        sents, labels = read_examples_from_file(f"./data/{dataset}-lingual/en/{transys}-trans/trans-train.{lang}.conll")
        assert len(sents) == len(ori_sents)
        new_sents = []
        new_labels = []
        for k in range(len(sents)):
            if k in dup_ids:
                continue
            new_sents.append(sents[k])
            new_labels.append(labels[k])
        convert_to_conll(new_sents, new_labels, f"./data/{dataset}-lingual/en/{transys}-trans/dup-trans-train.{lang}.conll")

    # deduplicate the unlabeled data of target langauge for KNN
    for lang in lang_list:
        sents, labels = read_examples_from_file(f"./data/{dataset}-lingual/{lang}/train.txt")
        new_sents = []
        new_labels = []
        dup_ids = []
        for k in range(len(sents)):
            if sents[k] in new_sents:
                dup_ids.append(k)
                continue
            new_sents.append(sents[k])
            new_labels.append(labels[k])
        convert_to_conll(new_sents, new_labels, f"./data/{dataset}-lingual/{lang}/dup-train.txt")
        print(len(dup_ids))