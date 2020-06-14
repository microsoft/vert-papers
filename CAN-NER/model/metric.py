# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-02-16 09:53:19
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:32:53

## input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def readSentence(input_file):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences, labels


def readTwoLabelSentence(input_file, truth_col=-1):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[truth_col])
            predict_label.append(pair[0])

    return sentences, golden_labels, predict_labels


def weibo_readTwoLabelSentence(input_file, truth_col=-1, type=None):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            if type != None:
                if type in pair[0]:
                    sentence.append('O')
                    predict_label.append('O')
                else:
                    sentence.append(pair[0])
                    predict_label.append(pair[0])
                if type in pair[truth_col]:
                    golden_label.append('O')
                else:
                    golden_label.append(pair[truth_col])
            else:
                sentence.append(pair[0])
                golden_label.append(pair[truth_col])
                predict_label.append(pair[0])

    return sentences, golden_labels, predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file:", golden_file, predict_file)
    print("Label format:", label_type)
    golden_sent, golden_labels = readSentence(golden_file)
    predict_sent, predict_labels = readSentence(predict_file)
    acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print("Acc:%s, P:%s R:%s, F:%s" % (acc, P, R, F))


def fmeasure_from_singlefile(twolabel_file, label_type="BMES", truth_col=-1, base_path=None, is_test=False, type="test",
                             is_weibo=False):
    if is_weibo:
        sent, golden_labels, predict_labels = weibo_readTwoLabelSentence(twolabel_file, truth_col, "NOM")
        acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
        print_helper(base_path, is_test, type, acc, P, R, F, suffix=".NOM")
        print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc))

        sent, golden_labels, predict_labels = weibo_readTwoLabelSentence(twolabel_file, truth_col, "NAM")
        acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
        print_helper(base_path, is_test, type, acc, P, R, F, suffix=".NAM")
        print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc))

        sent, golden_labels, predict_labels = weibo_readTwoLabelSentence(twolabel_file, truth_col)
        acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
        print_helper(base_path, is_test, type, acc, P, R, F, suffix=".all")
        print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc))

        return F
    else:
        sent, golden_labels, predict_labels = readTwoLabelSentence(twolabel_file, truth_col)

        acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
        print_helper(base_path, is_test, type, acc, P, R, F)
        print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc))

        return F


def print_helper(base_path, is_test, type, acc, P, R, F, suffix=""):
    if is_test:
        if type == "test":
            print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc),
                  file=open(base_path + "/test_eval_in_training" + suffix, 'a'))
        elif type == "dev":
            print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc),
                  file=open(base_path + "/dev_eval_in_training" + suffix, 'a'))
        elif type == "train":
            print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc),
                  file=open(base_path + "/train_eval_in_training" + suffix, 'a'))
    else:
        print("P:%s, R:%s, F:%s, ACC:%s" % (P, R, F, acc), file=open(base_path + "/eval_results" + suffix, 'w+'))

def fmeasure_from_singlefile_lengths(file, lengths, label_type="BMES", truth_col=-1):
    sent, golden_labels, predict_labels = readTwoLabelSentence(file, truth_col)
    for length in lengths:
        # print("Length:%s" % length)
        P, R, F = get_ner_fmeasure_length(golden_labels, predict_labels, length, label_type)
    # print_helper(base_path, is_test, type, acc, P, R, F)
        print("P:%s, R:%s, F:%s" % (P, R, F))


def get_ner_fmeasure_length(golden_lists, predict_lists, length, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        # for idy in range(len(golden_list)):
        #     if golden_list[idy] == predict_list[idy]:
        #         right_tag += 1
        # all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        gold_matrix = [x for x in gold_matrix if filter_length(x,length)]
        pred_matrix = [x for x in pred_matrix if filter_length(x,length)]

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    # accuracy = (right_tag + 0.0) / all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("Length:", length, "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    # return accuracy, precision, recall, f_measure
    return precision, recall, f_measure

def filter_length(matrix,length):
    if isinstance(length,tuple):
        range_s = length[0]
        range_e = length[1]
        start = matrix.index("[")
        if "," not in matrix:
            pred_len = 1
        else:
            middle = matrix.index(",")
            end = matrix.index("]")
            pred_len = int(matrix[middle + 1:end]) - int(matrix[start + 1:middle]) + 1
        if pred_len>=range_s and pred_len<=range_e:
            return True
        else:
            return False
    else:
        start = matrix.index("[")
        if "," not in matrix:
            if length==1:
                return True
        else:
            middle = matrix.index(",")
            end = matrix.index("]")
            if length != int(matrix[middle + 1:end]) - int(matrix[start + 1:middle]) + 1:
                return False
            else:
                return True


