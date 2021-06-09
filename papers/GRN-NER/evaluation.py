import codecs
import os
import math
import time

from constants import Constants
from data_format_util import iobes_to_iob2
from data_processing import generate_mini_batch_input


def get_perf_metric(name, best_f1):
    """
    checked
    Evalute the result file and get the new f1 value
    :param name: name of the model
    :param best_f1: the current best f1 value
    :return: new best f1 value, the new f1 value, whether the new best f1 value is updated
    """
    should_save = False
    new_f1 = 0.0

    eval_path = Constants.Eval_Folder
    eval_tmp_folder = Constants.Eval_Temp_Folder
    eval_script = Constants.Eval_Script

    prediction_file = eval_tmp_folder + '/predition.' + name
    score_file = eval_tmp_folder + '/score.' + name

    os.system('perl %s <%s >%s' % (eval_script, prediction_file, score_file))

    evaluation_lines = [line.rstrip() for line in codecs.open(score_file, 'r', 'utf8')]

    for i, line in enumerate(evaluation_lines):
        if i == 1:
            new_f1 = float(line.strip().split()[-1])
            if new_f1 > best_f1:
                best_f1 = new_f1
                should_save = True

    return best_f1, new_f1, should_save


def evaluating(model, datas, best_F, name, mappings, char_mode, use_gpu, device, mini_batch_size):
    """
    checked
    Evaluate the F1 score on a given dataset
    :param model: To-be-tested model
    :param datas: Data set loader
    :param best_F: Best F1 score already obtained
    :param name: Name of the model
    :param mappings: Dict, a mapping dictionary containing tag_to_id, id_to_tag, word_to_id, id_to_word, char_to_id, id_to_char
    :param char_mode: CharEmbeddingSchema, char embedding type, in [LSTM, CNN]
    :param use_gpu: Use gpu or not
    :param device: Device to run the function
    :param mini_batch_size: Mini-batch size for evaluation
    :return:
    """
    prediction = []

    eval_tmp_folder = Constants.Eval_Temp_Folder
    id_to_tag = mappings["id_to_tag"]
    char_to_id = mappings["char_to_id"]

    train_indecies = list(range(len(datas)))
    batch_count = math.ceil(len(datas) / mini_batch_size)
    time_cost = 0

    for batch_i in range(batch_count):
        start_idx = batch_i * mini_batch_size
        end_idx = min((batch_i + 1) * mini_batch_size, len(datas))

        mini_batch_idx = train_indecies[start_idx:end_idx]
        sentence_masks, words, chars, tags, \
        sentence_char_lengths, sentence_char_position_map, str_words, unaligned_tags = \
            generate_mini_batch_input(datas, mini_batch_idx, mappings, char_mode)

        if use_gpu:
            sentence_masks = sentence_masks.to(device)
            words = words.to(device)
            chars = chars.to(device)
            tags = tags.to(device)
            sentence_char_lengths = sentence_char_lengths.to(device)

        eval_start_time = time.time()
        scores, tag_id_seqs = model(words, sentence_masks, chars, sentence_char_lengths, sentence_char_position_map, device)
        eval_time_cost = time.time() - eval_start_time
        time_cost += eval_time_cost

        predicted_tags = [[id_to_tag[id] for id in predicted_id] for predicted_id in tag_id_seqs]
        predicted_tags_bio = [iobes_to_iob2(predicted_tags_sentence) for predicted_tags_sentence in predicted_tags]

        ground_truth_tags = [[id_to_tag[id] for id in ground_truth_id] for ground_truth_id in unaligned_tags]
        ground_truth_tags_bio = [iobes_to_iob2(ground_truth_tags_sentence) for ground_truth_tags_sentence in ground_truth_tags]

        for si in range(end_idx - start_idx):
            for (str_word, true_tag, pred_tag) in zip(str_words[si], ground_truth_tags_bio[si], predicted_tags_bio[si]):
                line = ' '.join([str_word, true_tag, pred_tag])
                prediction.append(line)
            prediction.append('')
    predf = os.path.join(eval_tmp_folder, 'predition.' + name)

    with codecs.open(predf, 'w', 'utf8') as f:
        f.write('\n'.join(prediction))
        f.flush()

    best_F, new_F, save = get_perf_metric(name, best_F)

    return best_F, new_F, save, time_cost
