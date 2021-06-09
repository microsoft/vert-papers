from __future__ import print_function, division

import codecs

import torch

from constants import Constants
from data_format_util import iob1_to_iob2, iob2_to_iobes, digit_to_zero
from enums import LabellingSchema
from enums import CharEmbeddingSchema


def load_prebuilt_word_embedding(embedding_path, embedding_dim):
    """
    checked
    Read prebuilt word embeddings from a file
    :param embedding_path: string, file path of the word embeddings
    :param embedding_dim: int, dimensionality of the word embeddings
    :return: a dictionary mapping each word to its corresponding word embeddings
    """
    word_embedding_map = dict()

    if embedding_path is not None and len(embedding_path) > 0:
        for line in codecs.open(embedding_path, mode="r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            else:
                word_embedding = line.split()
                assert len(word_embedding) == 1 + embedding_dim
                word = word_embedding[0]
                embedding = [float(val) for val in word_embedding[1:]]
                if word in word_embedding_map.keys():
                    continue
                else:
                    word_embedding_map[word] = embedding

    return word_embedding_map


def create_mapping(words, freq_threshold=1):
    """
    checked
    Build a lookup table mapping word to its corresponding index
    :param words: list of string, the list of words, may not be unique at all
    :param freq_threshold: int, only keep words with the corresponding frequency larger than freq_threshold
    :return: lookup table
    """

    assert words is not None

    if freq_threshold > 1:
        word_freq = dict()
        for word in words:
            if word in word_freq.keys():
                word_freq[word] = word_freq[word] + 1
            else:
                word_freq[word] = 1
        word_set = sorted(set([k for k, v in word_freq.items() if v >= freq_threshold]))
    else:
        word_set = sorted(set(words))

    lookup_table = dict()
    for word in word_set:
        lookup_table[word] = len(lookup_table)

    return lookup_table


def create_mapping_tags(tags):
    """
    checked
    Build up the lookup tables for all tags
    :param tags: list of string, all tags
    :return: the tag_to_id and the id_to_tag lookup tables
    """
    tag_to_id = create_mapping(tags)
    if any(x in tag_to_id.keys() for x in [Constants.Tag_End, Constants.Tag_Start]):
        raise Exception("Error: <START> or <END> cannot appear in the tag set")
    else:
        tag_to_id[Constants.Tag_Start] = len(tag_to_id)
        tag_to_id[Constants.Tag_End] = len(tag_to_id)

    id_to_tag = dict([(v, k) for k, v in tag_to_id.items()])

    return tag_to_id, id_to_tag


def create_mapping_words(words, embedding_path, embedding_dim, to_lower=False, freq_threshold=1):
    """
    checked
    Build up the lookup tables for all words
    :param words: list of string, all words
    :param embedding_path: string, file path for loading prebuilt word embeddings
    :param embedding_dim: int, dimensionality of the word embeddings
    :param to_lower: bool, invoke lower() or not
    :param freq_threshold: int, only keep words with the frequency larger than freq_threshold
    :return: the word_to_id and the id_to_word lookup tables
    """

    word_to_id = create_mapping(words, freq_threshold)
    prebuilt_word_embedding = dict()

    if embedding_path is not None and len(embedding_path) > 0:
        prebuilt_word_embedding = load_prebuilt_word_embedding(embedding_path, embedding_dim)
        sorted_prebuilt_words = sorted(prebuilt_word_embedding.keys())
        for word in sorted_prebuilt_words:
            word = word.lower() if to_lower else word
            if word not in word_to_id.keys():
                word_to_id[word] = len(word_to_id)

    if any(x in word_to_id.keys() for x in [Constants.Word_Pad, Constants.Word_Unknown]):
        raise Exception("Error: <PAD> or <UNK> cannot appear in the word set")
    else:
        word_to_id[Constants.Word_Unknown] = len(word_to_id)
        word_to_id[Constants.Word_Pad] = len(word_to_id)

    id_to_word = dict([(v, k) for k, v in word_to_id.items()])

    return word_to_id, id_to_word, prebuilt_word_embedding


def create_mapping_chars(chars):
    """
    checked
    Build up the lookup tables for all characters
    :param chars: list of string, all characters
    :return: the char_to_id and the id_to_char lookup tables
    """
    char_to_id = create_mapping(chars)
    if any(x in char_to_id.keys() for x in [Constants.Char_Pad, Constants.Char_Unknown]):
        raise Exception("Error: <C_PAD> or <C_UNK> cannot appear in the char set")
    else:
        char_to_id[Constants.Char_Unknown] = len(char_to_id)
        char_to_id[Constants.Char_Pad] = len(char_to_id)

    id_to_char = dict([(v, k) for k, v in char_to_id.items()])

    return char_to_id, id_to_char


def load_dataset_conll(data_file, label_schema=LabellingSchema.IOBES, digits_to_zeros=False, remove_doc_start=True):
    """
    checked
    Read sentences from a CoNLL format data file
    :param data_file: string, path of the data set
    :param label_schema: enum of LabellingSchema, the labelling scheme
    :param digits_to_zeros: bool, transfer all digits into zeros
    :return: sentences of the dataset, with each sentence containing word_tag pairs
    """
    # read the data file
    sentences = []
    tags = []
    sentence = []
    sentence_tag = []
    for line in codecs.open(data_file, mode="r", encoding="utf-8"):
        line = line.strip()
        if not line:
            if len(sentence) > 0 and ((not remove_doc_start) or ("DOCSTART" not in sentence[0])):
                sentences.append(sentence)
                tags.append(sentence_tag)
            sentence = []
            sentence_tag = []
        else:
            words = line.split()
            assert len(words) >= 2
            sentence.append(digit_to_zero(words[0]) if digits_to_zeros else words[0])
            sentence_tag.append(words[-1])

    if len(sentence) > 0 and ((not remove_doc_start) or ("DOCSTART" not in sentence[0])):
        sentences.append(sentence)
        tags.append(sentence_tag)

    assert len(sentences) > 0
    assert len(sentences) == len(tags)

    # from IOB1 => IOB2 => IOBES
    for i, sentence_tag in enumerate(tags):
        if iob1_to_iob2(sentence_tag):
            if label_schema == LabellingSchema.IOBES:
                sentence_tag = iob2_to_iobes(sentence_tag)
            tags[i] = sentence_tag
        else:
            raise Exception("Error: the input dataset is not in IOB format")

    return sentences, tags


def create_mapping_dataset_conll(data_paths,
                                 word_embedding_path,
                                 word_embedding_dim,
                                 label_schema=LabellingSchema.IOBES,
                                 word_to_lower=False,
                                 word_freq_threshold=1,
                                 digits_to_zeros=False):
    """
    checked
    Build the mapping for words/tags/chars for a given list of datasets
    :param data_paths: list of string, file paths of to-be-processed dataset
    :param word_embedding_path: string, file path for the prebuilt word embedding
    :param word_embedding_dim: int, dimensionality of the prebuilt word embedding
    :param label_schema: LabellingSchema, labelling scheme for the dataset
    :param word_to_lower: bool, transform all words to lower case or not
    :param word_freq_threshold: int, only keep words with the corresponding frequency larger than word_freq_threshold
    :param digits_to_zeros: bool, transfer all digits into zeros
    :return: a mapping dictionary containing tag_to_id, id_to_tag, word_to_id, id_to_word, char_to_id, id_to_char
            and also the prebuilt word embedding dictionary
    """
    assert isinstance(data_paths, list)
    assert len(data_paths) > 0

    all_sentences = []
    all_sentence_tags = []
    for data_path in data_paths:
        dataset_sentences, dataset_tags = load_dataset_conll(data_path, label_schema, digits_to_zeros)
        all_sentences.extend(dataset_sentences)
        all_sentence_tags.extend(dataset_tags)

    assert len(all_sentences) == len(all_sentence_tags)

    tags = []
    words = []
    chars = []
    max_word_length = 0
    for i_sentence in range(len(all_sentences)):
        sentence_words = all_sentences[i_sentence]
        sentence_tags = all_sentence_tags[i_sentence]

        assert len(sentence_words) == len(sentence_tags)

        words.extend([word.lower() if word_to_lower else word for word in sentence_words])
        tags.extend(sentence_tags)
        for wi, sentence_word in enumerate(sentence_words):
            sentence_word_chars = [c for c in sentence_word]
            chars.extend(sentence_word_chars)
            max_word_length = max_word_length if max_word_length >= len(sentence_word_chars) else len(sentence_word_chars)

    tag_to_id, id_to_tag = create_mapping_tags(tags)
    word_to_id, id_to_word, prebuilt_word_embedding = create_mapping_words(words,
                                                                           word_embedding_path,
                                                                           word_embedding_dim,
                                                                           word_to_lower,
                                                                           word_freq_threshold)
    char_to_id, id_to_char = create_mapping_chars(chars)

    mappings = {
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "max_word_length": max_word_length,
        "word_to_lower": word_to_lower,
        "digits_to_zeros": digits_to_zeros,
        "label_schema": label_schema
    }

    return mappings, prebuilt_word_embedding


def pad_word_chars(words, char_pad_id):
    """
    Pad the characters of the words in a sentence.
    Input:
        :param words: list of list of int, char ids of all words in the sentence
        :param char_pad_id: int, id of Char_Pad
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [char_pad_id] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def generate_mini_batch_input(data_loader_conll, mini_batch_idx, mappings, char_mode):
    """
    checked
    Pack a mini-batch into tensors
    :param data_loader_conll: DataLoaderConLL, the given data loader
    :param mini_batch_idx: list, indecies of the samples in the training set
    :param mappings: dict, a mapping dictionary containing tag_to_id, id_to_tag, word_to_id, id_to_word, char_to_id, id_to_char
    :param char_mode: CharEmbeddingSchema, char embedding type, in [LSTM, CNN]
    :return: a dict containing masks, tags, words, and chars
    """
    sentences = [data_loader_conll[idx] for idx in mini_batch_idx]
    sentences = sorted(sentences, key=lambda x: len(x["words"]), reverse=True) # longest sentence must be the first
    max_sentence_length = max([len(sentence["words"]) for sentence in sentences])
    max_word_length = max([max(len(tmp_chars) for tmp_chars in sentence["chars"]) for sentence in sentences])

    assert max_sentence_length > 0
    assert len(sentences) > 0
    assert max_word_length > 0

    tag_to_id = mappings["tag_to_id"]
    word_to_id = mappings["word_to_id"]
    char_to_id = mappings["char_to_id"]

    sentence_masks = []
    words = []
    sentence_char_lengths = []
    chars = []
    chars_positions = []
    tags = []
    str_words = []
    unaligned_tags = []

    for si, sentence in enumerate(sentences):
        _words = sentence["words"]
        _chars = sentence["chars"]
        _tags = sentence["tags"]
        _str_words = sentence["str_words"]
        unaligned_tags.append(_tags.copy())

        length = len(_words)
        pad_word_num = max_sentence_length - length

        _words.extend([word_to_id[Constants.Word_Pad]] * pad_word_num)
        _tags.extend([tag_to_id[Constants.Tag_End]] * pad_word_num)

        char_lengths = [len(tmp_chars) for tmp_chars in _chars]
        sentence_char_lengths.extend(char_lengths)

        _chars_positions = [(si, wi) for wi in range(len(_chars))]
        chars_positions.extend(_chars_positions)

        _mask = [1] * length
        _mask.extend([0] * pad_word_num)

        sentence_masks.append(_mask)
        words.append(_words)
        tags.append(_tags)
        chars.extend(_chars)
        str_words.append(_str_words)

    sentence_char_position_map = {}
    if char_mode == CharEmbeddingSchema.LSTM:
        # rank by the word length in a descending order
        chars_with_positions = zip(chars, chars_positions, sentence_char_lengths)
        sorted_chars_with_positions = sorted(chars_with_positions, key=lambda x: len(x[0]), reverse=True)
        chars, chars_positions, sentence_char_lengths = zip(*sorted_chars_with_positions)

    sentence_char_position_map = dict([(j, chars_positions[j]) for j in range(len(chars_positions))])
    for word_chars in chars:
        word_chars.extend([char_to_id[Constants.Char_Pad]] * (max_word_length - len(word_chars)))

    sentence_masks_tensor = torch.tensor(sentence_masks, requires_grad=False, dtype=torch.long)
    words_tensor = torch.tensor(words, requires_grad=False, dtype=torch.long)
    chars_tensor = torch.tensor(chars, requires_grad=False, dtype=torch.long)
    tags_tensor = torch.tensor(tags, requires_grad=False, dtype=torch.long)
    sentence_char_lengths_tensor = torch.tensor(sentence_char_lengths, requires_grad=False, dtype=torch.long)

    return sentence_masks_tensor, words_tensor, chars_tensor, tags_tensor, \
           sentence_char_lengths_tensor, sentence_char_position_map, str_words, unaligned_tags