from torch.utils.data import Dataset

from constants import Constants
from data_processing import load_dataset_conll
from enums import LabellingSchema


class DataLoaderCoNLL(Dataset):
    """Data loader for loading data from specific folders"""

    def __init__(self, data_file, mappings):
        """
        checked
        :param data_file: string, file path storing the data
        :param mappings: a mapping dictionary containing tag_to_id, id_to_tag, word_to_id, id_to_word, char_to_id, id_to_char
        """
        super(DataLoaderCoNLL, self).__init__()
        self.data_file = data_file

        self.tag_to_id = mappings["tag_to_id"]
        self.word_to_id = mappings["word_to_id"]
        self.char_to_id = mappings["char_to_id"]
        self.word_to_lower = mappings["word_to_lower"]
        self.digits_to_zeros = mappings["digits_to_zeros"]
        self.label_schema = mappings["label_schema"]

        # read the data file
        sentences, tags = load_dataset_conll(data_file, self.label_schema, self.digits_to_zeros)

        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        """
        checked
        Length of the dataset
        :return: length
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        checked
        Get an item of the dataset
        :param idx: index of the item
        :return: corresponding data item
        """
        sentence = self.sentences[idx]
        sentence_words = [word.lower() if self.word_to_lower else word for word in sentence]
        sentence_tag = self.tags[idx]

        sentence_word_id = [self.word_to_id[word if word in self.word_to_id.keys() else Constants.Word_Unknown]
            for word in sentence_words]
        sentence_tag_id = [self.tag_to_id[tag] for tag in sentence_tag]
        sentence_char_id = [[self.char_to_id[c if c in self.char_to_id.keys() else Constants.Char_Unknown] for c in word]
            for word in sentence]

        return {
            "str_words": sentence,
            "words": sentence_word_id,
            "tags": sentence_tag_id,
            "chars": sentence_char_id
        }
