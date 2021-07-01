from typing import Dict, List, Sequence, Iterable, Tuple
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul, enumerate_spans
from allennlp.data.fields import ListField, TextField, SequenceLabelField, Field, MetadataField, SpanField
 
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False
        
        
def _extract_spans(tags: List[str]) -> Dict[Tuple[int, int], str]:
    cur_tag = None
    cur_start = None
    gold_spans = {}    
    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag # inclusive start & end, accord with conll-coref settings
        return _gold_spans
    
    # iterate over the tags
    # (BIO1 scheme)
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing 
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)            
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id+1, gold_spans)
    return gold_spans

# adding span information
@DatasetReader.register("spanconll2003")
class SpanConll2003DatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_tags``, ``chunk_tags``, ``ner_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
        
    Returns:
    
    spans: all possible spans
    span_labels: labels of all possible spans
    gold_spans: entity spans, as well as non-entity singular tokens 
                whose spans are (start, start)
    gold_span_labels: labels associate with gold_spans
    
    tags: token level labels
    
    """
    _VALID_LABELS = {'ner', 'pos', 'chunk'}

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "BIOUL",
                 max_span_width: int = -1, 
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
                
        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.label_namespace = label_namespace
        self.coding_scheme = coding_scheme
        self._original_coding_scheme = "IOB1"
        if max_span_width != -1:
            self._max_span_width = max_span_width
        else:
            self._max_span_width = None
            

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ner_tags = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]
                    yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         chunk_tags: List[str] = None,
                         ner_tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}

        def _remove_BI(_one_tag):
            if _one_tag == 'O':
                return _one_tag
            else:
                return _one_tag[2:]
        
        if self.coding_scheme == "BIOUL":
            coded_chunks = to_bioul(chunk_tags,
                                    encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # TODO:
        # ner_tags -> spans of NE
        # return something like spans, span_labels ("O" if span not in golden_spans, "PER", "LOC"... otherwise)
        spans: List[Field] = []
        span_labels: List[str] = []
            
        gold_spans: List[Field] = []
        gold_span_labels: List[str] = []

        assert len(ner_tags) == len(tokens), "sentence:%s but ner_tags:%s"%(str(tokens), str(ner_tags))
        ner_gold_spans = _extract_spans(ner_tags) # ner_gold_spans: Dict[tuple(startid, endid), str(entity_type)]
        for start, end in enumerate_spans(ner_tags, offset=0, max_span_width=self._max_span_width):
            span_labels.append(ner_gold_spans.get((start, end), 'O'))
            spans.append(SpanField(start, end, sequence))
            pass
        
        _dict_gold_spans = {}
        for ky, val in ner_gold_spans.items():
            gold_span_labels.append(val)
            gold_spans.append(SpanField(ky[0], ky[1], sequence))
            if val != 'O':
                _dict_gold_spans[ky] = val
            pass
        
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens] ,
                                                    "gold_spans": _dict_gold_spans})
        
        assert len(spans) == len(span_labels), "span length not equal to span label length..."
        span_field = ListField(spans) # a list of (start, end) tuples...
        
        # contains all possible spans and there tags
        instance_fields['spans'] = span_field
        instance_fields['span_labels'] = SequenceLabelField(span_labels, span_field, "span_tags")
        
        # only contain gold_spans and there tags
        # e.g. (0,0,O), (1,1,O), (2,3,PER), (4,4,O) for 'I am Donald Trump .'
        gold_span_field = ListField(gold_spans)
        instance_fields['gold_spans'] = gold_span_field
        instance_fields['gold_span_labels'] = SequenceLabelField(gold_span_labels, 
                                                                 gold_span_field, "span_tags")

        # Add "feature labels" to instance
        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError("Dataset reader was specified to use chunk tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['chunk_tags'] = SequenceLabelField(coded_chunks, sequence, "chunk_tags")
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "token_tags")

        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         'token_tags')
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence,
                                                         'token_tags')
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence,
                                                         'token_tags')

        return Instance(instance_fields)