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
@DatasetReader.register("BC5CDR")
class BC5CDRDatasetReader(DatasetReader):
    _VALID_LABELS = {'ner', 'pos', 'chunk'}

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 coding_scheme: str = "BIOUL",
                 max_span_width: int = -1, 
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
                
        self.label_namespace = label_namespace
        self.coding_scheme = coding_scheme
        self._original_coding_scheme = "IOB1"
        if max_span_width != -1:
            self._max_span_width = max_span_width
        else:
            self._max_span_width = None            

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        
        def _to_IOB1(_io_s,_ner_tags):
            ans = ["O" for i in range(len(_io_s))]
            spans = []
            left_end, right_end = None, None
            current_type = "None"

            for i, (x,y) in enumerate(zip(_io_s,_ner_tags)):
                if x == "I":
                    if current_type != "None" and i > 0:
                        spans.append((left_end, i-1, current_type))
                    left_end = i
                    current_type = y

            for (_lend,_rend,_type) in spans:
                if _rend-_lend+1 == 1:
                    ans[_lend] = "B-" + _type
                else:
                    ans[_lend] = "B-" + _type
                    for i in range(_lend+1,_rend+1):
                        ans[i] = "I-" + _type
            return ans
        
        
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
                    tokens_,  io_s, ner_tags = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]
                    yield self.text_to_instance(tokens, _to_IOB1(io_s, ner_tags))

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
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
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
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
        
        # contains all possible spans and their tags
        instance_fields['spans'] = span_field
        instance_fields['span_labels'] = SequenceLabelField(span_labels, span_field, "span_tags")
        
        # only contain gold_spans and their tags
        # e.g. (0,0,O), (1,1,O), (2,3,PER), (4,4,O) for 'I am Donald Trump .'
        gold_span_field = ListField(gold_spans)
        instance_fields['gold_spans'] = gold_span_field
        instance_fields['gold_span_labels'] = SequenceLabelField(gold_span_labels, 
                                                                 gold_span_field, "span_tags")


        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         'token_tags')
        return Instance(instance_fields)