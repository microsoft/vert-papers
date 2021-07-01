from typing import Dict, Optional, List, Any
import warnings

from overrides import overrides
import torch
import numpy
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import (Seq2SeqEncoder, 
                              TimeDistributed, 
                              TextFieldEmbedder, 
                              Highway,
                              Seq2VecEncoder)
from allennlp.modules import ConditionalRandomField, FeedForward, Pruner, Highway
from allennlp.modules.conditional_random_field import allowed_transitions
import allennlp
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy

from modules.span_based_chunker import SpanBasedChunker
from metrics.span_f1 import MySpanF1


@Model.register("soft_dictionary_span_classifier_HSCRF")
class soft_dictionary_span_classifier_HSCRF(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 feature_size: int,
                 max_span_width: int,
                 encoder: Seq2SeqEncoder,
                 span_label_namespace: str = "span_tags",
                 token_label_namespace: str = "token_tags",
                 calculate_span_f1: bool = None,
                 verbose_metrics: bool = True,
                 feedforward: Optional[FeedForward] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 class_weight=None) -> None:
        
        super().__init__(vocab, regularizer)
        
        self.span_label_namespace = span_label_namespace
        self.token_label_namespace = token_label_namespace
        self.num_span_tags = self.vocab.get_vocab_size(span_label_namespace)
        self.num_token_tags = self.vocab.get_vocab_size(token_label_namespace)
        self.text_field_embedder = text_field_embedder
        
        self.encoder = TimeDistributed(encoder)
        
        self.length_embedder = torch.nn.Embedding(max_span_width, feature_size)

        self.max_span_width = max_span_width
        self.feature_size = feature_size
        self.soft_maxer = torch.nn.Softmax(dim=3)
        
        self._verbose_metrics = verbose_metrics
        self.BILOU_const = 4
        
        if class_weight is not None:
            # assert len(class_weight) == self.num_span_tags - 1, "size of class_weight has to be equal to num_class"
            self.class_weight = numpy.array(class_weight, dtype=numpy.float32)
        else:
            self.class_weight = None
        
        
        if not feedforward:
            self.BILOU_tag_projection_layer = torch.nn.Sequential(
                TimeDistributed( Linear(self.encoder.get_output_dim()+feature_size, self.BILOU_const*(self.num_span_tags-1)) )
            )
        else:
            self.feedforward = feedforward
            self.BILOU_tag_projection_layer = torch.nn.Sequential(
                TimeDistributed(self.feedforward),
                TimeDistributed( Linear(self.feedforward.get_output_dim(), self.BILOU_const*(self.num_span_tags-1)) )
            )
        
        self.metrics = {}
        self.calculate_span_f1 = True
        
        # get mask for loss calculation
        self.label_to_mask_for_loss = torch.nn.Embedding(self.num_span_tags, 2*(self.num_span_tags-1))
        # e.g. PER: [1,0,0,0,| 0,1,1,1], LOC: [0,1,0,0, | ,1,0,1,1], O[0,0,0,0, | ,1,1,1,1]
        self.label_to_mask_for_loss.weight.data.copy_(torch.from_numpy(self._get_label_to_category_mask()))
        
        # get mask for loss calculation
        self.HSCRF_scoring_mask = torch.nn.Embedding(self.max_span_width, 
                                                     self.max_span_width*self.BILOU_const*(self.num_span_tags-1))
        # e.g. PER: [1,0,0,0,| 0,1,1,1], LOC: [0,1,0,0, | ,1,0,1,1], O[0,0,0,0, | ,1,1,1,1]
        self.HSCRF_scoring_mask.weight.data.copy_(torch.from_numpy(self._get_HSCRF_scoring_mask().reshape(self.max_span_width,-1)))

        self._span_f1_metric = MySpanF1()
        initializer(self)

    def _get_label_to_category_mask(self):
        tag_cnter = 0
        label_to_mask = numpy.zeros([self.num_span_tags, 2*(self.num_span_tags-1)],dtype='float32')
        for i in range(self.num_span_tags):
            i_tag = self.vocab.get_token_from_index(i, namespace='span_tags')
            if i_tag == 'O':
                for j in range(1, 2*(self.num_span_tags-1), 2):
                    label_to_mask[i,j] = 1.0                
            else:
                label_to_mask[i,2*tag_cnter] = 1.0
                for j in range(1, 2*(self.num_span_tags-1), 2):
                    if j != 2*tag_cnter+1:
                        label_to_mask[i,j] = 1.0
                tag_cnter += 1
        return label_to_mask
    
    def _get_HSCRF_scoring_mask(self):
        HSCRF_mask = numpy.zeros([self.max_span_width,
                                  self.max_span_width,
                                  self.BILOU_const*(self.num_span_tags-1)],
                                 dtype='float32')
        for i in range(self.max_span_width):
            for j in range(i+1):
                for k in range(0, (self.num_span_tags-1)*self.BILOU_const, self.BILOU_const):
                    if i == j == 0:
                        HSCRF_mask[i,j,k] = 1.0 # U
                    elif j == 0:
                        HSCRF_mask[i,j,k+1] = 1.0 # B
                    elif j == i:
                        HSCRF_mask[i,j,k+3] = 1.0 # L
                    else:
                        HSCRF_mask[i,j,k+2] = 1.0 # I
        return HSCRF_mask
        
    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor, 
                gold_spans: torch.LongTensor, 
                tags: torch.LongTensor = None,
                span_labels: torch.LongTensor = None,
                gold_span_labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        '''
            tags: Shape(batch_size, seq_len)
                bilou scheme tags for crf modelling
        '''
        # Adding mask
        token_mask = util.get_text_field_mask(tokens)
        batch_size, max_seq_length = token_mask.shape
        len_in_token = token_mask.size(1)
        
        #shape (batch_size, num_span, span_width, dim)
        token_embedded = self.text_field_embedder(tokens).unsqueeze(1)
        token_mask = token_mask.unsqueeze(1)
        
        dim_2_pad = self.max_span_width - token_embedded.size(2)
        p2d = (0,0,0, dim_2_pad)
        # now shape (batch_size, num_span, max_span_width, dim)
        token_embedded = F.pad(token_embedded, p2d, "constant", 0.)
        token_mask = F.pad(token_mask, (0, dim_2_pad), "constant", 0.)
        
        length_vec = torch.autograd.Variable(torch.LongTensor(range(self.max_span_width))).cuda(util.get_device_of(spans))
        length_vec = self.length_embedder(length_vec).unsqueeze(0).unsqueeze(0).expand(batch_size, token_embedded.size(1), -1,-1)
        
        token_encoded = self.encoder(token_embedded, token_mask)
        token_encoded = torch.cat((token_encoded, length_vec), 3).contiguous()
        
        # Shape (batch_size, 1)
        lengths = token_mask.sum(-1).long() - 1
        HSCRF_scoring_mask = self.HSCRF_scoring_mask(lengths).detach()
        
        # shape (batch_size, 1 (only 1 span), max_span_wid, 4* span_tags)
        span_logits = self.BILOU_tag_projection_layer(token_encoded)
        span_logits = span_logits.view(batch_size, 
                                          1, 
                                          self.max_span_width,
                                          (self.num_span_tags-1),
                                          self.BILOU_const
                                          )*HSCRF_scoring_mask.view(batch_size, 
                                                                    1, 
                                                                    self.max_span_width, 
                                                                    (self.num_span_tags-1),
                                                                    self.BILOU_const)
        
        final_logits = span_logits.sum(-1).sum(2)
        # add dummy zero for O
        final_logits = torch.stack([final_logits, final_logits.new_zeros(*final_logits.size())], -1)
        
        spans = gold_spans
        span_labels = gold_span_labels
        # spans Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long().view(batch_size, -1 ,2)
        num_spans = spans.size(1)
        
        # shape (batch_size, 1 (only 1 span), span_tags, 2)   is PER / not PER , is LOC / not LOC ...
        span_probs = self.soft_maxer(final_logits).view(batch_size, 1, 2*(self.num_span_tags-1))
        # shape (batch_size, 1 (only 1 span), 2*(self.num_span_tags-1)) 
        span_probs_mask = self.label_to_mask_for_loss(span_labels).detach().float()
        
        
        # TODO:
        # Predict results
        pred_results = []
        mx_prob, mx_ind = span_probs.view(batch_size, 1,-1,2)[:,:,:,0].max(2)
        larger_than_half = mx_prob > 0.5
        # Shape: mx_prob:(batch_size, 1), mx_ind:(batch_size,1)
        for i in range(batch_size):
            pred_span = {}
            phrase_len = len(metadata[i]["words"])
            if mx_prob[i,0] > 0.5:
                pred_span = {(0,phrase_len-1):self.vocab.get_index_to_token_vocabulary(self.span_label_namespace)[int(mx_ind[i,0])]}
                pass
            else:
                pass
            pred_results.append(pred_span)
        
        output = {}
        output['span_logits'] = span_logits 

        ce_loss = span_probs_mask * (1e-6 + span_probs).log()  # may cause NaN error..., possibly use (eps + span_probs).log ?
        output['span_probs_mask'] = span_probs_mask
        output['ce_loss'] = ce_loss.view(batch_size,1,-1,2)
        if self.class_weight is not None:
            # re-weight classes during training
            pass #ce_loss = ce_loss.view(batch_size,1,-1,2) * torch.cuda.FloatTensor(self.class_weight).view(-1,1)  
        ce_loss = -ce_loss.sum() / (batch_size*num_spans)
        output['loss'] = ce_loss
        output['span_probs'] = span_probs.view(batch_size,1,-1,2)
        
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
    

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        if self.calculate_span_f1:
            span_f1_dict = self._span_f1_metric.get_metric(reset=reset)
            span_kys = list(span_f1_dict.keys())
            if self._verbose_metrics:
                metrics_to_return.update(span_f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in span_f1_dict.items() if
                        "overall" in x})
        return metrics_to_return
