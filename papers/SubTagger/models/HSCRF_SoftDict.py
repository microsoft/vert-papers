from typing import Dict, Optional, List, Any
import warnings
import copy
import numpy as np

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward, Pruner
from allennlp.modules.conditional_random_field import allowed_transitions
import allennlp
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy

from modules import hscrf_layer_SoftDict
from metrics.span_f1 import MySpanF1




@Model.register("HSCRF_SoftDict")
class HSCRF_SoftDict(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 softdict_text_field_embedder: TextFieldEmbedder,
                 softdict_encoder: Seq2SeqEncoder,
                 softdict_feedforward: FeedForward,
                 softdict_pretrained_path: str,
                 encoder: Seq2SeqEncoder,
                 feature_size: int,
                 max_span_width: int,
                 span_label_namespace: str = "span_tags",
                 token_label_namespace: str = "token_tags",
                 feedforward: Optional[FeedForward] = None,
                 token_label_encoding: Optional[str] = None,
                 constraint_type: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        
        super().__init__(vocab, regularizer)
        self.span_label_namespace = span_label_namespace
        self.token_label_namespace = token_label_namespace
        
        self.num_span_tags = self.vocab.get_vocab_size(span_label_namespace)
        self.num_token_tags = self.vocab.get_vocab_size(token_label_namespace)
        
        self.text_field_embedder = text_field_embedder
        
        self.max_span_width = max_span_width
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        
        self.end_token_embedding = torch.nn.Parameter(torch.zeros(text_field_embedder.get_output_dim()))
        
        bias = np.sqrt( 3.0 / text_field_embedder.get_output_dim())
        torch.nn.init.uniform(self.end_token_embedding, -bias, bias)
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward
        
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
            
        softdict_length_embedder = torch.nn.Embedding(max_span_width, feature_size)
        softdict_encoder = TimeDistributed(softdict_encoder)
        softdict_BILOU_tag_projection_layer = torch.nn.Sequential(
                TimeDistributed(softdict_feedforward),
                TimeDistributed(Linear(softdict_feedforward.get_output_dim(), 4*4))
            )
        
        self.load_weights(softdict_text_field_embedder, 
                         softdict_length_embedder,
                         softdict_encoder,
                         softdict_BILOU_tag_projection_layer,
                         softdict_pretrained_path)
        
        self.hscrf_layer = hscrf_layer_SoftDict.HSCRF(
            ix_to_tag=copy.copy(self.vocab.get_index_to_token_vocabulary(span_label_namespace)),
            word_rep_dim=output_dim,
            ALLOWED_SPANLEN=self.max_span_width,
            softdict_text_field_embedder=softdict_text_field_embedder,
            length_embedder=softdict_length_embedder,
            encoder=softdict_encoder,
            BILOU_tag_projection_layer=softdict_BILOU_tag_projection_layer
        )
        
        if constraint_type is not None:
            token_label_encoding = constraint_type
        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = token_label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = token_label_encoding is not None

        self.token_label_encoding = token_label_encoding # BILOU/BIO/BI
        
        if constrain_crf_decoding:
            token_labels = self.vocab.get_index_to_token_vocabulary(token_label_namespace)
            constraints = allowed_transitions(token_label_encoding, token_labels)
        else:
            constraints = None
            
        self.metrics = {}
        self.calculate_span_f1 = calculate_span_f1
        self._span_f1_metric = MySpanF1()
        
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)
        
    def load_weights(self, 
                     softdict_text_field_embedder, 
                     softdict_length_embedder,
                     softdict_encoder,
                     softdict_BILOU_tag_projection_layer,
                     pretrained_path):
        pretrained_model_state = torch.load(pretrained_path)
        
        softdict_text_field_embedder_statedict = {ky[len('text_field_embedder')+1:]:val for ky,val in pretrained_model_state.items() if ky.startswith('text_field_embedder')}
        tmp = softdict_text_field_embedder_statedict['token_embedder_tokens.weight']
        
        
        softdict_text_field_embedder_statedict['token_embedder_tokens.weight'] = torch.cat([tmp, tmp[0:1].expand(self.vocab.get_vocab_size('tokens') - tmp.size(0), -1)], dim=0)
        tmp = softdict_text_field_embedder_statedict['token_embedder_token_characters._embedding._module.weight']
        softdict_text_field_embedder_statedict['token_embedder_token_characters._embedding._module.weight'] = torch.cat([tmp, 
                                                                                                                         tmp[0:1].expand(self.vocab.get_vocab_size('token_characters') - tmp.size(0), 
                                                                                                                                         -1)], dim=0)
        
        softdict_text_field_embedder.load_state_dict(softdict_text_field_embedder_statedict)
        softdict_text_field_embedder.eval()

        for param in softdict_text_field_embedder.parameters():
            param.requires_grad = False
            
        softdict_length_embedder_statedict = {ky[len('length_embedder')+1:]:val for ky,val in pretrained_model_state.items() if ky.startswith('length_embedder')}
        softdict_length_embedder.load_state_dict(softdict_length_embedder_statedict)
        softdict_length_embedder.eval()
        for param in softdict_length_embedder.parameters():
            param.requires_grad = False
        
        softdict_encoder_statedict = {ky[len('encoder')+1:]:val for ky,val in pretrained_model_state.items() if ky.startswith('encoder')}
        softdict_encoder.load_state_dict(softdict_encoder_statedict)
        softdict_encoder.eval()
        
        for param in softdict_encoder.parameters():
            param.requires_grad = False
        
        softdict_BILOU_tag_projection_layer_statedict = {ky[len('BILOU_tag_projection_layer')+1:]:val for ky,val in pretrained_model_state.items() if ky.startswith('BILOU_tag_projection_layer')}
        softdict_BILOU_tag_projection_layer.load_state_dict(softdict_BILOU_tag_projection_layer_statedict)
        softdict_BILOU_tag_projection_layer.eval()
        
        for param in softdict_BILOU_tag_projection_layer.parameters():
            param.requires_grad = False
            
        return

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
        
        batch_size = spans.size(0)
        # Adding mask
        mask = util.get_text_field_mask(tokens)

        token_mask = torch.cat([mask, 
                                mask.new_zeros(batch_size, 1)],
                                dim=1)

        embedded_text_input = self.text_field_embedder(tokens)

        embedded_text_input = torch.cat([embedded_text_input, 
                                         embedded_text_input.new_zeros(batch_size, 1, embedded_text_input.size(2))],
                                        dim=1)

        # span_mask Shape: (batch_size, num_spans), 1 or 0
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        gold_span_mask = (gold_spans[:,:,0] >=0).squeeze(-1).float()
        last_span_indices = gold_span_mask.sum(-1,keepdim=True).long()

        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        batch_indices = util.move_to_device(batch_indices, 
                                            util.get_device_of(embedded_text_input))
        last_span_indices = torch.cat([batch_indices, last_span_indices],dim=-1)
        embedded_text_input[last_span_indices[:,0], last_span_indices[:,1]] += self.end_token_embedding.cuda(util.get_device_of(spans))

        token_mask[last_span_indices[:,0], last_span_indices[:,1]] += 1.
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.

        # spans Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()
        gold_spans = F.relu(gold_spans.float()).long()
        num_spans = spans.size(1)
        num_gold_spans = gold_spans.size(1)

        # Shape (batch_size, num_gold_spans, 4)
        hscrf_target = torch.cat([gold_spans, gold_spans.new_zeros(*gold_spans.size())],
                                 dim=-1)
        hscrf_target[:,:,2] = torch.cat([
            (gold_span_labels.new_zeros(batch_size, 1)+self.hscrf_layer.start_id).long(), # start tags in the front
            gold_span_labels.squeeze()[:,0:-1]],
            dim=-1)
        hscrf_target[:,:,3] = gold_span_labels.squeeze()
        # Shape (batch_size, num_gold_spans+1, 4)  including an <end> singular-span
        hscrf_target = torch.cat([hscrf_target, gold_spans.new_zeros(batch_size, 1, 4)],
                                 dim=1)

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],0:2] = \
                hscrf_target[last_span_indices[:,0], last_span_indices[:,1]-1][:,1:2] + 1

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],2] = \
                hscrf_target[last_span_indices[:,0], last_span_indices[:,1]-1][:,3]

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],3] = \
                self.hscrf_layer.stop_id
        
        

        # span_mask Shape: (batch_size, num_spans), 1 or 0
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()

        gold_span_mask = torch.cat([gold_span_mask.float(), 
                                gold_span_mask.new_zeros(batch_size, 1).float()], dim=-1)
        gold_span_mask[last_span_indices[:,0], last_span_indices[:,1]] = 1.


        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.

        # spans Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()
        num_spans = spans.size(1)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, token_mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        hscrf_neg_log_likelihood = self.hscrf_layer(
            encoded_text, 
            tokens,
            token_mask.sum(-1).squeeze(),
            hscrf_target,
            gold_span_mask
        )

        pred_results = self.hscrf_layer.get_scrf_decode(
            token_mask.sum(-1).squeeze()
        )
        self._span_f1_metric(
            pred_results, 
            [dic['gold_spans'] for dic in metadata],
            sentences=[x["words"] for x in metadata])
        output = {
            "mask": token_mask,
            "loss": hscrf_neg_log_likelihood,
            "results": pred_results
                 }
        
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
    
    
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        if self.calculate_span_f1:
            span_f1_dict = self._span_f1_metric.get_metric(reset=reset)
            span_kys = list(span_f1_dict.keys())
            for ky in span_kys:
                span_f1_dict[ky] = span_f1_dict.pop(ky)
            if self._verbose_metrics:
                metrics_to_return.update(span_f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in span_f1_dict.items() if
                        "overall" in x})
        return metrics_to_return
