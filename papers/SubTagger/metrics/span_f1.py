from typing import Any, Dict, List, Set, Tuple
from overrides import overrides
from collections import defaultdict
import torch

from allennlp.training.metrics.metric import Metric 


@Metric.register("myspanf1")
class MySpanF1(Metric):
    def __init__(self, non_entity_labels=['O']) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP = defaultdict(int)
        self._FP = defaultdict(int)
        self._TN = defaultdict(int)
        self._FN = defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)
        
        
    @overrides
    def __call__(self,  # type: ignore
                 batched_predicted_spans,
                 batched_gold_spans,
                 sentences=None):
        
        non_entity_labels = self.non_entity_labels
        for predicted_spans, gold_spans, sent in zip(batched_predicted_spans, batched_gold_spans, sentences):            
            self._num_gold_mentions += len(gold_spans)
            self._num_recalled_mentions += len(set(gold_spans) & set([x for x,y in predicted_spans.items() if y not in non_entity_labels]))
            self._num_predicted_mentions += len([x for x, y in predicted_spans.items() if y not in non_entity_labels])
            mem_dict = {}
            for ky, val in predicted_spans.items():
                if val in non_entity_labels:
                    continue
                if ky in gold_spans and val == gold_spans[ky]:
                    self._TP[val] += 1
                    mem_dict[ky] = True
                else:
                    self._FP[val] += 1
            for ky, val in gold_spans.items():
                if ky not in mem_dict:
                    self._FN[val] += 1
                    
                     

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        
        all_tags: Set[str] = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._FN.keys())
        all_metrics = {}
        
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._TP[tag],
                                                                  self._FP[tag],
                                                                  self._FN[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._TP.values()),
                                                              sum(self._FP.values()),
                                                              sum(self._FN.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        
     
        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions/float(self._num_gold_mentions)
            
        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)
        
        all_metrics['entity_recall'] = entity_recall
        all_metrics['entity_precision'] = entity_precision
        all_metrics['entity_f1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['entity_ALLTRUE'] = self._num_gold_mentions
        all_metrics['entity_ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['entity_ALLPRED'] = self._num_predicted_mentions
        if reset:
            self.reset()
        return all_metrics
    

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    @overrides
    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP = defaultdict(int)
        self._FP = defaultdict(int)
        self._TN = defaultdict(int)
        self._FN = defaultdict(int)
        