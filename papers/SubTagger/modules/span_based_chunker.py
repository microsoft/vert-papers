from typing import List, Tuple, Dict

import torch

from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util


class SpanBasedChunker(torch.nn.Module):
    '''
        binary chunker from span scores for entity recognition.
        
        input: scores(logits) of every possible span of a sequence
        
        output: 
                1. best chunking segments
                2. best scores
                3. sum of the scores across all possible chunking sequences
                        (denominator term for the log-likelihood)
                4. score of a given chunking sequence
    
    '''
    
    def __init__(self):
        super(SpanBasedChunker, self).__init__()
        # nothing...
        return
    
    def forward(self, spans: torch.Tensor, 
                span_scores: torch.Tensor, 
                gold_spans: torch.Tensor, 
                gold_span_labels: torch.Tensor,
                span_mask: torch.Tensor, 
                gold_span_mask: torch.Tensor, 
                token_mask: torch.Tensor):
        
        log_denominator = self._input_likelihood(token_mask, spans, span_scores, span_mask)
        log_numerator   = self._joint_likelihood(token_mask, spans, gold_spans, gold_span_labels, span_scores, gold_span_mask)
        
        return torch.sum(log_numerator - log_denominator)

    
    def _input_likelihood(self, token_mask: torch.Tensor, spans: torch.Tensor, span_scores: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible segmentation sequences. Using Dynamic programming.
        Dp matrix: Shape (batch_size, sequence_length, sequence_length+1)
                          batch_size, current_index  , last segment starting index
        """
        
        def _select_span_score(_span_scores, _spans, _start, _end, keepdim):
            return _span_scores.where((_spans[:,:,0]==_start)*(_spans[:,:,1]==_end), torch.zeros_like(_span_scores)).sum(dim=-1, keepdim=keepdim)

        batch_size, sequence_length = token_mask.size()

        # Transpose batch size and sequence dimensions
        
        # Shape: (num_spans, batch_size, 2)
        spans = spans.transpose(0, 1).contiguous()

        # Shape: (num_spans, batch_size)
        span_mask = span_mask.float().transpose(0, 1).contiguous()
        
        # Shape: (num_spans, batch_size, 1)
        span_scores = span_scores.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, sequence_length, sequence_length+1) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        
        alpha = span_scores.new_zeros(sequence_length+1, batch_size, sequence_length+1)
        # Shape: need to add one when indexing in alpha
        alpha[1, :, 0] = -_select_span_score(span_scores.transpose(0, 1), spans.transpose(0, 1), 0, 0, False) # negative logit --  non-entity score
        alpha[1, :, 1] = _select_span_score(span_scores.transpose(0, 1), spans.transpose(0, 1), 0, 0, False)        
        
        # For each i we compute logits for the transitions from timestep 0:i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(2, sequence_length+1):
            alpha[i, :, 0] = util.logsumexp(alpha[i - 1, :, :i] - _select_span_score(span_scores.transpose(0, 1), 
                                                                                     spans.transpose(0, 1), i-1, i-1, True), 1)
            for j in range(1, i+1):
                # j is the start index of the current span
                # j = 0: current word_i is not an entity
                alpha[i, :, j] = util.logsumexp(alpha[j-1, :, :j] + _select_span_score(span_scores.transpose(0, 1), 
                                                                                       spans.transpose(0, 1), j-1, i-1, True), 1)
                
                # We don't have to apply mask to alpha because we only select the indices of the scores of each sample
                # that we need
        
        total_score = util.batched_index_select(alpha.transpose(0, 1).contiguous(), token_mask.long().sum(-1, keepdim=True))
        total_score = torch.squeeze(total_score)

        total_score[:, 1:] += token_mask.log()
        return util.logsumexp(total_score, 1)
    
    
    
    def _joint_likelihood(self, token_mask: torch.Tensor, spans: torch.Tensor, gold_spans: torch.Tensor, gold_span_labels: torch.Tensor, span_scores: torch.Tensor, gold_span_mask: torch.Tensor) -> torch.Tensor:
    
        def _batch_select_span_score(_span_scores, _spans, _batch_gold_spans, keepdim):
            # _span_scores: shape(batch_size, num_spans)
            # _spans      : shape(batch_size, num_spans, 2)
            #_batch_gold_spans : shape(batch_size, 2)
            # return: shape(batch_size, 1)(if keepdim)
            return _span_scores.where(((_spans[:,:,0]==_batch_gold_spans[:,0].unsqueeze(-1))*(_spans[:,:,1]==_batch_gold_spans[:,1].unsqueeze(-1))),
                                      torch.zeros_like(_span_scores)).sum(dim=-1, keepdim=keepdim)

        batch_size, sequence_length = token_mask.size()
        num_gold_spans = gold_spans.data.shape[1]

        # Transpose batch size and sequence dimensions

        # Shape: (num_gold_spans, batch_size, 2)
        gold_spans = gold_spans.transpose(0, 1).contiguous()

        # Shape: (num_gold_spans, batch_size, 1), 0 for 'O', 1 for 'I'
        gold_span_labels = gold_span_labels.transpose(0, 1).contiguous()

        # Shape: (num_gold_spans, batch_size)
        gold_span_mask = gold_span_mask.float().transpose(0, 1).contiguous()

        # Shape: (num_spans, batch_size, 1)
        span_scores = span_scores.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, sequence_length, sequence_length+1) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.

        alpha = span_scores.new_zeros(batch_size)

        for i in range(num_gold_spans):
            
            alpha += _batch_select_span_score(span_scores.transpose(0,1), 
                                           spans.long(), 
                                           gold_spans[i].long(), keepdim=False) * gold_span_mask[i] * (2*gold_span_labels[i] - 1)

        return alpha
   

    def _best_chunking(self, token_mask: torch.Tensor, spans: torch.Tensor, span_scores: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        '''
            maximize the chunking score
            return:
                gold_spans
                gold_span_labels
                max_score
                gold_spans_masks
        '''    
        def _select_span_score(_span_scores, _spans, _start, _end, keepdim):
            return _span_scores.where((_spans[:,0]==_start)*(_spans[:,1]==_end), torch.zeros_like(_span_scores)).sum(dim=-1, keepdim=keepdim)

        batch_size, sequence_length = token_mask.size()
        # Initial alpha is the (batch_size, sequence_length, sequence_length+1) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.

        # For each i we compute logits for the transitions from timestep 0:i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)

        batch_best_scores = []
        batch_best_chunkings = []
        batch_best_chunking_labels = []
        for _batch_id, (_one_spans, _one_span_scores, _one_span_mask, _one_token_mask) in enumerate(zip(spans, span_scores, span_mask, token_mask)):
            best_chunkings = []
            best_chunking_labels = []

            seq_length = _one_token_mask.long().sum() # one scalar
            # chunks = [[None for _id in range(seq_length+1)] for _jd in range(seq_length+1)]
            alpha = span_scores.new_zeros(seq_length+1, seq_length+1)
            # Shape: need to add one when indexing in alpha
            alpha[1,0] = -_select_span_score(_one_span_scores, _one_spans, 0, 0, False) # negative logit --  non-entity score
            alpha[1,1] = _select_span_score(_one_span_scores, _one_spans, 0, 0, False)

            for i in range(2, int(seq_length)+1):
                # TODO:
                # make it parallel
                alpha[i,0], prev_start_index = (alpha[i - 1,:i] - _select_span_score(_one_span_scores, 
                                            _one_spans, i-1, i-1, False)).max(-1)
                # chunks[i][0] = int(prev_start_index)
                # j = 0: current word_i is not an entity
                for j in range(1, i+1):
                    # j is the start index of the current span
                    alpha[i,j], prev_start_index = (alpha[j-1,:j] + _select_span_score(_one_span_scores, 
                                                    _one_spans, j-1, i-1, False)).max(-1)
                    # chunks[i][j] = int(prev_start_index)
                    # We don't have to apply mask to alpha because we only select the indices of the scores of each sample
                    # that we need

            best_score, last_state = alpha[seq_length, :].max(-1)
            batch_best_scores.append(best_score)

            last_state = int(last_state)
            cur_ptr = int(seq_length)
            while(cur_ptr >= 1):
                if last_state == 0:
                    best_chunkings.append((cur_ptr-1, cur_ptr-1)) # current word does not belong to an entity
                    best_chunking_labels.append(0)
                    cur_ptr -= 1
                else: 
                    best_chunkings.append((last_state-1, cur_ptr-1))
                    best_chunking_labels.append(1)
                    cur_ptr = last_state - 1
                
                # ALERT...
                _, last_state = alpha[cur_ptr, :].max(-1) # ??? shouldn't it be alpha[cur_ptr, :cur_ptr+1].max(-1)
                last_state = int(last_state)

            batch_best_chunkings.append(torch.Tensor(best_chunkings))
            batch_best_chunking_labels.append(torch.Tensor(best_chunking_labels))

        num_spans = torch.tensor([len(x) for x in batch_best_chunkings])
        max_num_spans = num_spans.max()
        chunking_masks = torch.arange(max_num_spans).expand(len(num_spans), max_num_spans) < num_spans.unsqueeze(1)

        batch_best_chunkings = [torch.nn.functional.pad(x,(0,0,0,max_num_spans-x.shape[0]), value=0) for x in batch_best_chunkings]
        batch_best_chunkings = torch.stack(batch_best_chunkings)

        batch_best_chunking_labels = [torch.nn.functional.pad(x, (0, max_num_spans-x.shape[0]), value=0) for x in batch_best_chunking_labels]
        batch_best_chunking_labels = torch.stack(batch_best_chunking_labels)

        batch_best_scores = torch.Tensor(batch_best_scores)


        return {'best_score': batch_best_scores,
                'best_chunkings': batch_best_chunkings,
                'best_chunking_labels': batch_best_chunking_labels, 
                'chunking_masks': chunking_masks}
        
        