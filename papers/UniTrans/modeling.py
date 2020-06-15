import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForTokenClassification, BertPreTrainedModel, BertModel

class BertForTokenClassification_(BertForTokenClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    # def __init__(self, config):
    #     super(BertForTokenClassification, self).__init__(config)
    #     self.num_labels = config.num_labels
    #
    #     self.bert = BertModel(config)
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    #
    #     self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, loss_ignore_index=-100, active_CE=None, pseudo_labels=None, src_probs=None,
                lambda_original_loss=1.0, loss_with_crossEntropy=False, weight_crossEntropy=False):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            # Only keep active parts of the loss
            active_loss = (labels.view(-1) != loss_ignore_index) # batch_size x max_seq_len
            if attention_mask is not None:
                active_loss &= (attention_mask.view(-1) == 1)

            # active_CE: used in the unified loss, only compute cross entropy loss for translated data.
            if active_CE is not None:
                active_loss &= active_CE.view(-1)

            if pseudo_labels is not None:
                active_labels = pseudo_labels.view(-1)[active_loss]
            else:
                active_labels = labels.view(-1)[active_loss]

            active_logits = logits.view(-1, self.num_labels)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            # weight_crossEntropy: weight cross entropy loss of a token with corresponding probability obatained from the teacher model.
            if src_probs is not None and weight_crossEntropy:
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss] # () x num_labels
                active_src_probs = active_src_probs[range(len(active_labels)), active_labels] # ()
                loss = torch.mean(active_src_probs * loss)
            else:
                loss = torch.mean(loss)

            outputs = (loss,) + outputs

        if src_probs is not None:
            if lambda_original_loss > 0:
                loss_KD_fct = MSELoss(reduction="none")

                active_loss = (labels != loss_ignore_index)  # batch_size x max_seq_len
                if attention_mask is not None:
                    active_loss &= (attention_mask == 1)

                probs = torch.nn.functional.softmax(logits, dim=-1)
                loss_KD = loss_KD_fct(probs, src_probs)  # batch_size x max_seq_len x num_labels

                loss_KD = lambda_original_loss * torch.mean(loss_KD.view(-1, self.num_labels)[active_loss.view(-1)])

                if loss_with_crossEntropy:
                    loss_KD += loss

            else:
                loss_KD = loss if loss_with_crossEntropy else None

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)


class BaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BaseModel, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        # sequence_output = outputs[0]
        # pooled_output = outputs[1]

        return outputs

class ViterbiDecoder():
    def __init__(self, labels, pad_token_label_id, device):
        self.n_labels = len(labels)
        self.pad_token_label_id = pad_token_label_id
        self.label_map = {i: label for i, label in enumerate(labels)}

        self.transitions = torch.zeros([self.n_labels, self.n_labels], device=device) # pij: p(j -> i)
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                if labels[i][0] == "I" and labels[j][-3:] != labels[i][-3:]:
                    self.transitions[i, j] = -10000

    def forward(self, logprobs, attention_mask, label_ids):
        active_tokens = (attention_mask == 1) & (label_ids != self.pad_token_label_id)

        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        # scores = []
        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][active_tokens[idx]] # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0] # n_labels

            for j in range(1, len(logprob_i)): # for tag_feat in feat:
                next_label_var = forward_var + self.transitions # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1) # n_labels

                logp_j = logprob_i[j] # n_labels
                forward_var = viterbivars_t + logp_j # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            # terminal_var = forward_var

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            # path_score = path_score.item()
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append([self.label_map[label_id] for label_id in best_path])
            # scores.append(path_score)

        return label_seqs #, scores

