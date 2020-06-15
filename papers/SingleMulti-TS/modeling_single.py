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

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if src_probs is not None:
            # ## KL Divergence
            # loss_KD_fct = KLDivLoss(reduction="mean")
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_log_probs = log_probs.view(-1, self.num_labels)[active_loss]
            #     active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_log_probs, active_src_probs)
            # else:
            #     loss_KD = loss_KD_fct(log_probs, src_probs)

            # ## CrossEntropy
            # loss_KD_fct = CrossEntropyLoss()
            # src_labels = torch.argmax(src_probs.view(-1, self.num_labels), dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_src_labels = src_labels[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_logits, active_src_labels)
            # else:
            #     loss_KD = loss_KD_fct(logits.view(-1, self.num_labels), src_labels)

            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

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


class LIOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, n_langs, gr_lambda=-1.0):
        super(LIOutputLayer, self).__init__()
        self.n_langs = n_langs

        if gr_lambda > 0:
            self.dropout = torch.nn.Sequential(GradientReversal(lambda_=gr_lambda), torch.nn.Dropout(hidden_dropout_prob))
        else:
            self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, n_langs)

    def forward(self, pooled_output, labels=None):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,)

        if labels is not None:
            if self.n_langs == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.n_langs), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


class TaskOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, n_labels):
        super(TaskOutputLayer, self).__init__()
        self.n_labels = n_labels

        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, n_labels)

    def forward(self, sequence_output, src_probs=None, attention_mask=None, labels=None): # src_probs: batch_size x n_src_langs x n_labels
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # logits: batch_size x seq_len x n_labels

        outputs = (logits,)

        # compute the supervised loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # compute the li_probs weighted KD loss (L2 norm)
        if src_probs is not None:
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_probs = probs.view(-1, self.n_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.n_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), logits


# class LanguageIdentifier(torch.nn.Module):
#     def __init__(self, base_model, output_layer):
#         super(LanguageIdentifier, self).__init__()
#
#         self.base_model = base_model
#         self.output_layer = output_layer
#
# class TaskModel(torch.nn.Module):
#     def __init__(self, base_model, output_layer):
#         super(TaskModel, self).__init__()
#
#         self.base_model = base_model
#         self.output_layer = output_layer


class GRFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from: https://github.com/jvanvugt/pytorch-domain-adaptation
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRFunction.apply(x, self.lambda_)


class DomainLearner(torch.nn.Module):
    # def __init__(self, domain_vocab_size, domain_hidden_size, feature_hidden_size, low_rank_size, weights_init=None):
    #     super(DomainLearner, self).__init__()
    #
    #     self.domain_embed = torch.nn.Parameter(torch.randn(domain_vocab_size, domain_hidden_size))
    #     if weights_init is not None:
    #         self.domain_embed.data.copy_(weights_init)
    #
    #     self.simU = torch.nn.Linear(feature_hidden_size, low_rank_size)
    #     self.simV = torch.nn.Linear(domain_hidden_size, low_rank_size)
    def __init__(self, domain_vocab_size, hidden_size, low_rank_size, weights_init=None, gamma=0.00001):
        super(DomainLearner, self).__init__()
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.domain_embed = torch.nn.Parameter(torch.randn(domain_vocab_size, hidden_size))
        if weights_init is not None:
            self.domain_embed.data.copy_(weights_init)

        self.simU = torch.nn.Linear(hidden_size, low_rank_size)
        self.simV = torch.nn.Linear(hidden_size, low_rank_size)

    def forward(self, features, labels=None, device="cuda"):

        U_fi = self.simU(features) # batch_size x low_rank_size
        V_mu_all = self.simV(self.domain_embed).transpose(0, 1) # vocab_size x low_rank_size = > low_rank_size x vocab_size

        logits = torch.mm(U_fi, V_mu_all) # batch_size x vocab_size

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss() # nn.LogSoftmax() + nn.NLLLoss()
            loss_f = loss_fct(logits, labels)

            R = torch.mm(self.domain_embed.transpose(0, 1), self.domain_embed) - torch.eye(self.hidden_size).to(device)
            loss_R = self.gamma * torch.sum(R * R)

            loss = loss_f + loss_R

            outputs = (loss, loss_f, loss_R) + outputs

        return outputs # (loss), logits

    def get_domain_embeds(self):
        return self.domain_embed.detach()

    def get_domain_similarity(self, domain_idx, domain_idy, method="default"):
        if method == "default":
            f_x = self.simU(self.domain_embed[domain_idx]) # low_rank_size
            f_y = self.simV(self.domain_embed[domain_idy]) # low_rank_size

            sim = torch.sum(f_x * f_y).detach().item()
        elif method == "cosine":
            sim = torch.nn.functional.cosine_similarity(self.domain_embed[domain_idx], self.domain_embed[domain_idy],
                                                        dim=0, eps=1e-8)
        elif method == "l2":
            sim = torch.norm(self.domain_embed[domain_idx] - self.domain_embed[domain_idy])
        else:
            sim = -1.0

        return sim