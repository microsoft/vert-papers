# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss

from transformers import BertForTokenClassification


class LanguageDiscriminatorTokenLevel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LanguageDiscriminatorTokenLevel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.loss_fct = BCELoss()
        self.relu = torch.nn.ReLU()

    def forward(self, seqs: list, mask: list, fake: bool = True):
        device = seqs.device
        B, T, H = seqs.shape
        mask = mask.bool()
        labels = torch.tensor(
            [[0.0 if fake else 1.0 for sl in range(T)] for b in range(B)]
        ).to(device)

        # seqs = [batch size, sent len, emb dim]
        # relu = [batch size, sent len, self.hidden_dim]
        relu = self.relu(self.fc1(seqs))
        # logits = [batch size, 2]
        out = self.fc2(relu)
        logits = F.sigmoid(out)
        confidences = torch.cat((1 - logits, logits), dim=2)
        logits = logits.view(-1, 128)

        preds = logits.round()
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)
        corrects = torch.sum(preds == labels)

        logits = torch.masked_select(logits, mask)
        loss = self.loss_fct(logits.view(-1, 1), labels.view(-1, 1))

        return loss, confidences, int(corrects) / int(labels.size(0))


class BertForTokenClassificationKD(BertForTokenClassification):
    def forward(
        self,
        input_ids,
        src_probs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        loss_ignore_index=-100,
        eval=False,
        confidence=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if eval:
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
            # L2 Norm
            loss_KD_fct = MSELoss(reduction="mean" if confidence is None else "none")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            src_probs = torch.nn.functional.softmax(src_probs, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # Note that, even in TS learning, actually here we do NOT use the label information. Instead we just remove the loss w.r.t sub-word starting with "##" in BERT
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)

                if confidence is not None:
                    confidence = (
                        confidence.unsqueeze(1).expand(-1, 128, -1).contiguous()
                    )
                    print(
                        confidence.size(),
                        confidence.view(-1, 2).size(),
                        active_loss.size(),
                    )
                    confidence = confidence.view(-1, 2)[active_loss]
                    confidence = 1 - torch.abs(confidence[:, 1] - 0.5)
                    loss_KD = torch.mean(loss_KD * confidence)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)
            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)
