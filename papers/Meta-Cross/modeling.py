import torch
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertForTokenClassification

class BertForTokenClassification_(BertForTokenClassification):

    # add `max-loss`, which is the second term in the `Delta Loss` of https://arxiv.org/pdf/1904.11816v1.pdf
    def forward_wuqh(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, lambda_max_loss=0.0, lambda_mask_loss=0.0):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        batch_size = logits.size(0)
        max_seq_len = logits.size(1)

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')  # modified by wuqh, 2019/5/21
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

                loss_crossEntropy = torch.mean(loss)

                if lambda_max_loss == 0.0 and lambda_mask_loss == 0.0:
                    return loss_crossEntropy

                active_loss = active_loss.view(batch_size, max_seq_len)

                active_max = []
                active_mask = []
                start_id = 0
                for i in range(batch_size):
                    sent_len = torch.sum(active_loss[i])
                    # mask-loss
                    if lambda_mask_loss != 0.0:
                        active_mask.append((input_ids[i] == 103)[: sent_len]) # id of [MASK] is 103, according to the bertTokenizer
                    # max-loss
                    if lambda_max_loss != 0.0:
                        end_id = start_id + sent_len
                        active_max.append(torch.max(loss[start_id: end_id]))
                        start_id = end_id

                # max-loss
                if lambda_max_loss != 0.0:
                    loss_max = torch.mean(torch.stack(active_max))
                else:
                    loss_max = 0.0

                # mask-loss
                if lambda_mask_loss != 0.0:
                    active_mask = torch.cat(active_mask)
                    if sum(active_mask) != 0:
                        loss_mask = torch.sum(loss[active_mask]) / sum(active_mask)
                else:
                    loss_mask = 0.0

                return loss_crossEntropy + lambda_max_loss * loss_max + lambda_mask_loss * loss_mask
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                assert False
        else:
            return logits