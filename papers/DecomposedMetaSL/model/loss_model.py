import torch
from torch import nn
import torch.nn.functional as F

class MaxLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(MaxLoss, self).__init__()
        self.gamma = gamma
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        return

    def forward(self, logits, targets):
        assert logits.dim() == 3 #batch_size, seq_len, label_num
        batch_size, seq_len, class_num = logits.size()
        token_loss = self.loss_fct(logits.view(-1, class_num), targets.view(-1)).view(batch_size, seq_len)
        act_pos = targets.ne(-1)
        loss = token_loss[act_pos].mean() 
        if self.gamma > 0:
            max_loss = torch.max(token_loss, dim=1)[0]
            loss += self.gamma * max_loss.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        return

    def forward(self, logits, targets):
        assert logits.dim() == 2
        assert torch.min(targets).item() > -1

        logp = F.log_softmax(logits, dim=1)
        target_logp = logp.gather(1, targets.view(-1, 1)).view(-1)
        target_p = torch.exp(target_logp)
        weight = (1 - target_p) ** self.gamma
        loss = - weight * target_logp
        if self.reduction:
            loss = loss.mean()
        return loss