import torch.nn as nn
import torch
from typing import Dict, List
from typing import Tuple

B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

class LinearCRF(nn.Module):
    def __init__(self, tag_size: int, schema: str, add_constraint: bool = False,
                 label2idx: Dict = None, device: torch.device = None):
        super(LinearCRF, self).__init__()
        self.label_size = tag_size + 3
        self.label2idx = label2idx
        self.tag_list = list(self.label2idx.keys())
        self.start_idx = tag_size
        self.end_idx = tag_size + 1
        self.pad_idx = tag_size + 2
        self.schema = schema
        self.add_constraint = add_constraint
        self.init_params(device=device)
        return

    def reset(self, label2idx, device):
        if len(label2idx) == len(self.label2idx):
            return
        tag_size = len(label2idx)
        self.label_size = tag_size + 3
        self.label2idx = label2idx
        self.tag_list = list(self.label2idx.keys())
        self.start_idx = tag_size
        self.end_idx = tag_size + 1
        self.pad_idx = tag_size + 2
        self.add_constraint = True
        self.init_params(device)
        return

    def init_params(self, device=None):
        if device is None:
            device = torch.device('cpu')
        init_transition = torch.zeros(self.label_size, self.label_size, device=device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0
        if self.add_constraint:
            if self.schema == "BIO":
                self.add_constraint_for_bio(init_transition)
            elif self.schema == "BIOES":
                self.add_constraint_for_iobes(init_transition)
            else:
                print("[ERROR] wrong schema name!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.transition = nn.Parameter(init_transition, requires_grad=False)
        return

    def add_constraint_for_bio(self, transition: torch.Tensor):
        for prev_label in self.tag_list:
            for next_label in self.tag_list:
                if prev_label == O and next_label.startswith(I_PREF):
                    transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if (prev_label.startswith(B_PREF) or prev_label.startswith(I_PREF)) and next_label.startswith(I_PREF):
                    if prev_label[2:] != next_label[2:]:
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
        for label in self.tag_list:
            if label.startswith(I_PREF):
                transition[self.start_idx, self.label2idx[label]] = -10000.0
        return

    def add_constraint_for_iobes(self, transition: torch.Tensor):
        for prev_label in self.tag_list:
            for next_label in self.tag_list:
                if prev_label == O and (next_label.startswith(I_PREF) or next_label.startswith(E_PREF)):
                    transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith(B_PREF) or prev_label.startswith(I_PREF):
                    if next_label.startswith(O) or next_label.startswith(B_PREF) or next_label.startswith(S_PREF):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                    elif prev_label[2:] != next_label[2:]:
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith(S_PREF) or prev_label.startswith(E_PREF):
                    if next_label.startswith(I_PREF) or next_label.startswith(E_PREF):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
        for label in self.tag_list:
            if label.startswith(I_PREF) or label.startswith(E_PREF):
                transition[self.start_idx, self.label2idx[label]] = -10000.0
            if label.startswith(I_PREF) or label.startswith(B_PREF):
                transition[self.label2idx[label], self.end_idx] = -10000.0
        return

    def forward(self, lstm_scores, word_seq_lens, tags, mask, decode_flag=False):
        all_scores = self.calculate_all_scores(lstm_scores=lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        per_sent_loss = (unlabed_score - labeled_score).sum() / mask.size(0)
        if decode_flag:
            _, decodeIdx = self.viterbi_decode(all_scores, word_seq_lens)
            return per_sent_loss, decodeIdx
        return per_sent_loss

    def get_marginal_score(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        marginal = self.forward_backward(lstm_scores=lstm_scores, word_seq_lens=word_seq_lens)
        return marginal

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] 

        for word_idx in range(1, seq_len):
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = torch.logsumexp(last_alpha.view(batch_size, self.label_size, 1), dim=1).view(batch_size) #log Z(x)
        return last_alpha

    def backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

        last_beta = torch.gather(beta, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_beta = torch.logsumexp(last_beta, dim=1)

        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]
        return torch.sum(last_beta)

    def forward_backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]
        alpha[:, 0, :] = scores[:, 0, self.start_idx, :]
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + scores[ :, word_idx, :, :]
            alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = torch.logsumexp(last_alpha.view(batch_size, self.label_size), dim=-1).view(batch_size, 1, 1).expand(batch_size, seq_len, self.label_size)
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]
        return alpha + beta - last_alpha - lstm_scores

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        tagTransScoresMiddle = None
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, :sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = tagTransScoresBegin + tagTransScoresEnd
        masks = masks.type(torch.float32)

        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.mul(masks[:, 1:]), dim=1)
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths, new_label2idx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if new_label2idx is not None:
            self.reset(new_label2idx, features.device)
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size], device=curr_dev)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64, device=curr_dev)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64, device=curr_dev)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(curr_dev)

        scores = all_scores
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            curIdx = torch.clamp(word_seq_lens - distance2Last - 1, min=1).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)
            lastNIdxRecord = torch.gather(idxRecord, 1, curIdx).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)
        perm_pos = torch.arange(1, sentLength + 1).to(curr_dev)
        perm_pos = perm_pos.unsqueeze(0).expand(batchSize, sentLength)
        perm_pos = word_seq_lens.unsqueeze(1).expand(batchSize, sentLength) - perm_pos
        perm_pos = perm_pos.masked_fill(perm_pos < 0, 0)
        decodeIdx = torch.gather(decodeIdx, 1, perm_pos)
        return bestScores, decodeIdx