import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextLayer(nn.Module):
    def __init__(self, feature_dim=400, exclusive_context=False, use_gpu=True):
        # checked
        super(ContextLayer, self).__init__()

        self.feature_dim = feature_dim
        self.exclusive_context = exclusive_context
        self.use_gpu = use_gpu

        self.forget_gate_linear0 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.forget_gate_linear1 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)

    def forward(self, input, input_masks, device):
        """
        checked
        Get the context vector for each word
        :param input: features, batch x max_seq x embed/feature
        :param input_masks: binary mask matrix for the sentence words, batch x max_seq
        :param device: device to run the algorithm
        :return: context vector for each word, batch x max_seq x embed/feature
        """
        batch_size, max_seq_size, embed_size = input.size()
        sentence_lengths = torch.sum(input_masks, 1)

        assert self.feature_dim == embed_size

        # attention-based context information
        # batch x max_seq x embed --> batch x max_seq x max_seq x embed
        forget_gate0_linear = self.forget_gate_linear0(input)
        forget_gate1_linear = self.forget_gate_linear1(input)

        sigmoid_input = forget_gate0_linear.view(batch_size, max_seq_size, 1, embed_size)\
            .expand(batch_size, max_seq_size, max_seq_size, embed_size) + forget_gate1_linear\
            .view(batch_size, 1, max_seq_size, embed_size).expand(batch_size, max_seq_size, max_seq_size, embed_size)

        # batch x max_seq x max_seq x embed
        forget_gate = torch.sigmoid(sigmoid_input)

        input_row_expanded = input.view(batch_size, 1, max_seq_size, embed_size) \
            .expand(batch_size, max_seq_size, max_seq_size, embed_size)

        forget_result = torch.mul(input_row_expanded, forget_gate)

        # start_t0 = time.time()
        selection_mask = input_masks.view(batch_size, max_seq_size, 1) \
            .mul(input_masks.view(batch_size, 1, max_seq_size))

        if self.exclusive_context:
            eye_matrix_seed = torch.eye(max_seq_size).byte()
            if self.use_gpu:
                eye_matrix_seed = eye_matrix_seed.to(device)
            eye_matrix = eye_matrix_seed.view(1, max_seq_size, max_seq_size) \
                .expand(batch_size, max_seq_size, max_seq_size)
            selection_mask.masked_fill_(eye_matrix, 0)  # exclude each word itself
            context_lengths = (sentence_lengths + sentence_lengths.eq(1).long()) - 1
        else:
            context_lengths = sentence_lengths

        selection_mask = selection_mask.view(batch_size, max_seq_size, max_seq_size, 1) \
            .expand(batch_size, max_seq_size, max_seq_size, self.feature_dim)

        # batch x max_seq x max_seq x embed
        forget_result_masked = torch.mul(forget_result, selection_mask.float())

        # batch x max_seq x embed
        context_sumup = torch.sum(forget_result_masked, 2)

        # average
        context_vector = torch.div(context_sumup, context_lengths.view(batch_size, 1, 1)
                                   .expand(batch_size, max_seq_size, self.feature_dim).float())

        output_result = F.tanh(context_vector)

        return output_result
