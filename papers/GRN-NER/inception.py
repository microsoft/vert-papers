import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=400, kernel_dim=400, inception_mode=1):
        # checked
        super(InceptionCNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim
        self.inception_mode = inception_mode

        if self.inception_mode == 1:
            self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(1, self.kernel_dim), padding=(0, 0))
            self.conv_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(3, self.kernel_dim), padding=(1, 0))
            self.conv_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(5, self.kernel_dim), padding=(2, 0))
            # self.linear = nn.Linear(in_features=self.out_channels*3, out_features=self.out_channels)
        elif self.inception_mode == 2:
            self.conv_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(3, self.kernel_dim), padding=(1, 0))

    def forward(self, input):
        # checked
        if self.inception_mode == 0:
            return input

        batch_size, max_seq, word_embed_size = input.size()

        # batch x 1 x max_seq x word_embed
        input_ = input.unsqueeze(1)

        # batch x out x max_seq x 1
        if self.inception_mode == 1:
            input_1 = F.tanh(self.conv_1(input_))[:, :, :max_seq, :]
            input_3 = F.tanh(self.conv_3(input_))[:, :, :max_seq, :]
            input_5 = F.tanh(self.conv_5(input_))[:, :, :max_seq, :]

            # # batch x (3*out) x max_seq --> batch x max_seq x (3*out)
            # linear_input = torch.cat([input_1.squeeze(3), input_3.squeeze(3), input_5.squeeze(3)], 1).transpose(1, 2)
            # # batch x max_seq x out
            # output = self.linear(linear_input)

            # batch x max_seq x out
            # output = (input_1 + input_3 + input_5).squeeze(3).transpose(1, 2)

            # batch x out_channels x max_seq x 3
            pooling_input = torch.cat([input_1, input_3, input_5], 3)
            # batch x out_channels x max_seq x 1
            output = F.max_pool2d(pooling_input, kernel_size=(1, pooling_input.size(3)))
        elif self.inception_mode == 2:
            output = F.tanh(self.conv_3(input_))[:, :, :max_seq, :]

        # batch x out x max_seq -> batch x max_seq x out
        output = output.squeeze(3).transpose(1, 2)

        #assert output.size() == input.size()

        return output