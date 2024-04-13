#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""RNN."""
from .base import *


class BoundRNN(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.complex = True
        self.output_index = output_index
        raise NotImplementedError(
            'torch.nn.RNN is not supported at this time.'
            'Please implement your RNN with torch.nn.RNNCell and a manual for-loop.'
            'See an example of LSTM:'
            'https://github.com/Verified-Intelligence/auto_LiRPA/blob/10a9b30/examples/sequence/lstm.py#L9')

    def forward(self, x, weight_input, weight_recurrent, bias, sequence_length, initial_h):
        assert (torch.sum(torch.abs(initial_h)) == 0)

        self.input_size = x.shape[-1]
        self.hidden_size = weight_input.shape[-2]

        class BoundRNNImpl(nn.Module):
            def __init__(self, input_size, hidden_size,
                         weight_input, weight_recurrent, bias, output_index):
                super().__init__()

                self.input_size = input_size
                self.hidden_size = hidden_size

                self.cell = torch.nn.RNNCell(
                    input_size=input_size,
                    hidden_size=hidden_size
                )

                self.cell.weight_ih.data.copy_(weight_input.squeeze(0).data)
                self.cell.weight_hh.data.copy_(weight_recurrent.squeeze(0).data)
                self.cell.bias_ih.data.copy_((bias.squeeze(0))[:hidden_size].data)
                self.cell.bias_hh.data.copy_((bias.squeeze(0))[hidden_size:].data)

                self.output_index = output_index

            def forward(self, x, hidden):
                length = x.shape[0]
                outputs = []
                for i in range(length):
                    hidden = self.cell(x[i, :], hidden)
                    outputs.append(hidden.unsqueeze(0))
                outputs = torch.cat(outputs, dim=0)

                if self.output_index == 0:
                    return outputs
                else:
                    return hidden

        self.model = BoundRNNImpl(
            self.input_size, self.hidden_size,
            weight_input, weight_recurrent, bias,
            self.output_index)
        self.input = (x, initial_h)

        return self.model(*self.input)