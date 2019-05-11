import torch
from torch import Tensor

import sys
"""
Implementation adapted from https://github.com/ermongroup/neuralsort
"""

class NeuralSort (torch.nn.Module):

    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()

        self.hard = hard
        self.tau = tau

    def forward(self, input : Tensor, scores: Tensor, cuda=None):
        """

        :param input:
        :param scores: logits of the scores by which the elements in input should be sorted.
        :param cuda:
        :return:
        """

        cuda = input.is_cuda if cuda is None else cuda
        dv = 'cuda' if cuda else 'cpu'

        # scores: elements to be sorted. Typical shape: batch_size x n x 1
        scores = scores.unsqueeze(-1)
        bsize, dim = scores.size()[:2]

        #one = torch.cuda.FloatTensor(dim, 1).fill_(1)
        one = torch.ones(dim, 1, device=dv)

        scores = torch.exp(scores)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim, device=dv) + 1)).type(torch.float)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=dv)

            b_idx = torch.arange(bsize, device=dv).repeat([1, dim]).view(dim, bsize)
            b_idx = b_idx.transpose(dim0=1, dim1=0).flatten().type(torch.long)

            r_idx = torch.arange(dim, device=dv).repeat([bsize, 1]).flatten().type(torch.long)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat

        # multiply the input by the permutation matrix
        b, s, z = input.size()
        out = torch.bmm(P_hat, input)

        return out, P_hat