# coding: utf-8
# this file contains Wasserstein Index Generation model class

import torch
from torch import Tensor, nn

# from utils import timer


class WasserIndexGen(nn.Module):
    def __init__(self, batch_size=64, num_topics=4, reg=0.1,
                 numItermax=1000, stopThr=1e-9, dtype=torch.float32,
                 device='cuda'):
        super(WasserIndexGen, self).__init__()

        # hyperparameters
        self.batch_size = batch_size
        self.num_topics = num_topics
        self.reg = reg

        # numeric parameters
        self.numItermax = numItermax
        self.stopThr = stopThr

        # precision level
        self.dtype = dtype
        self.device = torch.device(device)

    def forward(self, a, M, b, lbd, reg):
        loss = sinkloss(a, M, b, lbd, reg,
                        self.numItermax, self.stopThr, self.dtype, self.device,
                        verbose=False)
        return loss


# @timer
def sinkloss(a, M, b, lbd, reg, numItermax, stopThr, dtype, device,
             verbose=False):
    '''
    parameters:
    a: source dist (1 doc),
    b: target dist (k topics),
    lbd: weights of 1 doc -> k topics,
    M: underlying distance to be distilled
    '''
    # torch.reset_default_graph()
    # init data
    Nini = a.shape[0]
    Nfin = b.shape[0]
    nbb = b.shape[1]
    # init u and v
    u = torch.ones((Nini, nbb), dtype=dtype, device=device)
    v = torch.ones((Nfin, nbb), dtype=dtype, device=device)

    C = expjit(M, reg)

    yhat = torch.zeros(a.shape, dtype=dtype, device=device)
    cpt = 0
    err = 1.

    # forward loop update
    while (err > stopThr and cpt < numItermax):
        uprev = u  # b^{l} = u = beta
        vprev = v
        yhatprev = yhat

        C, Cu, u, v, yhat = sinkjit(b, lbd, C, u, v, yhat)

        if (torch.any(Cu == 0) or torch.any(torch.isnan(u)) or
            torch.any(torch.isinf(u)) or torch.any(torch.isnan(v)) or
            torch.any(torch.isinf(v)) or torch.any(torch.isnan(yhat)) or
                torch.any(torch.isinf(yhat))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            yhat = yhatprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            lhs = torch.sum((u - uprev)**2) / torch.sum((u)**2)
            rhs = torch.sum((v - vprev)**2) / torch.sum((v)**2)
            err = lhs + rhs
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    loss = torch.norm(yhat - a) ** 2
    # print(loss)
    # pdb.set_trace()
    return loss


@torch.jit.script
def sinkjit(b, lbd, C: Tensor, u: Tensor, v: Tensor, yhat: Tensor):
    """jit computation in main sinkhorn loop"""
    Cu = torch.matmul(C, u)
    v = torch.matmul(C.transpose(0, 1), torch.div(b, Cu))
    yhat = torch.prod(torch.pow(v, lbd.transpose(0, 1)), dim=1).reshape(-1, 1)
    u = torch.div(yhat, v)  # yhat / phi
    return C, Cu, u, v, yhat


@torch.jit.script
def expjit(M: Tensor, reg: Tensor):
    """C: energy function"""
    return torch.exp(- M / reg)
