# imports
import pdb

# import matplotlib.pyplot as plt
# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
from sklearn import linear_model

# from torch import Tensor
# from torch import nn as nn
# from torch import optim as optim
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/loss')


class Lasso(nn.Module):
    "Lasso for compressing dictionary, xxxxxx"

    def __init__(self, input_size):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def lasso(x, y, lr=0.05, max_iter=1000, tol=1e-4, opt='adam'):
    # x = x.detach()
    # y = y.detach()
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='sum')
    if opt == 'adam':
        optimizer = optim.Adam(lso.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(lso.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(lso.parameters(), lr=lr)
    w_prev = torch.tensor(0.)
    running_loss = 0.
    for it in range(max_iter):
        # lso.linear.zero_grad()
        optimizer.zero_grad()
        # with torch.no_grad():
        out = lso(x)  # prediction
        loss = criterion(out, y)
        l1_norm = 0.1 * torch.norm(lso.linear.weight, p=1)
        loss += l1_norm
        loss.backward()
        # pdb.set_trace()
        # TODO: perform self-defined optimiztion (Lasso coordinate descent)
        optimizer.step()

        w = lso.linear.weight.detach()
        # if bool(torch.norm(w_prev - w) < tol):
        #     break
        running_loss += loss.item()
        if it % 100 == 0:
            print(f'epoch {it}, loss {loss.item()}')
            print('grad_optm', tt)
        w_prev = w
        # if it % 100 == 0:
        # print(loss.item() - loss_prev)
    return lso.linear.weight.detach()


def lasso_hand(x, y, lr=0.05, max_iter=1000, tol=1e-4):
    # x = x.detach()
    # y = y.detach()
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='sum')

    w2 = lso.linear.weight.detach()
    running_loss = 0.
    with torch.no_grad():
        for it in range(max_iter):
            out = lso(x)  # prediction
            loss = criterion(out, y)
            l1_norm = 0.1 * torch.norm(lso.linear.weight, p=1)
            loss += l1_norm

            rho = torch.zeros((1, x.shape[1]))

            for j in range(x.shape[1]):
                x_j = x[:, j].view(-1, 1)
                gg = x_j.T @ (y - out - w2[:, j] * x_j)
                # pdb.set_trace()
                rho[:, j] = gg
            # pdb.set_trace()
            w2 = soft_threshold(rho, 0.1)
            lso.linear.weight = w2

            # w = lso.linear.weight.detach()
            # if bool(torch.norm(w_prev - w) < tol):
            #     break
            running_loss += loss.item()
            if it % 100 == 0:
                print(f'epoch {it}, loss {loss.item()}')
                # print('grad_optm', tt)
                print('grad_hand', rho)
            # w_prev = w
    return w2


def soft_threshold(X, lbd):
    x1 = torch.where(X > lbd, X - lbd, torch.tensor(0.))
    x2 = torch.where(X < -lbd, X + lbd, torch.tensor(0.))
    return x1 + x2


a = torch.randn((4, 5), device='cpu')
b = torch.randn((4, 1), device='cpu')

# r = lasso(a, b, opt='adam')
r2 = lasso_hand(a, b)
# print(r)
print(r2)
l = linear_model.Lasso(alpha=0.1, fit_intercept=False)
l.fit(a, b)
# l.path(a, b, verbose=True)
print(l.coef_)
