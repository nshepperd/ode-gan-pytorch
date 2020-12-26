#!/usr/bin/env python
import torch
import numpy as np

class RK2(object):
    """ODE-GAN optimizer using Heun's method (RK2).

    g_loss and d_loss should be functions that accept no parameters
    and compute the loss. They are each evaluated twice per step.

    """
    def __init__(self, g_params, d_params, g_loss, d_loss, lr=0.02, reg=0.002):
        self.g_params = list(g_params)
        self.d_params = list(d_params)
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.lr = lr
        self.reg = reg

    def step(self):
        # First compute gradients at x1
        do_reg = self.reg > 0
        d_grad1 = torch.autograd.grad(self.d_loss(), self.d_params)
        g_grad1 = torch.autograd.grad(self.g_loss(), self.g_params, create_graph = do_reg)

        if do_reg:
            g_grad_magnitude = sum(g.square().sum() for g in g_grad1)
            d_penalty = torch.autograd.grad(g_grad_magnitude, self.d_params)
            [g.detach() for g in g_grad1]
            del g_grad_magnitude

        # Then we step to x2 = x1 - lr*g1
        with torch.no_grad():
            for (param, grad) in zip(self.g_params, g_grad1):
                param.sub_(self.lr * grad)
            for (param, grad) in zip(self.d_params, d_grad1):
                param.sub_(self.lr * grad)

        # At x2 = x1 - lr*g1, we compute gradients again, and move to the
        # destination:
        #
        # x1 - (lr/2)(g1+g2) = x1 - lr*g1 + (lr/2)(g1-g2) = x2 + (lr/2)(g1-g2)
        #
        g_grad2 = torch.autograd.grad(self.g_loss(), self.g_params)
        d_grad2 = torch.autograd.grad(self.d_loss(), self.d_params)
        with torch.no_grad():
            for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                param.add_(0.5 * self.lr * (g1-g2))
            # D gets additional -reg*lr*penalty for gradient regularization.
            if do_reg:
                for (param, g1, g2, gp) in zip(self.d_params, d_grad1, d_grad2, d_penalty):
                    param.add_(0.5 * self.lr * (g1-g2) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2) in zip(self.d_params, d_grad1, d_grad2):
                    param.add_(0.5 * self.lr * (g1-g2))
