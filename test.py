#!/usr/bin/env python3
"""Test ode_gan on a mixture of gaussians distribution, similar to the one in the paper."""

import torch
import numpy as np
import ode_gan
import matplotlib.pyplot as plot

LATENT = 32

class Generator(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(LATENT, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 2))

    def forward(self, z):
        return self.layers(z)

class Discriminator(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1))

    def forward(self, z):
        return self.layers(z)

def main():
    generator = Generator()
    discriminator = Discriminator()

    bce = torch.nn.BCEWithLogitsLoss()

    def g_loss():
        # z = torch.randn(512, LATENT)
        fake_data = generator(z)
        fake_score = discriminator(fake_data)
        return bce(fake_score, torch.ones(fake_score.shape))

    def d_loss():
        # z = torch.randn(512, LATENT)
        # x0 = torch.randint(low=0,high=4,size=[512,2]).float()
        # real_data = 0.2 * torch.randn(512, 2) + x0
        fake_data = generator(z)
        real_score = discriminator(real_data)
        fake_score = discriminator(fake_data)
        return bce(real_score, torch.ones(real_score.shape)) + bce(fake_score, torch.zeros(fake_score.shape))

    opt = ode_gan.RK2(g_params=generator.parameters(), d_params=discriminator.parameters(),
                      g_loss=g_loss, d_loss=d_loss, lr=0.2)

    counter = 1
    while True:
        z = torch.randn(512, LATENT)
        x0 = torch.randint(low=0,high=4,size=[512,2]).float()
        real_data = 0.2 * torch.randn(512, 2) + x0
        opt.step()
        print(counter, g_loss().item(), d_loss().item())
        if counter % 100 == 0:
            plot.cla()
            real_data = real_data.detach().numpy()
            fake_data = generator(z)
            fake_data = fake_data.detach().numpy()
            plot.scatter(real_data[:,0], real_data[:,1], c='r')
            plot.scatter(fake_data[:,0], fake_data[:,1], c='b')
            plot.pause(0.001)
        counter += 1


if __name__ == '__main__':
    main()
