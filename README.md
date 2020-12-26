Pytorch implementation of [Training Generative Adversarial Networks
by Solving Ordinary Differential
Equations](https://arxiv.org/abs/2010.15040).

Usage
-----

Just drop `ode_gan.py` in your project directory, import ode_gan, and
instantiate your choice of optimizer (currently only `RK2` [Heun's
method] is implemented), passing the generator and discriminator
parameters and loss functions.

Call the `step()` method in your training loop to execute a single
training step.

See Also
--------

https://github.com/nshepperd/ode-gan-tf Tensorflow implementation.
