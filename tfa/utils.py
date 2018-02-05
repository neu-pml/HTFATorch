#!/usr/bin/env python
"""Utilities for topographic factor analysis"""

__author__ = 'Eli Sennesh'
__email__ = 'e.sennesh@northeastern.edu'

import numpy as np
import scipy.optimize
import torch

try:
    import matplotlib
    matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt

def linear(x, m, b):
    """Your basic linear function f(x) = mx+b"""
    return m * x + b

def exponential(x, a, b, c):
    """Your basic exponential decay function"""
    return np.array((a * torch.exp(-b * torch.DoubleTensor(x)) - c).tolist())

def logistic(x, a, b, c):
    """Your basic logistic function"""
    return np.array((a * (1 + torch.exp(-b * (torch.DoubleTensor(x) - c)))).tolist())

def plot_losses(losses):
    epochs = range(losses.shape[1])

    free_energy_fig = plt.figure(figsize=(10, 10))

    plt.plot(epochs, losses[0,:], 'b.', label='Data')
    try:
        parameters, pcov = scipy.optimize.curve_fit(logistic, epochs, losses[0,:])
        func = logistic
        fit = 'Logistic'
    except RuntimeError:
        logging.warn("Falling back to exponential curve for free-energy figure")
        try:
            parameters, pcov = scipy.optimize.curve_fit(exponential,
                                                        epochs,
                                                        losses[0,:])
            func = exponential
            fit = 'Exponential'
        except RuntimeError:
            logging.warn("Falling back to linear curve for free-energy figure")
            parameters, pcov = scipy.optimize.curve_fit(linear,
                                                        epochs,
                                                        losses[0,:])
            func = linear
            fit = 'Linear'
    plt.plot(epochs, func(epochs, *parameters), 'b', label=fit + " Fit")
    plt.legend()

    free_energy_fig.tight_layout()
    plt.title('Free-energy / -ELBO change over training')
    free_energy_fig.axes[0].set_xlabel('Epoch')
    free_energy_fig.axes[0].set_ylabel('Free-energy / -ELBO (nats)')

    kl_fig = plt.figure(figsize=(10, 10))

    plt.plot(epochs, losses[1, :], 'r.', label='Data')
    try:
        parameters, pcov = scipy.optimize.curve_fit(logistic, epochs, losses[1,:])
        func = logistic
        fit = 'Logistic'
    except RuntimeError:
        logging.warn("Falling back to exponential curve for KL divergence figure")
        try:
            parameters, pcov = scipy.optimize.curve_fit(exponential,
                                                        epochs,
                                                        losses[1,:])
            func = exponential
            fit = 'Exponential'
        except RuntimeError:
            logging.warn("Falling back to linear curve for KL divergence figure")
            parameters, pcov = scipy.optimize.curve_fit(linear,
                                                        epochs,
                                                        losses[1,:])
            func = linear
            fit = 'Linear'
    plt.plot(epochs, func(epochs, *parameters), 'r', label=fit + " Fit")
    plt.legend()

    kl_fig.tight_layout()
    plt.title('KL divergence change over training')
    kl_fig.axes[0].set_xlabel('Epoch')
    kl_fig.axes[0].set_ylabel('KL divergence (nats)')

    plt.show()
