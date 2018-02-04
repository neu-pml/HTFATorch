#!/usr/bin/env python
"""Perform plain topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh'
__email__ = 'e.sennesh@northeastern.edu'

import argparse
import time
import logging
import numpy as np
import probtorch
import scipy.io as sio
import scipy.optimize
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

try:
    import matplotlib
    matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt

# check the availability of CUDA
CUDA = torch.cuda.is_available()
CUDA = False

# placeholder values for hyperparameters
LEARNING_RATE = 1e-7
NUM_FACTORS = 50
NUM_SAMPLES = 100
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

EPOCH_MSG = '[Epoch %d] (%ds) Posterior free-energy %.8e, Joint KL divergence %.8e'

# locations: V x 3
# centers: S x K x 3
# log_widths: S x K
def radial_basis(locations, centers, log_widths, num_voxels,
                 num_samples=NUM_SAMPLES):
    """The radial basis function used as the shape for the factors"""
    # V x 3 -> S x 1 x V x 3
    locations = locations.expand(num_samples, num_voxels, 3).unsqueeze(1)
    # S x K x 3 -> S x K x 1 x 3
    centers = centers.unsqueeze(2)
    # S x K x V x 3
    delta2s = (locations - centers)**2
    # S x K  -> S x K x 1
    log_widths = log_widths.unsqueeze(2)
    return torch.exp(-delta2s.sum(3) / torch.exp(log_widths))

def free_energy(q, p):
    """Calculate the free-energy (negative of the evidence lower bound)"""
    return -probtorch.objectives.montecarlo.elbo(q, p)

def kl_divergence(q, p):
    """Calculate the KL divergence from the joint distribution"""
    return probtorch.objectives.montecarlo.kl(q, p)

class TFAEncoder(nn.Module):
    """Variational guide for topographic factor analysis"""
    def __init__(self, num_times, num_factors=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self._num_factors = num_factors

        self._mean_weight = torch.randn((self._num_times, self._num_factors))
        self._weight_std_dev = torch.sqrt(torch.rand(
            (self._num_times, self._num_factors)
        ))
        self._mean_weight = Parameter(dists.Normal(
            self._mean_weight, self._weight_std_dev
        ).sample())
        self._weight_std_dev = Parameter(self._weight_std_dev)

        self._mean_factor_center = torch.randn((self._num_factors, 3))
        self._factor_center_std_dev = torch.sqrt(torch.rand(
            (self._num_factors, 3)
        ))
        self._mean_factor_center = Parameter(dists.Normal(
            self._mean_factor_center, self._factor_center_std_dev
        ).sample())
        self._factor_center_std_dev = Parameter(self._factor_center_std_dev)

        self._mean_factor_log_width = torch.randn((self._num_factors))
        self._factor_log_width_std_dev = torch.sqrt(torch.rand(
            (self._num_factors)
        ))
        self._mean_factor_log_width = Parameter(dists.Normal(
            self._mean_factor_log_width, self._factor_log_width_std_dev
        ).sample())
        self._factor_log_width_std_dev = Parameter(self._factor_log_width_std_dev)

    def forward(self, num_samples=NUM_SAMPLES):
        q = probtorch.Trace()

        mean_weight = self._mean_weight.expand(num_samples, self._num_times,
                                               self._num_factors)
        weight_std_dev = self._weight_std_dev.expand(num_samples,
                                                     self._num_times,
                                                     self._num_factors)

        mean_factor_center = self._mean_factor_center.expand(num_samples,
                                                             self._num_factors,
                                                             3)
        factor_center_std_dev = self._factor_center_std_dev.expand(
            num_samples, self._num_factors, 3
        )

        mean_factor_log_width = self._mean_factor_log_width.expand(
            num_samples, self._num_factors
        )
        factor_log_width_std_dev = self._factor_log_width_std_dev.expand(
            num_samples, self._num_factors
        )

        q.normal(mean_weight, weight_std_dev, name='Weights')

        q.normal(mean_factor_center, factor_center_std_dev,
                 name='FactorCenters')
        q.normal(mean_factor_log_width, factor_log_width_std_dev,
                 name='FactorLogWidths')

        return q

class TFADecoder(nn.Module):
    """Generative model for topographic factor analysis"""
    def __init__(self, brain_center, brain_center_std_dev, num_times,
                 num_voxels, num_factors=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self._num_factors = num_factors
        self._num_voxels = num_voxels

        self._mean_weight = Variable(torch.zeros(
            (self._num_times, self._num_factors)
        ))
        self._weight_std_dev = Variable(SOURCE_WEIGHT_STD_DEV *
                                        torch.ones((self._num_times,
                                                    self._num_factors)))

        self._mean_factor_center = Variable(
            brain_center.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        )
        self._factor_center_std_dev = Variable(
            brain_center_std_dev.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        )

        self._mean_factor_log_width = Variable(torch.ones((self._num_factors)))
        self._factor_log_width_std_dev = Variable(
            SOURCE_LOG_WIDTH_STD_DEV * torch.ones((self._num_factors))
        )

        self._voxel_noise = Variable(VOXEL_NOISE * torch.ones(self._num_times, self._num_voxels))

    def forward(self, activations, locations, q=None):
        p = probtorch.Trace()

        weights = p.normal(self._mean_weight, self._weight_std_dev,
                           value=q['Weights'], name='Weights')
        factor_centers = p.normal(self._mean_factor_center,
                                  self._factor_center_std_dev,
                                  value=q['FactorCenters'],
                                  name='FactorCenters')
        factor_log_widths = p.normal(self._mean_factor_log_width,
                                     self._factor_log_width_std_dev,
                                     value=q['FactorLogWidths'],
                                     name='FactorLogWidths')
        factors = radial_basis(locations, factor_centers, factor_log_widths,
                               num_voxels=self._num_voxels)
        p.normal(torch.matmul(weights, factors), self._voxel_noise,
                 value=activations, name='Y')

        return p

class TopographicalFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_file):
        dataset = sio.loadmat(data_file)
        # pull out the voxel activations and locations
        self.voxel_activations = torch.Tensor(dataset['data']).transpose(0, 1)
        self.voxel_locations = torch.Tensor(dataset['R'])

        # This could be a huge file.  Close it
        del dataset

        # Pull out relevant dimensions: the number of times-of-recording, and
        # the number of voxels in each timewise "slice"
        self.num_times = self.voxel_activations.shape[0]
        self.num_voxels = self.voxel_activations.shape[1]

        # Estimate further hyperparameters from the dataset
        self.brain_center = torch.mean(self.voxel_locations, 0).unsqueeze(0)
        self.brain_center_std_dev = torch.sqrt(
            10 * torch.var(self.voxel_locations, 0).unsqueeze(0)
        )

        self.enc = TFAEncoder(self.num_times)
        self.dec = TFADecoder(self.brain_center, self.brain_center_std_dev,
                              self.num_times, self.num_voxels)

        if CUDA:
            self.enc = torch.nn.DataParallel(self.enc)
            self.dec = torch.nn.DataParallel(self.dec)
            self.enc.cuda()
            self.dec.cuda()

    def train(self, num_steps=10, log_optimization=False):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        activations = Variable(self.voxel_activations)
        locations = Variable(self.voxel_locations)
        optimizer = torch.optim.Adam(list(self.enc.parameters()), lr=LEARNING_RATE)
        if CUDA:
            activations = activations.cuda()
            locations = locations.cuda()

        self.enc.train()
        self.dec.train()

        free_energies = np.zeros(num_steps)
        kls = np.zeros(num_steps)

        for n in range(num_steps):
            start = time.time()

            optimizer.zero_grad()
            q = self.enc(num_samples=NUM_SAMPLES)
            p = self.dec(activations=activations, locations=locations, q=q)

            free_energy_n = free_energy(q, p)
            kl = kl_divergence(q, p)

            free_energy_n.backward()
            optimizer.step()

            if CUDA:
                free_energy_n = free_energy_n.cpu()
                kl = kl.cpu()
            free_energies[n] = free_energy_n.data.numpy()[0]
            kls[n] = kl.data.numpy()[0]

            end = time.time()
            if log_optimization:
                msg = EPOCH_MSG % (n + 1, end - start, free_energy_n, kl)
                logging.info(msg)

        self.losses = np.vstack([free_energies, kls])
        return self.losses

    def results(self):
        """Return the inferred parameters"""
        q = self.enc(num_samples=NUM_SAMPLES)
        if CUDA:
            q['Weights'].value.data.cpu()
            q['FactorCenters'].value.data.cpu()
            q['FactorLogWidths'].value.data.cpu()

        weights = q['Weights'].value.data.numpy()
        factor_centers = q['FactorCenters'].value.data.numpy()
        factor_log_widths = q['FactorLogWidths'].value.data.numpy()

        result = {
            'weights': weights,
            'factor_centers': factor_centers,
            'factor_log_widths': factor_log_widths
        }
        return result

def linear(x, m, b):
    """Your basic linear function f(x) = mx+b"""
    return m * x + b

def exponential(x, a, b, c):
    """Your basic exponential decay function"""
    return a * np.exp(-b * np.array(x)) - c

def logistic(x, a, b, c):
    """Your basic logistic function"""
    return a * (1 + np.exp(-b * (np.array(x) - c)))

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

parser = argparse.ArgumentParser(description='Topographical factor analysis for fMRI data')
parser.add_argument('data_file', type=str, help='fMRI filename')
parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('--log', action='store_true', help='Whether to log optimization')



if __name__ == '__main__':
    args = parser.parse_args()
    if args.log:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=level)
    tfa = TopographicalFactorAnalysis(args.data_file)
    losses = tfa.train(num_steps=args.steps, log_optimization=args.log)
    plot_losses(losses)
