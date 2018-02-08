#!/usr/bin/env python
"""Perform plain topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import argparse
import time
import logging
import numpy as np
import probtorch
import scipy.io as sio
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data
from sklearn.cluster import KMeans
import math

import utils

# check the availability of CUDA
CUDA = torch.cuda.is_available()

# placeholder values for hyperparameters
LEARNING_RATE = 1e-4
NUM_FACTORS = 5
NUM_SAMPLES = 100
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

EPOCH_MSG = '[Epoch %d] (%dms) Posterior free-energy %.8e'

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

def initial_radial_basis(location, center, widths):
    """The radial basis function used as the shape for the factors"""
    # V x 3 -> 1 x V x 3
    location = np.expand_dims(location, 0)
    # K x 3 -> K x 1 x 3
    center = np.expand_dims(center, 1)
    #
    delta2s = (location - center) ** 2
    widths = np.expand_dims(widths,1)
    return np.exp(-delta2s.sum(2) / (widths))

def free_energy(q, p):
    """Calculate the free-energy (negative of the evidence lower bound)"""
    return -probtorch.objectives.montecarlo.elbo(q, p)

def log_likelihood(q, p):
    """The expected log-likelihood of observed data under the proposal distribution"""
    return probtorch.objectives.montecarlo.log_like(q, p, sample_dim=0)

class TFAEncoder(nn.Module):
    """Variational guide for topographic factor analysis"""
    def __init__(self, num_times, mean_centers, mean_widths, mean_weights,
                 num_factors=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self._num_factors = num_factors

        self._weight_std_dev = torch.sqrt(torch.rand(
            (self._num_times, self._num_factors)
        ))
        self.mean_weight = Parameter(mean_weights)
        self._weight_std_dev = Parameter(self._weight_std_dev)

        self._factor_center_std_dev = torch.sqrt(torch.rand(
            (self._num_factors, 3)
        ))
        self.mean_factor_center = Parameter(mean_centers)
        self._factor_center_std_dev = Parameter(self._factor_center_std_dev)

        self._factor_log_width_std_dev = torch.sqrt(torch.rand(
            (self._num_factors)
        ))
        self.mean_factor_log_width = Parameter(mean_widths*torch.ones(self._num_factors))
        self._factor_log_width_std_dev = Parameter(self._factor_log_width_std_dev)

    def forward(self, num_samples=NUM_SAMPLES):
        q = probtorch.Trace()

        mean_weight = self.mean_weight.expand(num_samples, self._num_times,
                                              self._num_factors)
        weight_std_dev = self._weight_std_dev.expand(num_samples,
                                                     self._num_times,
                                                     self._num_factors)

        mean_factor_center = self.mean_factor_center.expand(num_samples,
                                                            self._num_factors,
                                                            3)
        factor_center_std_dev = self._factor_center_std_dev.expand(
            num_samples, self._num_factors, 3
        )

        mean_factor_log_width = self.mean_factor_log_width.expand(
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
                 num_voxels, num_factors=NUM_FACTORS, voxel_noise=VOXEL_NOISE):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self._num_factors = num_factors
        self._num_voxels = num_voxels

        self.mean_weight = Variable(torch.zeros(
            (self._num_times, self._num_factors)
        ))
        self._weight_std_dev = Variable(SOURCE_WEIGHT_STD_DEV *
                                        torch.ones((self._num_times,
                                                    self._num_factors)))

        self.mean_factor_center = Variable(
            brain_center.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        )
        self._factor_center_std_dev = Variable(
            brain_center_std_dev.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        )

        self.mean_factor_log_width = Variable(torch.ones((self._num_factors)))
        self._factor_log_width_std_dev = Variable(
            SOURCE_LOG_WIDTH_STD_DEV * torch.ones((self._num_factors))
        )

        self._voxel_noise = voxel_noise

    def cuda(self, device=None):
        super().cuda(device)
        self.mean_weight = self.mean_weight.cuda()
        self._weight_std_dev = self._weight_std_dev.cuda()

        self.mean_factor_center = self.mean_factor_center.cuda()
        self._factor_center_std_dev = self._factor_center_std_dev.cuda()

        self.mean_factor_log_width = self.mean_factor_log_width.cuda()
        self._factor_log_width_std_dev = self._factor_log_width_std_dev.cuda()

    def cpu(self):
        super().cpu()
        self.mean_weight = self.mean_weight.cpu()
        self._weight_std_dev = self._weight_std_dev.cpu()

        self.mean_factor_center = self.mean_factor_center.cpu()
        self._factor_center_std_dev = self._factor_center_std_dev.cpu()

        self.mean_factor_log_width = self.mean_factor_log_width.cpu()
        self._factor_log_width_std_dev = self._factor_log_width_std_dev.cpu()

    def forward(self, activations, locations, q=None):
        p = probtorch.Trace()

        weights = p.normal(self.mean_weight, self._weight_std_dev,
                           value=q['Weights'], name='Weights')
        factor_centers = p.normal(self.mean_factor_center,
                                  self._factor_center_std_dev,
                                  value=q['FactorCenters'],
                                  name='FactorCenters')
        factor_log_widths = p.normal(self.mean_factor_log_width,
                                     self._factor_log_width_std_dev,
                                     value=q['FactorLogWidths'],
                                     name='FactorLogWidths')
        factors = radial_basis(locations, factor_centers, factor_log_widths,
                               num_voxels=locations.shape[0])
        p.normal(torch.matmul(weights, factors), self._voxel_noise,
                 value=activations, name='Y')

        return p

class TopographicalFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_file, num_factors=NUM_FACTORS):
        self.num_factors = num_factors

        dataset = sio.loadmat(data_file)
        # pull out the voxel activations and locations
        data = dataset['data']
        R = dataset['R']
        self.voxel_activations = torch.Tensor(data).transpose(0, 1)
        self.voxel_locations = torch.Tensor(R)

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

        mean_centers_init, mean_widths_init, mean_weights_init = \
            self.get_initialization(data, R)
        mean_centers_init = torch.Tensor(mean_centers_init)
        mean_weights_init = torch.Tensor(mean_weights_init)
        self.enc = TFAEncoder(self.num_times, mean_centers_init,
                              mean_widths_init, mean_weights_init,
                              num_factors=self.num_factors)

        self.dec = TFADecoder(self.brain_center, self.brain_center_std_dev,
                              self.num_times, self.num_voxels,
                              num_factors=self.num_factors)

        if CUDA:
            self.enc = torch.nn.DataParallel(self.enc)
            self.dec = torch.nn.DataParallel(self.dec)
            self.enc.cuda()
            self.dec.cuda()

    def get_initialization(self, data, R):
        kmeans = KMeans(init='k-means++',
                        n_clusters=self.num_factors,
                        n_init=10,
                        random_state=100)
        kmeans.fit(R)
        initial_centers = kmeans.cluster_centers_
        initial_widths = 2.0 * math.pow(np.nanmax(np.std(R, axis=0)), 2)
        F = initial_radial_basis(R, initial_centers, initial_widths)
        F = F.T

        # beta = np.var(voxel_activations)
        trans_F = F.T.copy()
        initial_weights = np.linalg.solve(trans_F.dot(F), trans_F.dot(data))

        return initial_centers, np.log(initial_widths), initial_weights.T

    def hotspot_initialization(self):
        """Calculate mean image, center it, and fold it.
           Use the top K peaks as initial centers for q."""

        mean_image = torch.mean(self.voxel_activations, 0)
        mean_activation = torch.mean(mean_image)
        mean_image = mean_image - mean_activation
        mean_image = torch.abs(mean_image)

        factor_centers = []
        for k in range(self.num_factors):
            _, i = mean_image.max(0)
            mean_image[i] = 0
            factor_centers.append(self.voxel_locations[i])

        hotspots = torch.cat(factor_centers) #Kx3 tensor

        return hotspots

    def train(self, num_steps=10, learning_rate=0.1, log_level=logging.WARNING,
              batch_size=64):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        voxels_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.voxel_activations.transpose(0, 1),
                self.voxel_locations
            ),
            batch_size=batch_size
        )
        optimizer = torch.optim.Adam(list(self.enc.parameters()), lr=learning_rate)

        self.enc.train()
        self.dec.train()

        free_energies = np.zeros(num_steps)
        lls = np.zeros(num_steps)

        for n in range(num_steps):
            start = time.time()

            for (batch, (activations, locations)) in enumerate(voxels_loader):
                activations = Variable(activations.transpose(0, 1))
                locations = Variable(locations)
                if CUDA:
                    activations.cuda()
                    locations.cuda()

                optimizer.zero_grad()
                q = self.enc(num_samples=NUM_SAMPLES)
                p = self.dec(activations=activations, locations=locations, q=q)

                free_energy_n = free_energy(q, p)
                ll = log_likelihood(q, p)
                free_energy_n.backward()
                optimizer.step()

            if CUDA:
                free_energy_n = free_energy_n.cpu()
                ll = ll.cpu()
            free_energies[n] = free_energy_n.data.numpy()[0]
            lls[n] = ll.data.numpy()[0]

            end = time.time()
            msg = EPOCH_MSG % (n + 1, (end - start) * 1000, free_energy_n)
            logging.info(msg)

        self.losses = np.vstack([free_energies, lls])
        return self.losses

    def results(self):
        """Return the inferred parameters"""
        q = self.enc(num_samples=NUM_SAMPLES)

        weights = q['Weights'].value.data
        factor_centers = q['FactorCenters'].value.data
        factor_log_widths = q['FactorLogWidths'].value.data

        if CUDA:
            weights = weights.cpu()
            factor_centers = factor_centers.cpu()
            factor_log_widths = factor_log_widths.cpu()

        result = {
            'weights': weights.numpy(),
            'factor_centers': factor_centers.numpy(),
            'factor_log_widths': factor_log_widths.numpy(),
        }
        return result

    def mean_parameters(self, log_level=logging.WARNING):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        if CUDA:
            mean_factor_center = self.enc.module.mean_factor_center.data.cpu()
            mean_factor_log_width = self.enc.module.mean_factor_log_width.data.cpu()
            mean_weight = self.enc.module.mean_weight.data.cpu()
        else:
            mean_factor_center = self.enc.mean_factor_center.data
            mean_factor_log_width = self.enc.mean_factor_log_width.data
            mean_weight = self.enc.mean_weight.data

        mean_factor_center = mean_factor_center.numpy()
        mean_factor_log_width = mean_factor_log_width.numpy()
        mean_weight = mean_weight.numpy()
        mean_factors = initial_radial_basis(self.voxel_locations.numpy(),
                                            mean_factor_center,
                                            np.exp(mean_factor_log_width))

        logging.info("Mean Factor Centers: %s", str(mean_factor_center))
        logging.info("Mean Factor Log Widths: %s", str(mean_factor_log_width))
        logging.info("Mean Weights: %s", str(mean_weight))
        logging.info('Reconstruction Error (Frobenius Norm): %.8e',
                     np.linalg.norm(mean_weight @ mean_factors - self.voxel_activations.numpy()))

        mean_parameters = {
            'mean_weight': mean_weight,
            'mean_factor_center': mean_factor_center,
            'mean_factor_log_width': mean_factor_log_width
        }
        return mean_parameters


parser = argparse.ArgumentParser(description='Topographical factor analysis for fMRI data')
parser.add_argument('data_file', type=str, help='fMRI filename')
parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate for optimization')
parser.add_argument('--log-optimization', action='store_true', help='Whether to log optimization')
parser.add_argument('--factors', type=int, default=50, help='Number of latent factors')

if __name__ == '__main__':
    args = parser.parse_args()
    tfa = TopographicalFactorAnalysis(args.data_file, num_factors=args.factors)
    if args.log_optimization:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    losses = tfa.train(num_steps=args.steps, learning_rate=args.learning_rate,
                       log_level=log_level)
    if args.log_optimization:
        utils.plot_losses(losses)
    tfa.mean_parameters(log_level=log_level)
