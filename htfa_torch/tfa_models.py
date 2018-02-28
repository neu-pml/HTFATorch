"""Topographic factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import numpy as np
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import probtorch

NUM_FACTORS = 5
NUM_SAMPLES = 10
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

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

    def forward(self, num_samples=NUM_SAMPLES, trs=None):
        q = probtorch.Trace()

        if trs is None:
            trs = (0, self._num_times)
        mean_weight = self.mean_weight[trs[0]:trs[1], :]
        mean_weight = mean_weight.expand(num_samples, trs[1] - trs[0],
                                         self._num_factors)
        weight_std_dev = self._weight_std_dev[trs[0]:trs[1], :]
        weight_std_dev = weight_std_dev.expand(num_samples,
                                               trs[1] - trs[0],
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

        self.register_buffer('mean_weight', Variable(torch.zeros(
            (self._num_times, self._num_factors)
        )))
        self.register_buffer('_weight_std_dev', Variable(
            SOURCE_WEIGHT_STD_DEV *  torch.ones(
                (self._num_times, self._num_factors)
            )
        ))

        self.register_buffer('mean_factor_center', Variable(
            brain_center.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        ))
        self.register_buffer('_factor_center_std_dev', Variable(
            brain_center_std_dev.expand(self._num_factors, 3) *
            torch.ones((self._num_factors, 3))
        ))

        self.register_buffer('mean_factor_log_width',
                             Variable(torch.ones((self._num_factors))))
        self.register_buffer('_factor_log_width_std_dev', Variable(
            SOURCE_LOG_WIDTH_STD_DEV * torch.ones((self._num_factors))
        ))

        self._voxel_noise = voxel_noise

    def forward(self, activations, locations, q=None, trs=None):
        p = probtorch.Trace()

        mean_weight = self.mean_weight
        weight_std_dev = self._weight_std_dev
        if trs is not None:
            mean_weight = mean_weight[trs[0]:trs[1], :]
            weight_std_dev = weight_std_dev[trs[0]:trs[1], :]

        weights = p.normal(mean_weight, weight_std_dev,
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
