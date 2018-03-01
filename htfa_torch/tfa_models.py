"""Topographic factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections

import flatdict
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

class Model(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, *args, trace=probtorch.Trace()):
        pass

class HyperPrior(Model):
    def __init__(self, vs, guide=True):
        super(Model, self).__init__()

        self._guide = guide
        for (k, v) in vs.items():
            if self._guide:
                self.register_parameter(k, Parameter(v))
            else:
                self.register_buffer(k, Variable(v))

    def forward(self):
        return self.state_dict(keep_vars=True)

class GuidePrior(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, num_samples=NUM_SAMPLES):
        pass

class GenerativePrior(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, guide=probtorch.Trace()):
        pass

class GenerativeLikelihood(Model):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, trace, *args, observations=collections.defaultdict()):
        pass

class TFAGuideHyperPrior(HyperPrior):
    def __init__(self, means, num_times, num_factors=NUM_FACTORS):
        self._num_times = num_times
        self._num_factors = num_factors

        params = flatdict.FlatDict(delimiter='_')
        params['weights'] = {
            'mu': means['weights'],
            'sigma': torch.sqrt(torch.rand(
                (self._num_times, self._num_factors)
            ))
        }
        params['factor_centers'] = {
            'mu': means['factor_centers'],
            'sigma': torch.sqrt(torch.rand((self._num_factors, 3)))
        }
        params['factor_log_widths'] = {
            'mu': means['factor_log_widths'] * torch.ones(self._num_factors),
            'sigma': torch.sqrt(torch.rand((self._num_factors)))
        }
        super(self.__class__, self).__init__(params, guide=True)

    def forward(self):
        state_dict = super(self.__class__, self).forward()

        return {
            'weights': {
                'mu': state_dict['weights_mu'],
                'sigma': state_dict['weights_sigma']
            },
            'factor_centers': {
                'mu': state_dict['factor_centers_mu'],
                'sigma': state_dict['factor_centers_sigma']
            },
            'factor_log_widths': {
                'mu': state_dict['factor_log_widths_mu'],
                'sigma': state_dict['factor_log_widths_sigma']
            }
        }

class TFAGuide(nn.Module):
    """Variational guide for topographic factor analysis"""
    def __init__(self, num_times, num_factors=NUM_FACTORS, hyper_means=None):
        super(self.__class__, self).__init__()
        self._num_times = num_times
        self._num_factors = num_factors

        if hyper_means:
            self.weight_params = {
                'mu': Parameter(hyper_means['weights']),
                'sigma': Parameter(torch.sqrt(torch.rand(
                    (self._num_times, self._num_factors)
                )))
            }
            self._weight_params = nn.ParameterList(
                list(self.weight_params.values())
            )
            self.factor_center_params = {
                'mu': Parameter(hyper_means['factor_centers']),
                'sigma': Parameter(torch.sqrt(torch.rand(
                    (self._num_factors, 3)
                )))
            }
            self._factor_center_params = nn.ParameterList(
                list(self.factor_center_params.values())
            )
            self.factor_log_width_params = {
                'mu': Parameter(hyper_means['factor_log_widths'] *
                                torch.ones(self._num_factors)),
                'sigma': Parameter(torch.sqrt(torch.rand(
                    (self._num_factors)
                )))
            }
            self._factor_log_width_params = nn.ParameterList(
                list(self.factor_log_width_params.values())
            )

    def forward(self, weight_params=None, factor_center_params=None,
                factor_log_width_params=None, num_samples=NUM_SAMPLES,
                times=None):
        q = probtorch.Trace()

        if times is None:
            times = (0, self._num_times)

        if weight_params is None:
            weight_params = {
                'mu': self.weight_params['mu'][times[0]:times[1], :],
                'sigma': self.weight_params['sigma'][times[0]:times[1], :],
            }
        weight_params['mu'] = weight_params['mu'].expand(
            num_samples, times[1] - times[0], self._num_factors
        )
        weight_params['sigma'] = weight_params['sigma'].expand(
            num_samples, times[1] - times[0], self._num_factors
        )

        if factor_center_params is None:
            factor_center_params = self.factor_center_params
        factor_center_params['mu'] = factor_center_params['mu'].expand(
            num_samples, self._num_factors, 3
        )
        factor_center_params['sigma'] = factor_center_params['sigma'].expand(
            num_samples, self._num_factors, 3
        )

        if factor_log_width_params is None:
            factor_log_width_params = self.factor_log_width_params
        factor_log_width_params['mu'] = factor_log_width_params['mu'].expand(
            num_samples, self._num_factors
        )
        factor_log_width_params['sigma'] =\
            factor_log_width_params['sigma'].expand(num_samples,
                                                    self._num_factors)

        q.normal(weight_params['mu'], weight_params['sigma'], name='Weights')

        q.normal(factor_center_params['mu'], factor_center_params['sigma'],
                 name='FactorCenters')
        q.normal(factor_log_width_params['mu'],
                 factor_log_width_params['sigma'], name='FactorLogWidths')

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

    def forward(self, activations, locations, q=None, times=None):
        p = probtorch.Trace()

        mean_weight = self.mean_weight
        weight_std_dev = self._weight_std_dev
        if times is not None:
            mean_weight = mean_weight[times[0]:times[1], :]
            weight_std_dev = weight_std_dev[times[0]:times[1], :]

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
