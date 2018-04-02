"""Deep factor analysis models as ProbTorch modules"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections

import numpy as np
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data

import probtorch

from . import tfa_models
from . import utils

class DeepTFAEmbedding(tfa_models.Model):
    def __init__(self, num_factors, embedding_dim=2):
        super(tfa_models.Model, self).__init__()

        self._num_factors = num_factors
        self._embedding_dim = embedding_dim
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim, int(self._num_factors / 2)),
            torch.nn.Sigmoid()
        )
        self.weights_generator = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim, int(self._num_factors / 2)),
            torch.nn.Sigmoid(),
            torch.nn.Linear(int(self._num_factors / 2), self._num_factors)
        )
        self.factor_centers_generator = torch.nn.Linear(
            int(self._num_factors / 2), self._num_factors * 3
        )
        self.factor_log_widths_generator = torch.nn.Linear(
            int(self._num_factors / 2),
            self._num_factors
        )

    def forward(self, trace, params, guide=probtorch.Trace(), subject=0):
        weights_embedding = trace.normal(params['embedding']['weights']['mu'],
                                         params['embedding']['weights']['sigma'],
                                         value=guide['z_w' + str(subject)],
                                         name='z_w' + str(subject))
        weights = self.weights_generator(weights_embedding)

        factors_embedding = self.embedder(trace.normal(
            params['embedding']['factors']['mu'],
            params['embedding']['factors']['sigma'],
            value=guide['z_f' + str(subject)],
            name='z_f' + str(subject)
        ))
        factor_centers = self.factor_centers_generator(factors_embedding)
        centers_shape = (self._num_factors, 3)
        if len(factor_centers.shape) > 1:
            centers_shape = (-1,) + centers_shape
        factor_centers = factor_centers.view(*centers_shape)
        factor_log_widths = self.factor_log_widths_generator(factors_embedding)

        return weights, factor_centers, factor_log_widths

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_times, embedding_dim=2):
        self.num_times = num_times
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_times, self.embedding_dim),
                'sigma': torch.ones(self.num_times, self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.embedding_dim),
                'sigma': torch.ones(self.embedding_dim),
            },
        }

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_times, embedding_dim=2):
        self.num_times = num_times
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_times, self.embedding_dim),
                'sigma': torch.ones(self.num_times, self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.embedding_dim),
                'sigma': torch.ones(self.embedding_dim),
            },
        }

        super(self.__class__, self).__init__(params, guide=True)
