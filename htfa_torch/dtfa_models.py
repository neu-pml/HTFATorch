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

from . import htfa_models
from . import tfa_models
from . import utils

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_tasks, num_times, embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'factors': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.num_times,
                                    self.embedding_dim),
            }
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, embedding_dim=2):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'factors': {
                'mu': torch.zeros(self.num_blocks, self.embedding_dim),
                'sigma': torch.ones(self.num_blocks, self.embedding_dim),
            },
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.num_times,
                                    self.embedding_dim),
            },
            'template': {
                'centers': {
                    'mu': torch.zeros(self._num_factors, 3),
                    'sigma': torch.ones(self._num_factors, 3),
                },
                'log_widths': {
                    'mu': torch.ones(self._num_factors),
                    'sigma': torch.ones(self._num_factors) *\
                             tfa_models.SOURCE_LOG_WIDTH_STD_DEV
                }
            },
        })

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, num_subjects, num_tasks, num_blocks=1,
                 num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   embedding_dim)

    def forward(self, trace, embedding, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)

        if blocks is None:
            blocks = list(range(self._num_blocks))
        weights = [b for b in blocks]
        centers = [b for b in blocks]
        log_widths = [b for b in blocks]
        for (i, b) in enumerate(blocks):
            weights[i], centers[i], log_widths[i] =\
                embedding(trace, params, times=times, block=b,
                          particles=True)

        return weights, centers, log_widths

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects, block_tasks,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2, hyper_means=None):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times

        center, center_std_dev = utils.brain_centroid(self._locations)
        self.embedding = DeepTFAEmbedding(self._num_factors, self._num_times,
                                          block_subjects, block_tasks,
                                          center, center_std_dev, embedding_dim,
                                          hyper_means)

        self.hyperparams = DeepTFAGenerativeHyperparams(
            self._num_blocks, self._num_times, self._num_factors,
            len(set(block_subjects)), len(set(block_tasks)),
            embedding_dim
        )

        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            self._locations, self._num_times[b], tfa_models.VOXEL_NOISE,
            block=b, register_locations=False
        ) for b in range(self._num_blocks)]
        for b, block_likelihood in enumerate(self.likelihoods):
            self.add_module('_likelihood' + str(b), block_likelihood)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=[], blocks=None):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(1, *v.shape)
        if blocks is None:
            blocks = list(range(self._num_blocks))
        activations = [b for b in blocks]

        for (i, b) in enumerate(blocks):
            weights, centers, log_widths = self.embedding(
                trace, params, guide=guide, times=times,
                block=b, particles=True
            )
            activations[i] = self.likelihoods[b](
                trace, weights, centers, log_widths,
                observations=observations[i]
            )

        return activations
