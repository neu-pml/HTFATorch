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
                                    self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'template': {
                'factor_centers': {
                    'mu': torch.zeros(self._num_factors, 3),
                    'sigma': torch.ones(self._num_factors, 3),
                },
                'factor_log_widths': {
                    'mu': torch.ones(self._num_factors),
                    'sigma': torch.ones(self._num_factors) *\
                             tfa_models.SOURCE_LOG_WIDTH_STD_DEV
                }
            },
        })

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, block_tasks, num_blocks=1,
                 num_times=[1], embedding_dim=2, hyper_means=None):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors
        self._embedding_dim = embedding_dim

        self.block_subjects = block_subjects
        self.block_tasks = block_tasks
        num_subjects = len(set(self.block_subjects))
        num_tasks = len(set(self.block_tasks))

        self.htfa_template = htfa_models.HTFAGuideTemplatePrior()
        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   embedding_dim)
        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._num_factors),
            nn.Tanh(),
            nn.Linear(self._num_factors, self._num_factors * 4),
        )
        self.weights_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._num_factors),
            nn.Tanhshrink(),
            nn.Linear(self._num_factors, self._num_factors),
        )

        self.epsilon = nn.Parameter(torch.Tensor([tfa_models.VOXEL_NOISE]))

        if hyper_means is not None:
            self.weights_embedding[-1].bias =\
                nn.Parameter(hyper_means['weights'][0])
            self.factors_embedding[-1].bias = nn.Parameter(torch.cat(
                (hyper_means['factor_centers'],
                 torch.Tensor([hyper_means['factor_log_widths']]).\
                 expand(self._num_factors, 1)),
                dim=1,
            ).view(self._num_factors * 4))

    def forward(self, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        self.htfa_template(trace, params, num_particles=num_particles)
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        if blocks is None:
            blocks = list(range(self._num_blocks))

        factor_weights = [None for b in blocks]
        factor_centers = [None for b in blocks]
        factor_log_widths = [None for b in blocks]

        for (i, b) in enumerate(blocks):
            subject = self.block_subjects[b]
            task = self.block_tasks[b]
            if times is None:
                ts = (0, self._num_times[b])
            else:
                ts = times

            if ('z^F_%d' % subject) not in trace:
                factors_embed = trace.normal(params['factors']['mu'][:, subject, :],
                                             params['factors']['sigma'][:, subject, :],
                                             name='z^F_%d' % subject)
            if ('z^P_%d' % subject) not in trace:
                subject_embed = trace.normal(params['subject']['mu'][:, subject, :],
                                             params['subject']['sigma'][:, subject, :],
                                             name='z^P_%d' % subject)
            if ('z^S_%dt%d_%d' % (task, ts[0], ts[1])) not in trace:
                task_embed = trace.normal(
                    params['task']['mu'][:, task, ts[0]:ts[1], :],
                    params['task']['sigma'][:, task, ts[0]:ts[1], :],
                    name='z^S_%dt%d_%d' % (task, ts[0], ts[1])
                )

            factor_params = self.factors_embedding(factors_embed)
            factor_params = factor_params.view(-1, self._num_factors, 4)
            subject_embed = utils.unsqueeze_and_expand(subject_embed, 1,
                                                       ts[1]-ts[0], True)
            weights_embed = torch.cat((subject_embed, task_embed), dim=-1)
            weights = self.weights_embedding(weights_embed)

            factor_weights[i] = trace.normal(
                weights, self.epsilon[0],
                name='Weights%dt%d-%d' % (b, times[0], times[1])
            )
            factor_centers[i] = trace.normal(factor_params[:, :, 0:3],
                                             self.epsilon[0],
                                             name='FactorCenters%d' % b)
            factor_log_widths[i] = trace.normal(
                factor_params[:, :, 3:].contiguous().view(-1, self._num_factors),
                self.epsilon[0], name='FactorLogWidths%d' % b
            )

        return factor_weights, factor_centers, factor_log_widths

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
