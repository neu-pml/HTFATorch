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
from torch.nn.functional import softplus
import torch.utils.data

import probtorch

from . import htfa_models
from . import tfa_models
from . import utils

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_tasks, num_times, num_factors,
                 embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV *\
                         tfa_models.SOURCE_LOG_WIDTH_STD_DEV,
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'template': {
                'weights': {
                    'mu': {
                        'mu': torch.randn(self._num_factors),
                        'sigma': torch.rand(self._num_factors),
                    },
                    'sigma': {
                        'mu': torch.ones(self._num_factors) *\
                              tfa_models.SOURCE_WEIGHT_STD_DEV,
                        'sigma': torch.rand(self._num_factors),
                    }
                }
            }
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, hyper_means, embedding_dim=2):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV *\
                         tfa_models.SOURCE_LOG_WIDTH_STD_DEV,
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'block': {
                'weights': {
                    'mu': hyper_means['weights'].mean(0).unsqueeze(0).expand(
                        self.num_blocks, self.num_times, self._num_factors
                    ),
                    'sigma': torch.sqrt(torch.rand(self.num_blocks,
                                                   self.num_times,
                                                   self._num_factors)),
                }
            }
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

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   hyper_means,
                                                   embedding_dim)
        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._num_factors),
            nn.Softsign(),
        )
        self.centers_embedding = nn.Linear(self._num_factors,
                                           self._num_factors * 3)
        self.log_widths_embedding = nn.Linear(self._num_factors,
                                              self._num_factors)
        self.weights_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._num_factors),
            nn.Softsign(),
            nn.Linear(self._num_factors, self._num_factors * 2),
        )

        self.epsilon = nn.Parameter(torch.Tensor([tfa_models.VOXEL_NOISE]))
        self.register_buffer('origin', torch.zeros(self._embedding_dim))

        if hyper_means is not None:
            self.centers_embedding.bias = nn.Parameter(
                hyper_means['factor_centers'].view(self._num_factors * 3)
            )
            self.log_widths_embedding.bias = nn.Parameter(
                torch.ones(self._num_factors) *
                hyper_means['factor_log_widths'] / 2
            )

    def forward(self, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        origin = self.origin.expand(num_particles, self._embedding_dim)
        if blocks is None:
            blocks = list(range(self._num_blocks))

        weights = [None for b in blocks]
        factor_centers = [None for b in blocks]
        factor_log_widths = [None for b in blocks]

        template_params = self.factors_embedding(origin)
        template_centers = self.centers_embedding(template_params).view(
            -1, self._num_factors, 3
        )
        trace.normal(template_centers, softplus(self.epsilon)[0],
                     name='template_factor_centers')
        template_log_widths = self.log_widths_embedding(template_params).\
                              view(-1, self._num_factors)
        trace.normal(template_log_widths, softplus(self.epsilon)[0],
                     name='template_factor_log_widths')

        for (i, b) in enumerate(blocks):
            subject = self.block_subjects[b]
            task = self.block_tasks[b]
            if times is None:
                ts = (0, self._num_times[b])
            else:
                ts = times

            if ('z^P_%d' % subject) not in trace:
                subject_embed = trace.normal(
                    params['subject']['mu'][:, subject, :],
                    softplus(params['subject']['sigma'][:, subject, :]),
                    name='z^P_%d' % subject
                )
            if ('z^S_%d' % task) not in trace:
                task_embed = trace.normal(
                    params['task']['mu'][:, task],
                    softplus(params['task']['sigma'][:, task]),
                    name='z^S_%d' % task
                )

            factor_params = self.factors_embedding(subject_embed)
            centers_predictions = self.centers_embedding(factor_params).view(
                -1, self._num_factors, 3
            )
            log_widths_predictions = self.log_widths_embedding(factor_params).\
                                     view(-1, self._num_factors)
            weights_embed = torch.cat((subject_embed, task_embed), dim=-1)
            weight_predictions = self.weights_embedding(weights_embed).view(
                -1, self._num_factors, 2
            )

            weights_mu = trace.normal(weight_predictions[:, :, 0],
                                      softplus(self.epsilon)[0],
                                      name='mu^W_%d' % b)
            weights_sigma = trace.normal(weight_predictions[:, :, 1],
                                         softplus(self.epsilon)[0],
                                         name='sigma^W_%d' % b)
            weights_params = params['block']['weights']
            weights[i] = trace.normal(
                weights_params['mu'][:, b, ts[0]:ts[1], :] +
                weights_mu.unsqueeze(1),
                softplus(weights_params['sigma'][:, b, ts[0]:ts[1], :] +
                         weights_sigma.unsqueeze(1)),
                name='Weights%dt%d-%d' % (b, ts[0], ts[1])
            )
            factor_centers[i] = trace.normal(
                centers_predictions,
                softplus(self.epsilon)[0],
                name='FactorCenters%d' % b
            )
            factor_log_widths[i] = trace.normal(
                log_widths_predictions,
                softplus(self.epsilon)[0], name='FactorLogWidths%d' % b
            )

        return weights, factor_centers, factor_log_widths

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects, block_tasks,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times
        self.block_subjects = block_subjects
        self.block_tasks = block_tasks

        self.hyperparams = DeepTFAGenerativeHyperparams(
            len(set(block_subjects)), len(set(block_tasks)), self._num_times,
            self._num_factors, embedding_dim
        )
        self.htfa_model = htfa_models.HTFAModel(locations, self._num_blocks,
                                                self._num_times,
                                                self._num_factors, volume=True)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=[], blocks=None):
        params = self.hyperparams.state_vardict()
        if blocks is None:
            blocks = list(range(self._num_blocks))

        weight_params = [b for b in blocks]
        for (i, b) in enumerate(blocks):
            subject = self.block_subjects[b]
            task = self.block_tasks[b]
            if times is None:
                ts = (0, self._num_times[b])
            else:
                ts = times

            if ('z^P_%d' % subject) not in trace:
                trace.normal(params['subject']['mu'][subject],
                             params['subject']['sigma'][subject],
                             value=guide['z^P_%d' % subject],
                             name='z^P_%d' % subject)
            if ('z^S_%d' % task) not in trace:
                trace.normal(params['task']['mu'][task],
                             params['task']['sigma'][task],
                             value=guide['z^S_%d' % task], name='z^S_%d' % task)

            weight_params[i] = {
                'mu': trace.normal(params['template']['weights']['mu']['mu'],
                                   params['template']['weights']['mu']['sigma'],
                                   value=guide['mu^W_%d' % b],
                                   name='mu^W_%d' % b),
                'sigma': trace.normal(
                    params['template']['weights']['sigma']['mu'],
                    params['template']['weights']['sigma']['sigma'],
                    value=guide['sigma^W_%d' % b], name='sigma^W_%d' % b
                ),
            }

        return self.htfa_model(trace, times, guide, blocks=blocks,
                               observations=observations)
