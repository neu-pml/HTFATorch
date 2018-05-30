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
    def __init__(self, num_subjects, num_tasks, num_times, num_factors,
                 embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'factors': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         tfa_models.SOURCE_LOG_WIDTH_STD_DEV,
            },
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
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
            'factors': {
                'mu': torch.zeros(self.num_blocks, self.embedding_dim),
                'sigma': torch.ones(self.num_blocks, self.embedding_dim),
            },
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim),
            },
            'template': {
                'factor_centers': {
                    'mu': hyper_means['factor_centers'],
                    'sigma': torch.ones(self._num_factors, 3),
                },
                'factor_log_widths': {
                    'mu': hyper_means['factor_log_widths'] *\
                          torch.ones(self._num_factors),
                    'sigma': torch.ones(self._num_factors) *\
                             tfa_models.SOURCE_LOG_WIDTH_STD_DEV
                }
            },
            'block': {
                'weights': {
                    'mu': hyper_means['weights'].mean(0).unsqueeze(0).expand(
                        self.num_blocks, self.num_times, self._num_factors
                    ),
                    'sigma': torch.ones(self.num_blocks, self.num_times,
                                        self._num_factors),
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

        self.htfa_template = htfa_models.HTFAGuideTemplatePrior()
        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   hyper_means,
                                                   embedding_dim)
        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._num_factors),
            nn.Tanhshrink(),
            nn.Linear(self._num_factors, self._num_factors * 8),
        )
        self.weights_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._num_factors),
            nn.Tanhshrink(),
            nn.Linear(self._num_factors, self._num_factors * 2),
        )
        self.softplus = nn.Softplus()

        self.epsilon = nn.Parameter(torch.Tensor([tfa_models.VOXEL_NOISE]))

        if hyper_means is not None:
            self.weights_embedding[-1].bias = nn.Parameter(torch.cat(
                (hyper_means['weights'].mean(0), torch.ones(self._num_factors)),
                dim=0
            ))
            self.factors_embedding[-1].bias = nn.Parameter(torch.cat(
                (hyper_means['factor_centers'],
                 torch.ones(self._num_factors, 3),
                 torch.ones(self._num_factors, 1) *
                 hyper_means['factor_log_widths'],
                 torch.sqrt(torch.rand(self._num_factors, 1))),
                dim=1,
            ).view(self._num_factors * 8))

    def forward(self, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        self.htfa_template(trace, params, num_particles=num_particles)
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        if blocks is None:
            blocks = list(range(self._num_blocks))

        weights = [None for b in blocks]
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
                factors_embed = trace.normal(
                    params['factors']['mu'][:, subject, :],
                    params['factors']['sigma'][:, subject, :],
                    name='z^F_%d' % subject
                )
            if ('z^P_%d' % subject) not in trace:
                subject_embed = trace.normal(
                    params['subject']['mu'][:, subject, :],
                    params['subject']['sigma'][:, subject, :],
                    name='z^P_%d' % subject
                )
            if ('z^S_%d' % task) not in trace:
                task_embed = trace.normal(params['task']['mu'][:, task],
                                          params['task']['sigma'][:, task],
                                          name='z^S_%d' % task)

            factor_params = self.factors_embedding(factors_embed)
            factor_params = factor_params.view(-1, self._num_factors, 8)
            weights_embed = torch.cat((subject_embed, task_embed), dim=-1)
            weight_params = self.weights_embedding(weights_embed).view(
                -1, self._num_factors, 2
            )

            trace.normal(weight_params[:, :, 0], self.epsilon[0],
                         name='mu^W_%d' % b)
            trace.normal(self.softplus(weight_params[:, :, 1]), self.epsilon[0],
                         name='sigma^W_%d' % b)
            weights[i] = trace.normal(
                params['block']['weights']['mu'][:, b, ts[0]:ts[1], :],
                params['block']['weights']['sigma'][:, b, ts[0]:ts[1], :],
                name='Weights%dt%d-%d' % (b, ts[0], ts[1])
            )
            factor_centers[i] = trace.normal(
                factor_params[:, :, 0:3],
                self.softplus(factor_params[:, :, 3:6]),
                name='FactorCenters%d' % b
            )
            factor_log_widths[i] = trace.normal(
                factor_params[:, :, 6].contiguous().view(
                    -1, self._num_factors
                ),
                self.softplus(factor_params[:, :, 7].contiguous().view(
                    -1, self._num_factors
                )), name='FactorLogWidths%d' % b
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

            if ('z^F_%d' % subject) not in trace:
                trace.normal(params['factors']['mu'][subject],
                             params['factors']['sigma'][subject],
                             value=guide['z^F_%d' % subject],
                             name='z^F_%d' % subject)
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
                               observations=observations,
                               weights_params=weight_params)
