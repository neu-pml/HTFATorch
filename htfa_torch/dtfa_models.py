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
    def __init__(self, num_blocks, num_subjects, num_tasks, num_times,
                 num_factors, embedding_dim=2):
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
                         np.sqrt(tfa_models.SOURCE_WEIGHT_STD_DEV**2 +\
                                 tfa_models.SOURCE_LOG_WIDTH_STD_DEV**2),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'block': {
                'weights': torch.zeros(self.num_blocks, self.num_times,
                                       self._num_factors)
            },
            'voxel_noise': torch.ones(self.num_blocks) * tfa_models.VOXEL_NOISE,
            'origin': {
                'mu': torch.zeros(self.embedding_dim),
                'sigma': torch.ones(self.embedding_dim) * 1e-3,
            },
            'template_factor_centers': {
                'mu': torch.zeros(self._num_factors, 3),
                'sigma': torch.ones(self._num_factors, 3) *\
                         tfa_models.SOURCE_CENTER_STD_DEV,
            },
            'template_factor_log_widths': {
                'mu': torch.zeros(self._num_factors),
                'sigma': torch.ones(self._num_factors) *\
                         tfa_models.SOURCE_LOG_WIDTH_STD_DEV,
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
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim),
            },
            'block': {
                'weights': hyper_means['weights'].expand(
                    self.num_blocks, self.num_times, self._num_factors
                )
            },
            'origin': {
                'mu': torch.zeros(self.embedding_dim),
                'sigma': torch.ones(self.embedding_dim),
            },
            'template_factor_centers': {
                'mu': hyper_means['factor_centers'],
                'sigma': torch.ones(self._num_factors, 3),
            },
            'template_factor_log_widths': {
                'mu': hyper_means['factor_log_widths'],
                'sigma': torch.ones(self._num_factors) / 3,
            }
        })

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFADecoder(nn.Module):
    """Neural network module mapping from embeddings to a topographic factor
       analysis"""
    def __init__(self, num_factors, embedding_dim=2):
        super(DeepTFADecoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_factors = num_factors

        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._num_factors),
            nn.Softsign(),
            nn.Linear(self._num_factors, self._num_factors * 2),
            nn.Softsign(),
        )
        self.centers_embedding = nn.Linear(self._num_factors * 2,
                                           self._num_factors * 3)
        self.centers_embedding.bias = nn.Parameter(
            torch.zeros(self._num_factors * 3)
        )
        self.log_widths_embedding = nn.Linear(self._num_factors * 2,
                                              self._num_factors)
        self.log_widths_embedding.bias = nn.Parameter(
            torch.zeros(self._num_factors)
        )
        self.weights_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._num_factors // 2),
            nn.Softsign(),
            nn.Linear(self._num_factors // 2, self._num_factors),
            nn.Softsign(),
            nn.Linear(self._num_factors, self._num_factors * 2),
        )
        self.weights_embedding[-1].bias = nn.Parameter(torch.cat(
            (torch.zeros(self._num_factors), torch.ones(self._num_factors)),
            dim=-1
        ))

    def predict(self, trace, params, guide, subject, task, origin):
        if subject and ('z^P_%d' % subject) not in trace:
            subject_embed = origin + trace.normal(
                params['subject']['mu'][:, subject],
                softplus(params['subject']['sigma'][:, subject]),
                value=utils.clamped('z^P_%d' % subject, guide),
                name='z^P_%d' % subject
            )
        elif subject:
            subject_embed = trace['z^P_%d' % subject].value
        else:
            subject_embed = origin
        if task and ('z^S_%d' % task) not in trace:
            task_embed = origin + trace.normal(
                params['task']['mu'][:, task],
                softplus(params['task']['sigma'][:, task]),
                value=utils.clamped('z^S_%d' % task, guide),
                name='z^S_%d' % task
            )
        elif task:
            task_embed = trace['z^S_%d' % task].value
        else:
            task_embed = origin

        factor_params = self.factors_embedding(subject_embed)

        centers_predictions = self.centers_embedding(factor_params).view(
            -1, self._num_factors, 3
        )
        log_widths_predictions = self.log_widths_embedding(factor_params).\
                                 view(-1, self._num_factors)
        joint_embed = torch.cat((subject_embed, task_embed), dim=-1)
        weight_predictions = self.weights_embedding(joint_embed).view(
            -1, self._num_factors, 2
        )

        return centers_predictions, log_widths_predictions, weight_predictions

    def forward(self, trace, blocks, block_subjects, block_tasks, params, times,
                guide=None, num_particles=tfa_models.NUM_PARTICLES,
                expand_params=False):
        params = utils.vardict(params)
        if expand_params:
            for k, v in params.items():
                params[k] = v.expand(num_particles, *v.shape)

        if 'origin' in params:
            if 'z^P_{-1}' not in trace:
                origin = trace.normal(
                    params['origin']['mu'], softplus(params['origin']['sigma']),
                    value=utils.clamped('z^P_{-1}', guide), name='z^P_{-1}',
                )
            else:
                origin = trace['z^P_{-1}'].value
        else:
            origin = torch.zeros(num_particles, self._embedding_dim)
            origin = origin.to(device=self.device)

        if 'template_factor_centers' not in trace:
            template_factor_centers = trace.normal(
                params['template_factor_centers']['mu'],
                softplus(params['template_factor_centers']['sigma']),
                value=utils.clamped('template_factor_centers', guide),
                name='template_factor_centers',
            )
        else:
            template_factor_centers = trace['template_factor_centers'].value
        if 'template_factor_log_widths' not in trace:
            template_factor_log_widths = trace.normal(
                params['template_factor_log_widths']['mu'],
                softplus(params['template_factor_log_widths']['sigma']),
                value=utils.clamped('template_factor_log_widths', guide),
                name='template_factor_log_widths',
            )
        else:
            template_factor_log_widths = trace['template_factor_log_widths'].value

        if blocks:
            weights = [None for b in blocks]
            factor_centers = [None for b in blocks]
            factor_log_widths = [None for b in blocks]

            for (i, b) in enumerate(blocks):
                subject = block_subjects[i] if b else None
                task = block_tasks[i] if b else None

                factor_centers[i], factor_log_widths[i], weight_predictions =\
                    self.predict(trace, params, guide, subject, task, origin)
                factor_centers[i] = factor_centers[i] + template_factor_centers
                factor_log_widths[i] = factor_log_widths[i] +\
                                       template_factor_log_widths

                weights_params = params['block']['weights']
                weights[i] = trace.normal(
                    weight_predictions[:, :, 0].unsqueeze(1) +
                    weights_params[:, b, times[0]:times[1], :],
                    softplus(weight_predictions[:, :, 1].unsqueeze(1)),
                    value=utils.clamped(
                        'Weights%dt%d-%d' % (b or -1, times[0], times[1]),
                        guide
                    ),
                    name='Weights%dt%d-%d' % (b or -1, times[0], times[1])
                )
        else:
            factor_centers, factor_log_widths, weights =\
                self.predict(trace, params, guide, None, None, origin)
            factor_centers = factor_centers + template_factor_centers
            factor_log_widths = factor_log_widths + template_factor_log_widths
            weights = weights[:, :, 0].unsqueeze(1).expand(
                -1, times[1]-times[0], self._num_factors
            )

        return weights, factor_centers, factor_log_widths

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

    def forward(self, decoder, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        if blocks is None:
            blocks = list(range(self._num_blocks))

        block_subjects = [self.block_subjects[b]
                          for b in range(self._num_blocks)
                          if b in blocks]
        block_tasks = [self.block_tasks[b] for b in range(self._num_blocks)
                       if b in blocks]

        return decoder(trace, blocks, block_subjects, block_tasks, params,
                       times=times, num_particles=num_particles)

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
            self._num_blocks, len(set(block_subjects)), len(set(block_tasks)),
            self._num_times, self._num_factors, embedding_dim
        )
        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times[b], block=b, register_locations=False
        ) for b in range(self._num_blocks)]
        for b, block_likelihood in enumerate(self.likelihoods):
            self.add_module('likelihood' + str(b), block_likelihood)

    def forward(self, decoder, trace, times=None, guide=probtorch.Trace(),
                observations=[], blocks=None):
        params = self.hyperparams.state_vardict()
        if times is None:
            times = (0, max(self._num_times))
        if blocks is None:
            blocks = list(range(self._num_blocks))

        block_subjects = [self.block_subjects[b]
                          for b in range(self._num_blocks)
                          if b in blocks]
        block_tasks = [self.block_tasks[b] for b in range(self._num_blocks)
                       if b in blocks]

        weights, centers, log_widths = decoder(trace, blocks, block_subjects,
                                               block_tasks, params, times,
                                               guide=guide,
                                               num_particles=1,
                                               expand_params=True)

        return [self.likelihoods[b](trace, weights[i], centers[i],
                                    log_widths[i], params, times=times,
                                    observations=observations[i])
                for (i, b) in enumerate(blocks)]
