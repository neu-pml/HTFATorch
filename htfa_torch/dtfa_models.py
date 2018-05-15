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

class DeepTFAEmbedding(tfa_models.Model):
    def __init__(self, num_factors, num_times, block_subjects, block_tasks,
                 brain_center, brain_center_std_dev, embedding_dim=2,
                 hyper_means=None):
        super(tfa_models.Model, self).__init__()

        self._num_factors = num_factors
        self._num_times = max(num_times)
        self._embedding_dim = embedding_dim

        self.subjects = block_subjects
        self.tasks = block_tasks

        self.weights_generator = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._num_factors),
            nn.Sigmoid(),
            nn.Linear(self._num_factors, self._num_factors * 2),
        )
        self.factors_generator = nn.Sequential(
            nn.Linear(self._embedding_dim, self._num_factors),
            nn.Sigmoid(),
            nn.Linear(self._num_factors, self._num_factors * 4),
        )
        self.softplus = nn.Softplus()

        if hyper_means is not None:
            hyper_means['weights'] = hyper_means['weights'].mean(0)
            hyper_means['factor_log_widths'] =\
                torch.Tensor([hyper_means['factor_log_widths']]).\
                expand(self._num_factors, 1)
            self.weights_generator[2].bias = nn.Parameter(torch.cat(
                (hyper_means['weights'], torch.ones(self._num_factors)),
                dim=-1
            ).view(self._num_factors * 2))
            self.factors_generator[2].bias = nn.Parameter(torch.cat(
                (hyper_means['factor_centers'],
                 hyper_means['factor_log_widths']),
                dim=-1
            ).view(self._num_factors * 4))

        self.prior = utils.vardict({
            'centers': {
                'mu': brain_center,
                'sigma': brain_center_std_dev,
            },
            'log_widths': {
                'mu': torch.ones(self._num_factors),
                'sigma': torch.ones(self._num_factors) *\
                         tfa_models.SOURCE_LOG_WIDTH_STD_DEV,
            },
        })

        utils.register_vardict(self.prior, self, parameter=False)
        self.prior = utils.vardict(self.state_dict(keep_vars=True))

    # Assumes that all tensors have a particle dimension as their first
    def forward(self, trace, params, guide=probtorch.Trace(), times=None,
                block=0, particles=False):
        if times is None:
            times = (0, self._num_times)

        subject = self.subjects[block]
        task = self.tasks[block]
        subject_params = {
            'mu': params['subject']['mu'][:, subject],
            'sigma': params['subject']['sigma'][:, subject],
        }
        task_params = {
            'mu': params['task']['mu'][:, task, times[0]:times[1]],
            'sigma': params['task']['sigma'][:, task, times[0]:times[1]],
        }
        factor_params = {
            'mu': params['factors']['mu'][:, subject],
            'sigma': params['factors']['sigma'][:, subject],
        }

        if ('z^P_%d' % subject) not in trace:
            subject_embed = trace.normal(subject_params['mu'],
                                         self.softplus(subject_params['sigma']),
                                         value=guide['z^P_%d' % subject],
                                         name='z^P_%d' % subject)
        else:
            subject_embed = trace['z^P_%d' % subject].value
        if ('z^S_%d' % task) not in trace:
            task_embed = trace.normal(task_params['mu'],
                                      self.softplus(task_params['sigma']),
                                      value=guide['z^S_%d' % task],
                                      name='z^S_%d' % task)
        else:
            task_embed = trace['z^S_%d' % task].value
        if ('z^F_%d' % subject) not in trace:
            factors_embed = trace.normal(factor_params['mu'],
                                         self.softplus(factor_params['sigma']),
                                         value=guide['z^F_%d' % subject],
                                         name='z^F_%d' % subject)
        else:
            factors_embed = trace['z^F_%d' % subject].value

        weights_embed = torch.cat(
            (subject_embed.unsqueeze(1).expand(
                subject_embed.shape[0], times[1] - times[0],
                *subject_embed.shape[1:]),
             task_embed),
            dim=-1)

        weight_params = self.weights_generator(weights_embed)
        weight_params = weight_params.view(-1, weight_params.shape[1], 2,
                                           self._num_factors)
        weights = trace.normal(weight_params[:, :, 0],
                               self.softplus(weight_params[:, :, 1]),
                               value=guide['W_%d' % block],
                               name='W_%d' % block)

        factors = self.factors_generator(factors_embed)
        factors = factors.view(-1, self._num_factors, 4)

        factor_centers = factors[:, :, 0:3]
        factor_log_widths = factors[:, :, 3]

        if not particles:
            weights = weights[0]
            factor_centers = factor_centers[0]
            factor_log_widths = factor_log_widths[0]

        prior = utils.vardict(self.state_dict(keep_vars=True))

        for k, v in prior['centers'].items():
            prior['centers'][k] = Variable(v.expand(factor_centers.shape[0], *v.shape))
        #trace.normal(prior['centers']['mu'],
        #             prior['centers']['sigma'],
        #             value=factor_centers,
        #             name='factor_centers%d' % block)
        for k, v in prior['log_widths'].items():
            prior['log_widths'][k] = Variable(v.expand(factor_log_widths.shape[0],
                                                       *v.shape))
        #trace.normal(prior['log_widths']['mu'],
        #             prior['log_widths']['sigma'],
        #             value=factor_log_widths,
        #             name='factor_log_widths%d' % block)

        return weights, factor_centers, factor_log_widths

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
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
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.num_times,
                                    self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
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
            }
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
