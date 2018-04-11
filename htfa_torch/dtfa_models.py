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
    def __init__(self, num_factors, num_times, hyper_means, embedding_dim=2):
        super(tfa_models.Model, self).__init__()

        self._num_factors = num_factors
        self._num_times = max(num_times)
        self._embedding_dim = embedding_dim
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim, int(self._num_factors / 2)),
            torch.nn.Sigmoid()
        )
        self.weights_generator = torch.nn.Sequential(
            torch.nn.Linear(self._embedding_dim + 1,
                            int(self._num_factors / 2) + 1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(int(self._num_factors / 2) + 1, self._num_factors)
        )
        t = np.random.choice(hyper_means['weights'].shape[0], 1)[0]
        self.weights_generator[2].bias = torch.nn.Parameter(
            hyper_means['weights'][t]
        )
        self.factors_generator = torch.nn.Linear(int(self._num_factors / 2),
                                                 self._num_factors * 4)
        self.factors_generator.bias = torch.nn.Parameter(
            torch.cat((hyper_means['factor_centers'],
                       hyper_means['factor_log_widths'].unsqueeze(1)),
                      1).view(self._num_factors * 4)
        )

    def forward(self, trace, params, guide=probtorch.Trace(), times=None,
                subject=0):
        if times is None:
            times_range = torch.arange(self._num_times).unsqueeze(1)
        else:
            times_range = torch.arange(times[0], times[1]).unsqueeze(1)
        weights_embedding = trace.normal(params['embedding']['weights']['mu'],
                                         params['embedding']['weights']['sigma'],
                                         value=guide['z_w' + str(subject)],
                                         name='z_w' + str(subject))

        if len(weights_embedding.shape) > 2:
            times_range = times_range.expand(weights_embedding.shape[0],
                                             *times_range.shape)
        if weights_embedding.is_cuda:
            times_range = times_range.cuda()
        weights_embedding = torch.cat(
            (weights_embedding, Variable(times_range)),
            dim=len(weights_embedding.shape) - 1
        )
        weights = self.weights_generator(weights_embedding)

        factors_embedding = self.embedder(trace.normal(
            params['embedding']['factors']['mu'],
            params['embedding']['factors']['sigma'],
            value=guide['z_f' + str(subject)],
            name='z_f' + str(subject)
        ))
        factors = self.factors_generator(factors_embedding)
        factors_shape = (self._num_factors, 4)
        if len(factors.shape) > 1:
            factors_shape = (-1,) + factors_shape
        factors = factors.view(*factors_shape)

        if len(factors.shape) > 2:
            factor_centers = factors[:, :, 0:3]
            factor_log_widths = factors[:, :, 3]
        else:
            factor_centers = factors[:, 0:3]
            factor_log_widths = factors[:, 3]

        return weights, factor_centers, factor_log_widths

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_times, embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_times = max(num_times)
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_subjects, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.num_times,
                                    self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
        }

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_times, embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_times = max(num_times)
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_subjects, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.num_times,
                                    self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
        }

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_subjects=1, num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._num_subjects = num_subjects
        self._num_times = num_times

        self.hyperparams = DeepTFAGuideHyperparams(self._num_subjects,
                                                   self._num_times,
                                                   embedding_dim)

    def forward(self, trace, embedding, times=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        weights = [s for s in range(self._num_subjects)]
        centers = [s for s in range(self._num_subjects)]
        log_widths = [s for s in range(self._num_subjects)]
        for s in range(self._num_subjects):
            subject_params = utils.vardict()
            for k, v in params.iteritems():
                if 'weights' in k and times is not None:
                    subject_params[k] = utils.unsqueeze_and_expand(
                        v[s][times[0]:times[1]], 0, num_particles, clone=True
                    )
                else:
                    subject_params[k] = utils.unsqueeze_and_expand(
                        v[s], 0, num_particles, clone=True
                    )
            weights[s], centers[s], log_widths[s] =\
                embedding(trace, subject_params, times=times, subject=s)

        return weights, centers, log_widths

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, activations,
                 num_factors=tfa_models.NUM_FACTORS, num_subjects=1,
                 num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_subjects = num_subjects
        self._num_times = num_times

        s = np.random.choice(self._num_subjects, 1)[0]
        centers, widths, weights = utils.initial_hypermeans(
            activations[s].numpy().T, locations[s].numpy(), self._num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths * torch.ones(self._num_factors),
        }

        self.embedding = DeepTFAEmbedding(self._num_factors, self._num_times,
                                          hyper_means, embedding_dim)

        self.hyperparams = DeepTFAGenerativeHyperparams(self._num_subjects,
                                                        self._num_times,
                                                        embedding_dim)

        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            self._locations[s], self._num_times[s], tfa_models.VOXEL_NOISE,
            subject=s
        ) for s in range(self._num_subjects)]
        for s, subject_likelihood in enumerate(self.likelihoods):
            self.add_module('_likelihood' + str(s), subject_likelihood)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=[]):
        params = self.hyperparams.state_vardict()
        activations = [s for s in range(self._num_subjects)]
        for s in range(self._num_subjects):
            subject_params = utils.vardict()
            if times is None:
                for k, v in params.iteritems():
                    subject_params[k] = v[s]
            else:
                for k, v in params.iteritems():
                    subject_params[k] = v[s]
                    if 'weights' in k and times is not None:
                        subject_params[k] = v[s][times[0]:times[1]]

            weights, centers, log_widths = self.embedding(trace, subject_params,
                                                          guide=guide, times=times,
                                                          subject=s)
            activations[s] = self.likelihoods[s](trace, weights, centers,
                                                 log_widths,
                                                 observations=observations[s])

        return activations
