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

    def forward(self, trace, params, guide=probtorch.Trace(), times=None,
                block=0):
        if times is None:
            times_range = torch.arange(self._num_times).unsqueeze(1)
        else:
            times_range = torch.arange(times[0], times[1]).unsqueeze(1)
        weights_embedding = trace.normal(params['embedding']['weights']['mu'],
                                         params['embedding']['weights']['sigma'],
                                         value=guide['z_w' + str(block)],
                                         name='z_w' + str(block))

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
            value=guide['z_f' + str(block)],
            name='z_f' + str(block)
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
    def __init__(self, brain_center, brain_center_std_dev, num_blocks,
                 num_times, num_factors, embedding_dim=2):
        self.num_blocks = num_blocks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_blocks, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_blocks, self.num_times,
                                    self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.num_blocks, self.embedding_dim),
                'sigma': torch.sqrt(torch.rand(self.num_blocks, self.embedding_dim)),
            },
        }
        params['template'] = {
            'factor_centers': {
                'mu': brain_center.expand(self._num_factors, 3),
                'sigma': brain_center_std_dev.expand(self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': torch.ones(self._num_factors),
                'sigma': tfa_models.SOURCE_LOG_WIDTH_STD_DEV *\
                         torch.ones(self._num_factors),
            },
        }

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, hyper_means, num_blocks, num_times, num_factors,
                 embedding_dim=2):
        self.num_blocks = num_blocks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict()
        params['embedding'] = {
            'weights': {
                'mu': torch.zeros(self.num_blocks, self.num_times,
                                  self.embedding_dim),
                'sigma': torch.ones(self.num_blocks, self.num_times,
                                    self.embedding_dim),
            },
            'factors': {
                'mu': torch.zeros(self.num_blocks, self.embedding_dim),
                'sigma': torch.ones(self.num_blocks, self.embedding_dim),
            },
        }
        params['template'] = utils.vardict(htfa_models.TEMPLATE_SHAPE)
        params['template']['factor_centers'] =\
            utils.gaussian_populator(self._num_factors, 3)
        params['template']['factor_centers']['mu'] +=\
            hyper_means['factor_centers']
        params['template']['factor_log_widths'] = {
            'mu': hyper_means['factor_log_widths'] * torch.ones(self._num_factors),
            'sigma': torch.sqrt(torch.rand(self._num_factors)),
        }

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, hyper_means, num_factors, num_blocks=1, num_times=[1],
                 embedding_dim=2):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors

        self.hyperparams = DeepTFAGuideHyperparams(hyper_means,
                                                   self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   embedding_dim)
        self.template = htfa_models.HTFAGuideTemplatePrior()

    def forward(self, trace, embedding, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()

        template = self.template(trace, params, num_particles=num_particles)

        if blocks is None:
            blocks = list(range(self._num_blocks))
        weights = [b for b in blocks]
        centers = [b for b in blocks]
        log_widths = [b for b in blocks]
        for (i, b) in enumerate(blocks):
            block_params = utils.vardict()
            for k, v in [(k, v) for k, v in params.iteritems() if 'embedding' in k]:
                if 'weights' in k and times is not None:
                    block_params[k] = utils.unsqueeze_and_expand(
                        v[b][times[0]:times[1]], 0, num_particles, clone=True
                    )
                else:
                    block_params[k] = utils.unsqueeze_and_expand(
                        v[b], 0, num_particles, clone=True
                    )
            weights[i], centers[i], log_widths[i] =\
                embedding(trace, block_params, times=times, block=b)
            centers[i] = centers[i] + template['factor_centers']
            log_widths[i] = torch.log(torch.exp(log_widths[i]) +
                                      torch.exp(template['factor_log_widths']))

        return weights, centers, log_widths

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, hyper_means,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times

        b = np.random.choice(self._num_blocks, 1)[0]
        center, center_sigma = utils.brain_centroid(self._locations[b])

        self.embedding = DeepTFAEmbedding(self._num_factors, self._num_times,
                                          hyper_means, embedding_dim)

        self.hyperparams = DeepTFAGenerativeHyperparams(center, center_sigma,
                                                        self._num_blocks,
                                                        self._num_times,
                                                        self._num_factors,
                                                        embedding_dim)
        self.template = htfa_models.HTFAGenerativeTemplatePrior()

        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            self._locations[b], self._num_times[b], tfa_models.VOXEL_NOISE,
            block=b, register_locations=False
        ) for b in range(self._num_blocks)]
        for b, block_likelihood in enumerate(self.likelihoods):
            self.add_module('_likelihood' + str(b), block_likelihood)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=[], blocks=None):
        params = self.hyperparams.state_vardict()
        if blocks is None:
            blocks = list(range(self._num_blocks))
        activations = [b for b in blocks]

        template = self.template(trace, params, guide=guide)

        for (i, b) in enumerate(blocks):
            block_params = utils.vardict()
            if times is None:
                for k, v in [(k, v) for k, v in params.iteritems() if 'embedding' in k]:
                    block_params[k] = v[b]
            else:
                for k, v in [(k, v) for k, v in params.iteritems() if 'embedding' in k]:
                    block_params[k] = v[b]
                    if 'weights' in k and times is not None:
                        block_params[k] = v[b][times[0]:times[1]]

            weights, center_resids, log_width_resids = self.embedding(
                trace, block_params, guide=guide, times=times, block=b
            )
            activations[i] = self.likelihoods[b](
                trace, weights,
                template['factor_centers'] + center_resids,
                torch.log(torch.exp(template['factor_log_widths']) + torch.exp(log_width_resids)),
                observations=observations[i]
            )

        return activations
