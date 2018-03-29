"""Hierarchical factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
import numpy as np
import torch
from torch.autograd import Variable
import probtorch
import torch.nn as nn

from . import tfa_models
from . import utils

TEMPLATE_SHAPE = utils.vardict({
    'factor_centers': None,
    'factor_log_widths': None,
})

class HTFAGuideHyperParams(tfa_models.HyperParams):
    def __init__(self, hyper_means, num_times, num_subjects,
                 num_factors=tfa_models.NUM_FACTORS):
        self._num_times = num_times
        self._num_subjects = num_subjects
        self._num_factors = num_factors

        params = utils.vardict()
        params['template'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy()),
            utils.gaussian_populator,
            self._num_factors
        )
        params['template']['factor_centers'] =\
            utils.gaussian_populator(self._num_factors, 3)
        params['template']['factor_centers']['mu'] +=\
            hyper_means['factor_centers']
        params['template']['factor_log_widths']['mu'] =\
            hyper_means['factor_log_widths'] * torch.ones(self._num_factors)
        params['template']['factor_log_widths']['sigma'] =\
            torch.sqrt(torch.rand(self._num_factors))

        params['subject'] = utils.vardict({
            'factor_centers': {
                'mu': hyper_means['factor_centers'].\
                        repeat(self._num_subjects, 1, 1),
                'sigma': torch.ones(self._num_subjects, self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': torch.ones(self._num_subjects, self._num_factors) *\
                      hyper_means['factor_log_widths'],
                'sigma': torch.sqrt(torch.rand(self._num_subjects, self._num_factors)),
            },
            'weights': {
                'mu': torch.randn(self._num_subjects, self._num_times,
                                  self._num_factors),
                'sigma': torch.ones(self._num_subjects, self._num_times,
                                    self._num_factors),
            },
            'voxel_noise': {
                'mu': torch.ones(self._num_subjects),
                'sigma': torch.sqrt(torch.rand(self._num_subjects))
            }
        })

        super(self.__class__, self).__init__(params, guide=True)

class HTFAGuideTemplatePrior(tfa_models.GuidePrior):
    def forward(self, trace, params, num_particles=tfa_models.NUM_PARTICLES):
        template_params = params['template']
        if num_particles and num_particles > 0:
            template_params = utils.unsqueeze_and_expand_vardict(
                template_params, 0, num_particles, True
            )
        template = TEMPLATE_SHAPE.copy()
        for (k, _) in template.iteritems():
            template[k] = trace.normal(template_params[k]['mu'],
                                       template_params[k]['sigma'],
                                       name='template_' + k)

        return template

class HTFAGuideSubjectPrior(tfa_models.GuidePrior):
    def __init__(self, num_subjects):
        super(self.__class__, self).__init__()
        self._num_subjects = num_subjects
        self._tfa_priors = [tfa_models.TFAGuidePrior(subject=s)\
                            for s in range(self._num_subjects)]

    def forward(self, trace, params, times=None,
                num_particles=tfa_models.NUM_PARTICLES):
        # We only expand the parameters for which we're actually going to sample
        # values in this very method, and thus want to expand to get multiple
        # particles.
        voxel_noise_params = params['subject']['voxel_noise']
        if num_particles and num_particles > 0:
            voxel_noise_params = utils.unsqueeze_and_expand_vardict(
                params['subject']['voxel_noise'], 0, num_particles, True
            )
        voxel_noise = trace.normal(voxel_noise_params['mu'],
                                   voxel_noise_params['sigma'],
                                   name='voxel_noise')

        weights = []
        factor_centers = []
        factor_log_widths = []
        for s in range(self._num_subjects):
            # The TFA prior is going to expand out particles all on its own, so
            # we never actually have to expand them.
            sparams = utils.vardict(params['subject'])
            for k, v in sparams.iteritems():
                sparams[k] = v[s]
            w, fc, flw = self._tfa_priors[s](trace, sparams, times=times,
                                             num_particles=num_particles)
            weights += [w]
            factor_centers += [fc]
            factor_log_widths += [flw]

        return weights, factor_centers, factor_log_widths, voxel_noise

class HTFAGuide(nn.Module):
    """Variational guide for hierarchical topographic factor analysis"""
    def __init__(self, activations, locations,
                 num_factors=tfa_models.NUM_FACTORS):
        super(self.__class__, self).__init__()
        self._num_subjects = len(activations)
        self._num_times = activations[0].shape[0]

        s = np.random.choice(self._num_subjects, 1)[0]
        centers, widths, weights = utils.initial_hypermeans(activations[s].numpy().T,
                                                            locations[s].numpy(),
                                                            num_factors)
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }
        self.hyperparams = HTFAGuideHyperParams(hyper_means, self._num_times,
                                                self._num_subjects, num_factors)
        self._template_prior = HTFAGuideTemplatePrior()
        self._subject_prior = HTFAGuideSubjectPrior(self._num_subjects)

    def forward(self, trace, times=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        self._template_prior(trace, params, num_particles=num_particles)
        return self._subject_prior(trace, params, times=times,
                                   num_particles=num_particles)

class HTFAGenerativeHyperParams(tfa_models.HyperParams):
    def __init__(self, brain_center, brain_center_std_dev, num_subjects,
                 num_factors=tfa_models.NUM_FACTORS):
        self._num_factors = num_factors
        self._num_subjects = num_subjects

        params = utils.vardict()
        params['template'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy()),
            utils.gaussian_populator,
            self._num_factors
        )

        params['template']['factor_centers']['mu'] =\
            brain_center.expand(self._num_factors, 3)
        params['template']['factor_centers']['sigma'] =\
            brain_center_std_dev.expand(self._num_factors, 3)

        params['template']['factor_log_widths']['mu'] =\
            torch.ones(self._num_factors)
        params['template']['factor_log_widths']['sigma'] =\
            tfa_models.SOURCE_LOG_WIDTH_STD_DEV * torch.ones(self._num_factors)

        params['subject'] = {
            'factor_center_noise': torch.ones(self._num_subjects),
            'factor_log_width_noise': torch.ones(self._num_subjects),
            'weights': {
                'mu': torch.rand(self._num_subjects, self._num_factors),
                'sigma': tfa_models.SOURCE_WEIGHT_STD_DEV *\
                         torch.ones(self._num_subjects, self._num_factors)
            },
            'voxel_noise': utils.gaussian_populator(self._num_subjects)
        }
        super(self.__class__, self).__init__(params, guide=False)

class HTFAGenerativeTemplatePrior(tfa_models.GenerativePrior):
    def forward(self, trace, params, guide=probtorch.Trace()):
        template = utils.vardict(TEMPLATE_SHAPE.copy())
        for (k, _) in template.iteritems():
            template[k] = trace.normal(params['template'][k]['mu'],
                                       params['template'][k]['sigma'],
                                       value=guide['template_' + k],
                                       name='template_' + k)

        return template

class HTFAGenerativeSubjectPrior(tfa_models.GenerativePrior):
    def __init__(self, num_subjects, num_times):
        super(self.__class__, self).__init__()
        self._num_subjects = num_subjects
        self._num_times = num_times
        self._tfa_priors = [tfa_models.TFAGenerativePrior(self._num_times[s],
                                                          subject=s)\
                            for s in range(self._num_subjects)]

    def forward(self, trace, params, template, times=None,
                guide=probtorch.Trace()):
        voxel_noise = trace.normal(params['subject']['voxel_noise']['mu'],
                                   params['subject']['voxel_noise']['sigma'],
                                   value=guide['voxel_noise'],
                                   name='voxel_noise')

        weights = []
        factor_centers = []
        factor_log_widths = []
        for s in range(self._num_subjects):
            sparams = utils.vardict({
                'factor_centers': {
                    'mu': template['factor_centers'],
                    'sigma': params['subject']['factor_center_noise'][s],
                },
                'factor_log_widths': {
                    'mu': template['factor_log_widths'],
                    'sigma': params['subject']['factor_log_width_noise'][s],
                },
                'weights': {
                    'mu': params['subject']['weights']['mu'][s],
                    'sigma': params['subject']['weights']['sigma'][s],
                }
            })
            w, fc, flw = self._tfa_priors[s](trace, sparams, times=times,
                                             guide=guide)
            weights += [w]
            factor_centers += [fc]
            factor_log_widths += [flw]

        return weights, factor_centers, factor_log_widths, voxel_noise

class HTFAModel(nn.Module):
    """Generative model for hierarchical topographic factor analysis"""
    def __init__(self, locations, num_subjects, num_times,
                 num_factors=tfa_models.NUM_FACTORS):
        super(self.__class__, self).__init__()

        self._locations = locations
        self._num_factors = num_factors
        self._num_subjects = num_subjects
        self._num_times = num_times

        s = np.random.choice(self._num_subjects, 1)[0]
        center, center_sigma = utils.brain_centroid(self._locations[s])

        self._hyperparams = HTFAGenerativeHyperParams(center, center_sigma,
                                                      self._num_subjects,
                                                      self._num_factors)
        self._template_prior = HTFAGenerativeTemplatePrior()
        self._subject_prior = HTFAGenerativeSubjectPrior(
            self._num_subjects, self._num_times
        )
        self._likelihoods = [tfa_models.TFAGenerativeLikelihood(
            self._locations[s], self._num_times[s], tfa_models.VOXEL_NOISE,
            subject=s
        ) for s in range(self._num_subjects)]
        for s, subject_likelihood in enumerate(self._likelihoods):
            self.add_module('_likelihood' + str(s), subject_likelihood)

    def forward(self, trace, times=None, guide=probtorch.Trace(),
                observations=[]):
        params = self._hyperparams.state_vardict()

        template = self._template_prior(trace, params, guide)
        weights, centers, log_widths, voxel_noise = self._subject_prior(
            trace, params, template, times=times, guide=guide
        )

        return [self._likelihoods[s](trace, weights[s], centers[s], log_widths[s],
                                     times=times, observations=observations[s],
                                     voxel_noise=voxel_noise)
                for s in range(self._num_subjects)]
