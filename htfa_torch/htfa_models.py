"""Hierarchical factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
import numpy as np
import torch
import probtorch
import torch.nn as nn

from . import tfa_models
from . import utils
from . import hotspot_initialization
# from suggested_initialization import *

TEMPLATE_SHAPE = utils.vardict()
TEMPLATE_SHAPE['weights'] = {
    'mu': {
        'mu': None,
        'sigma': None,
    },
    'sigma': {
        'mu': None,
        'sigma': None,
    }
}
TEMPLATE_SHAPE['factor_centers'] = {
    'mu': None,
    'sigma': None,
}
TEMPLATE_SHAPE['factor_log_widths'] = {
    'mu': None,
    'sigma': None,
}

class HTFAGuideHyperParams(tfa_models.HyperParams):
    def __init__(self, hyper_means, num_times, num_subjects,
                 num_factors=tfa_models.NUM_FACTORS):
        self._num_times = num_times
        self._num_subjects = num_subjects
        self._num_factors = num_factors

        params = utils.vardict()
        params['template'] = utils.vardict(TEMPLATE_SHAPE.copy())
        params['template']['weights'] = utils.populate_vardict(
            params['template']['weights'],
            utils.gaussian_populator,
            self._num_factors,
        )
        params['template']['weights']['mu']['mu']['mu'] =\
            hyper_means['weights'].mean(0)
        params['template']['factor_centers'] = utils.populate_vardict(
            params['template']['factor_centers'],
            utils.gaussian_populator,
            self._num_factors, 3,
        )
        params['template']['factor_centers']['mu']['mu'] =\
            hyper_means['factor_centers']
        params['template']['factor_log_widths'] = utils.populate_vardict(
            params['template']['factor_log_widths'],
            utils.gaussian_populator,
            self._num_factors,
        )
        params['template']['factor_log_widths']['mu']['mu'] +=\
            hyper_means['factor_log_widths']

        params['subject'] = {
            'voxel_noise': {
                'mu': torch.zeros(self._num_subjects),
                'sigma': torch.ones(self._num_subjects),
            },
            'weight_dists': {
                'mu': {
                    'mu': torch.zeros(self._num_subjects, self._num_factors),
                    'sigma': torch.ones(self._num_subjects, self._num_factors),
                },
                'sigma': {
                    'mu': torch.ones(self._num_subjects, self._num_factors),
                    'sigma': torch.ones(self._num_subjects, self._num_factors),
                }
            },
            'weights': {
                'mu': torch.zeros(self._num_subjects, self._num_times, self._num_factors),
                'sigma': torch.ones(self._num_subjects, self._num_times, self._num_factors),
            },
            'factor_centers': {
                'mu': hyper_means['subject_factor_centers'],
                'sigma': torch.ones(self._num_subjects, self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': hyper_means['subject_factor_log_widths'],
                'sigma': torch.ones(self._num_subjects, self._num_factors),
            # 'weights': {
            #     'mu': torch.zeros(self._num_subjects, self._num_times, self._num_factors),
            #     'sigma': torch.ones(self._num_subjects, self._num_times, self._num_factors),
            # },
            # 'factor_centers': {
            #     'mu': torch.zeros(self._num_subjects, self._num_factors, 3),
            #     'sigma': torch.ones(self._num_subjects, self._num_factors, 3),
            # },
            # 'factor_log_widths': {
            #     'mu': torch.zeros(self._num_subjects, self._num_factors),
            #     'sigma': torch.ones(self._num_subjects, self._num_factors),
            }
        }

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
        subject_params = utils.vardict({
            'voxel_noise': params['subject']['voxel_noise'],
            'weight_dists': params['subject']['weight_dists']
        })
        if num_particles and num_particles > 0:
            subject_params = utils.unsqueeze_and_expand_vardict(
                subject_params, 0, num_particles, True
            )
        voxel_noise = trace.normal(subject_params['voxel_noise']['mu'],
                                   subject_params['voxel_noise']['sigma'],
                                   name='voxel_noise')

        for k in params['subject']['weights'].keys():
            trace.normal(
                subject_params['weight_dists'][k]['mu'],
                subject_params['weight_dists'][k]['sigma'],
                name='subject_weights_' + k
            )

        weights = []
        factor_centers = []
        factor_log_widths = []
        for s in range(self._num_subjects):
            # The TFA prior is going to expand out particles all on its own, so
            # we never actually have to expand the rest.
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
        # centers, widths, weights = utils.initial_hypermeans(activations[s].numpy().T,
        #                                                     locations[s].numpy(),
        #                                                     num_factors)
        centers,widths,weights,\
        s_centers,s_widths,s_weights = hotspot_initialization.initialize_centers_widths_weights(
                                                            activations,
                                                            locations,
                                                            num_factors)
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': torch.Tensor(widths),
            'subject_weights' : torch.Tensor(s_weights),
            'subject_factor_centers' : torch.Tensor(s_centers),
            'subject_factor_log_widths' : torch.Tensor(s_widths)
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
        params['template']['factor_centers'] = utils.populate_vardict(
            utils.vardict(TEMPLATE_SHAPE.copy())['factor_centers'],
            utils.gaussian_populator,
            self._num_factors, 3
        )

        params['template']['weights']['sigma']['mu']['mu'] *=\
            tfa_models.SOURCE_WEIGHT_STD_DEV
        params['template']['factor_log_widths']['mu']['mu'] =\
            torch.ones(self._num_factors)
        params['template']['factor_log_widths']['sigma']['mu'] *=\
            tfa_models.SOURCE_LOG_WIDTH_STD_DEV

        params['template']['factor_centers']['mu']['mu'] =\
            brain_center.expand(self._num_factors, 3)
        params['template']['factor_centers']['sigma']['mu'] =\
            brain_center_std_dev.expand(self._num_factors, 3)

        params['voxel_noise'] = {
            'mu': torch.zeros(self._num_subjects),
            'sigma': torch.ones(self._num_subjects),
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
        voxel_noise = trace.normal(params['voxel_noise']['mu'],
                                   params['voxel_noise']['sigma'],
                                   value=guide['voxel_noise'],
                                   name='voxel_noise')

        subject_weights_template = utils.unsqueeze_and_expand_vardict(
            template['weights'], len(template['weights']['mu']['mu'].shape) - 1,
            self._num_subjects, True
        )
        subject_weight_params = utils.vardict({
            'mu': None,
            'sigma': None,
        })
        for k in subject_weight_params.iterkeys():
            subject_weight_params[k] = trace.normal(
                subject_weights_template[k]['mu'],
                subject_weights_template[k]['sigma'],
                value=guide['subject_weights_' + k], name='subject_weights_' + k
            )
            if len(subject_weight_params[k].shape) >\
               len(params['template']['weights'][k]['mu']['mu'].shape) + 1:
                select_dim = 1
            else:
                select_dim = 0

        weights = []
        factor_centers = []
        factor_log_widths = []
        for s in range(self._num_subjects):
            sparams = utils.vardict()
            sparams['weights'] = {k: v.select(select_dim, s) for k, v in subject_weight_params.iteritems()}
            sparams['factor_centers'] = template['factor_centers']
            sparams['factor_log_widths'] = template['factor_log_widths']
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
