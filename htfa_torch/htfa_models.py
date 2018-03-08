"""Hierarchical factor analysis models as ProbTorch modules"""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import numpy as np
import torch
import probtorch
import torch.nn as nn

from . import tfa_models
from . import utils

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
    'mu': {
        'mu': None,
        'sigma': None,
    },
    'sigma': {
        'mu': None,
        'sigma': None,
    }
}
TEMPLATE_SHAPE['factor_log_widths'] = {
    'mu': {
        'mu': None,
        'sigma': None,
    },
    'sigma': {
        'mu': None,
        'sigma': None,
    }
}

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
        params['template']['factor_log_widths']['sigma']['mu']['mu'] *=\
            tfa_models.SOURCE_LOG_WIDTH_STD_DEV

        params['template']['factor_centers']['mu']['mu']['mu'] =\
            brain_center.expand(self._num_factors, 3)
        params['template']['factor_centers']['sigma']['mu']['mu'] =\
            brain_center_std_dev.expand(self._num_factors, 3)

        params['voxel_noise'] = {
            'mu': torch.zeros(self._num_subjects),
            'sigma': torch.ones(self._num_subjects),
        }
        super(self.__class__, self).__init__(params, guide=False)

class HTFAGenerativeTemplatePrior(tfa_models.GenerativePrior):
    def forward(self, trace, params, guide=probtorch.Trace()):
        template = TEMPLATE_SHAPE.copy()
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

        subject_template = utils.unsqueeze_and_expand_vardict(
            template, 0, self._num_subjects, True
        )
        subject_params = utils.vardict({
            'weights': {
                'mu': None,
                'sigma': None,
            },
            'factor_centers': {
                'mu': None,
                'sigma': None,
            },
            'factor_log_widths': {
                'mu': None,
                'sigma': None,
            },
        })
        for (k, _) in subject_params.iteritems():
            subject_params[k] = trace.normal(subject_template[k]['mu'],
                                             subject_template[k]['sigma'],
                                             value=guide['subject_params_' + k],
                                             name='subject_params_' + k)

        weights = []
        factor_centers = []
        factor_log_widths = []
        for s in range(self._num_subjects):
            sparams = utils.vardict(subject_params)
            for k, v in sparams.iteritems():
                sparams[k] = v[s]
            w, fc, flw = self._tfa_priors[s](trace, sparams, times=times,
                                             guide=guide)
            weights += [w]
            factor_centers += [fc]
            factor_log_widths += [flw]

        return weights, factor_centers, factor_log_widths, voxel_noise
