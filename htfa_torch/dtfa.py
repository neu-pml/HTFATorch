"""Perform deep topographic factor analysis on fMRI data"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import logging
import os
import pickle
import time

import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import probtorch

from . import dtfa_models
from . import tfa
from . import tfa_models
from . import utils

class DeepTFA:
    """Overall container for a run of Deep TFA"""
    def __init__(self, data_files, num_factors=tfa_models.NUM_FACTORS,
                 embedding_dim=2):
        self.num_factors = num_factors
        self.num_subjects = len(data_files)
        datasets = [utils.load_dataset(data_file) for data_file in data_files]
        self.voxel_activations = [dataset[0] for dataset in datasets]
        self._images = [dataset[1] for dataset in datasets]
        self.voxel_locations = [dataset[2] for dataset in datasets]
        self._names = [dataset[3] for dataset in datasets]
        self._templates = [dataset[4] for dataset in datasets]

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, self.voxel_activations, self.num_factors,
            self.num_subjects, self.num_times, embedding_dim
        )
        self.variational = dtfa_models.DeepTFAGuide(self.num_subjects,
                                                    self.num_times,
                                                    embedding_dim)

    def sample(self, posterior_predictive=False, num_particles=1):
        q = probtorch.Trace()
        if posterior_predictive:
            self.variational(q, self.generative.embedding,
                             num_particles=num_particles)
        p = probtorch.Trace()
        self.generative(p, guide=q,
                        observations=[q for s in range(self.num_subjects)])
        return p, q

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              use_cuda=True):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        activations = [{'Y': Variable(self.voxel_activations[s])}
                       for s in range(self.num_subjects)]
        if tfa.CUDA and use_cuda:
            variational = torch.nn.DataParallel(self.variational)
            generative = torch.nn.DataParallel(self.generative)
            variational.cuda()
            generative.cuda()
            for acts in activations:
                acts['Y'] = acts['Y'].cuda()
        else:
            variational = self.variational
            generative = self.generative

        optimizer = torch.optim.Adam(list(variational.parameters()) +\
                                     list(generative.parameters()),
                                     lr=learning_rate)
        variational.train()
        generative.train()

        free_energies = list(range(num_steps))
        lls = list(range(num_steps))

        for epoch in range(num_steps):
            start = time.time()

            optimizer.zero_grad()
            q = probtorch.Trace()
            variational(q, self.generative.embedding,
                        num_particles=num_particles)
            p = probtorch.Trace()
            generative(p, guide=q, observations=activations)

            free_energies[epoch] = tfa.free_energy(q, p, num_particles=num_particles)
            lls[epoch] = tfa.log_likelihood(q, p, num_particles=num_particles)

            free_energies[epoch].backward()
            optimizer.step()

            if tfa.CUDA and use_cuda:
                free_energies[epoch] = free_energies[epoch].cpu()
                lls[epoch] = lls[epoch].cpu()
            free_energies[epoch] = free_energies[epoch].data.numpy().sum(0)
            lls[epoch] = lls[epoch].data.numpy().sum(0)

            end = time.time()
            msg = tfa.EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)

        if tfa.CUDA and use_cuda:
            variational.cpu()
            generative.cpu()

        return np.vstack([free_energies, lls])

    def plot_factor_centers(self, subject, filename=None, show=True,
                            trace=None):
        hyperparams = self.variational.hyperparams[subject].state_vardict()
        z_f_std_dev = hyperparams['embedding']['factors']['sigma']

        if trace:
            z_f = trace['z_f%d' % subject].value
            if len(z_f.shape) > 1:
                if z_f.shape[0] > 1:
                    z_f_std_dev = z_f.std(0)
                z_f = z_f.mean(0)
        else:
            z_f = hyperparams['embedding']['factors']['mu']

        z_f_embedded = self.generative.embedding.embedder(z_f)

        factor_centers = self.generative.embedding.factor_centers_generator(
            z_f_embedded
        )
        centers_shape = (self.num_factors, 3)
        if len(factor_centers.shape) > 1:
            centers_shape = (-1,) + centers_shape
        factor_centers = factor_centers.view(*centers_shape)
        factor_uncertainties = z_f_std_dev.norm().expand(self.num_factors, 1)

        factor_log_widths =\
            self.generative.embedding.factor_log_widths_generator(
                z_f_embedded
            )
        if len(factor_log_widths.shape) > 1:
            factor_log_widths = factor_log_widths.view(-1, self.num_factors)

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            factor_centers.data.numpy(),
            node_color=utils.uncertainty_palette(factor_uncertainties.data),
            node_size=np.exp(factor_log_widths.data.numpy() - np.log(2))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot
