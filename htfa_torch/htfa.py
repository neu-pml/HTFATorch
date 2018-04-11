"""Perform hierarchical topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import logging
import os
import pickle
import time

import hypertools as hyp
import nilearn.image
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

from . import htfa_models
from . import tfa
from . import tfa_models
from . import utils

class HierarchicalTopographicFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_files, num_factors=tfa_models.NUM_FACTORS,
                 mask=None):
        self.num_factors = num_factors
        self.num_subjects = len(data_files)
        if mask is None:
            raise ValueError('please provide a mask')
        else:
            self.mask = mask
        datasets = [utils.load_dataset(data_file, mask=mask)
                    for data_file in data_files]
        self.voxel_activations = [dataset[0] for dataset in datasets]
        self._images = [dataset[1] for dataset in datasets]
        self.voxel_locations = [dataset[2] for dataset in datasets]
        self._names = [dataset[3] for dataset in datasets]
        self._templates = [dataset[4] for dataset in datasets]

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        self.enc = htfa_models.HTFAGuide(self.voxel_activations,
                                         self.voxel_locations,
                                         self.num_factors)
        self.dec = htfa_models.HTFAModel(self.voxel_locations, self.num_subjects,
                                         self.num_times, self.num_factors)

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)
        activations = torch.Tensor(max(self.num_times), max(self.num_voxels),
                                   len(self.voxel_activations))
        for s in range(self.num_subjects):
            activations[:, :, s] = self.voxel_activations[s]
        activations_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                activations,
                torch.zeros(activations.shape[0])
            ),
            batch_size=batch_size
        )
        if tfa.CUDA and use_cuda:
            enc = torch.nn.DataParallel(self.enc)
            dec = torch.nn.DataParallel(self.dec)
            enc.cuda()
            dec.cuda(0)
        else:
            enc = self.enc
            dec = self.dec
        optimizer = torch.optim.Adam(list(self.enc.parameters()),
                                     lr=learning_rate)
        enc.train()
        dec.train()

        free_energies = list(range(num_steps))
        lls = list(range(num_steps))

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = list(range(len(activations_loader)))
            epoch_lls = list(range(len(activations_loader)))

            for (batch, (data, _)) in enumerate(activations_loader):
                activations = [{'Y': Variable(data[:, :, s])}
                               for s in range(self.num_subjects)]
                trs = (batch * batch_size, None)
                trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])


                optimizer.zero_grad()
                q = probtorch.Trace()
                enc(q, times=trs, num_particles=num_particles)
                p = probtorch.Trace()
                dec(p, times=trs, guide=q, observations=activations)


                epoch_free_energies[batch] =\
                    tfa.free_energy(q, p, num_particles=num_particles)
                epoch_lls[batch] =\
                    tfa.log_likelihood(q, p, num_particles=num_particles)
                epoch_free_energies[batch].backward()
                optimizer.step()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].cpu().data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).sum(0)
            free_energies[epoch] = free_energies[epoch].sum(0)
            lls[epoch] = np.array(epoch_lls).sum(0)
            lls[epoch] = lls[epoch].sum(0)

            end = time.time()
            msg = tfa.EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)

        if tfa.CUDA and use_cuda:
            dec.cpu()
            enc.cpu()

        return np.vstack([free_energies, lls])

    def save(self, out_dir='.'):
        '''Save a HierarchicalTopographicFactorAnalysis'''
        with open(out_dir + '/' + self._name + '.htfa', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        '''Load a saved HierarchicalTopographicFactorAnalysis from a file'''
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def results(self):
        """Return the inferred variational parameters"""
        return self.enc.hyperparams.state_vardict()

    def plot_voxels(self, subject=None):
        if subject:
            hyp.plot(self.voxel_locations[subject].numpy(), 'k.')
        else:
            for s in range(self.num_subjects):
                hyp.plot(self.voxel_locations[s].numpy(), 'k.')

    def plot_factor_centers(self, subject=None, filename=None, show=True,
                            trace=None):
        hyperparams = self.results()

        if trace:
            if subject is not None:
                factor_centers = trace['FactorCenters%d' % subject].value
                factor_log_widths = trace['FactorLogWidths%d' % subject].value
                factor_uncertainties = hyperparams['subject']['factor_centers']['sigma'][subject]
            else:
                factor_centers = trace['template_factor_centers'].value
                factor_log_widths = trace['template_factor_log_widths'].value
                factor_uncertainties =\
                    hyperparams['template']['factor_centers']['sigma']
            if len(factor_centers.shape) > 2:
                factor_centers = factor_centers.mean(0)
                factor_log_widths = factor_log_widths.mean(0)
            if len(factor_uncertainties.shape) > 2:
                factor_uncertainties = factor_uncertainties.mean(0)
        else:
            if subject is not None:
                factor_centers =\
                    hyperparams['subject']['factor_centers']['mu'][subject]
                factor_log_widths =\
                    hyperparams['subject']['factor_log_widths']['mu'][subject]
                factor_uncertainties =\
                    hyperparams['subject']['factor_centers']['sigma'][subject]
            else:
                factor_centers =\
                    hyperparams['template']['factor_centers']['mu']
                factor_log_widths =\
                    hyperparams['template']['factor_log_widths']['mu']
                factor_uncertainties =\
                    hyperparams['template']['factor_centers']['sigma']

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

    def sample(self, times=None, posterior_predictive=False):
        q = probtorch.Trace()
        if posterior_predictive:
            self.enc(q, times=times, num_particles=1)
        p = probtorch.Trace()
        self.dec(p, times=times, guide=q,
                 observations=[q for s in range(self.num_subjects)])
        return p, q
