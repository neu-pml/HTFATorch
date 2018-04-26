"""Perform hierarchical topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import logging
import os
import pickle
import time

import hypertools as hyp
import nibabel as nib
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
from . import niidb
from . import tfa
from . import tfa_models
from . import utils

class HierarchicalTopographicFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, query, num_factors=tfa_models.NUM_FACTORS,
                 mask=None):
        self.num_factors = num_factors
        if mask is None:
            raise ValueError('please provide a mask')
        else:
            self.mask = mask
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self.voxel_locations = [block.locations for block in self._blocks]
        self._templates = [block.filename for block in self._blocks]

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        self.enc = htfa_models.HTFAGuide(query, self.num_factors)
        self.dec = htfa_models.HTFAModel(query, self.num_blocks, self.num_times,
                                         self.num_factors)

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)
        # S x T x V -> T x S x V
        activations_loader = torch.utils.data.DataLoader(
            utils.TFADataset(self.voxel_activations),
            batch_size=batch_size
        )
        if tfa.CUDA and use_cuda:
            enc = torch.nn.DataParallel(self.enc)
            dec = torch.nn.DataParallel(self.dec)
            enc.cuda()
            dec.cuda()
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

            for (batch, data) in enumerate(activations_loader):
                activations = [{'Y': Variable(data[:, b, :])}
                               for b in range(self.num_blocks)]
                for acts in activations:
                    if tfa.CUDA and use_cuda:
                        acts['Y'] = acts['Y'].cuda()
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

    def plot_voxels(self, block=None):
        if block:
            hyp.plot(self.voxel_locations[block].numpy(), 'k.')
        else:
            for b in range(self.num_blocks):
                hyp.plot(self.voxel_locations[b].numpy(), 'k.')

    def plot_factor_centers(self, block=None, filename=None, show=True,
                            trace=None):
        hyperparams = self.results()

        if trace:
            if block is not None:
                factor_centers = trace['FactorCenters%d' % block].value
                factor_log_widths = trace['FactorLogWidths%d' % block].value
                factor_uncertainties = hyperparams['block']['factor_centers']['sigma'][block]
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
            if block is not None:
                factor_centers =\
                    hyperparams['block']['factor_centers']['mu'][block]
                factor_log_widths =\
                    hyperparams['block']['factor_log_widths']['mu'][block]
                factor_uncertainties =\
                    hyperparams['block']['factor_centers']['sigma'][block]
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
                 observations=[q for b in range(self.num_blocks)])
        return p, q

    def plot_original_brain(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0):
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        image = nilearn.image.index_img(nib.load(self._templates[block]), t)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0):
        results = self.results()

        if block is not None:
            weights = results['block']['weights']['mu'][block]
            factor_centers = results['block']['factor_centers']['mu'][block]
            factor_log_widths =\
                results['block']['factor_log_widths']['mu'][block]
        else:
            factor_centers = results['template']['factor_centers']['mu']
            factor_log_widths =\
                results['template']['factor_log_widths']['mu']
            block = np.random.choice(self.num_blocks, 1)[0]
            weights = results['block']['weights']['mu'][block]

        factors = tfa_models.radial_basis(
            self.voxel_locations[block], factor_centers.data,
            factor_log_widths.data
        )
        times = (0, self.voxel_activations[block].shape[0])
        reconstruction = weights[times[0]:times[1], :].data @ factors

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations[block].numpy(),
                              self._templates[block])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(image_slice, plot_abs=plot_abs)

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e',
            np.linalg.norm(
                (reconstruction - self.voxel_activations[block]).numpy()
            )
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot
