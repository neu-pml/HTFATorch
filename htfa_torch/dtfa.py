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

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

import probtorch

from . import dtfa_models
from . import tfa
from . import tfa_models
from . import utils

class DeepTFA:
    """Overall container for a run of Deep TFA"""
    def __init__(self, data_files, mask, num_factors=tfa_models.NUM_FACTORS,
                 embedding_dim=2, tasks=[]):
        self.num_factors = num_factors
        self.num_subjects = len(data_files)
        self.mask = mask
        self.voxel_activations, self.voxel_locations, self._names,\
            self._templates = utils.load_collective_dataset(data_files, mask)
        self._tasks = tasks

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
            variational = torch.nn.DataParallel(self.variational)
            generative = torch.nn.DataParallel(self.generative)
            variational.cuda()
            generative.cuda()
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
            epoch_free_energies = list(range(len(activations_loader)))
            epoch_lls = list(range(len(activations_loader)))

            for (batch, data) in enumerate(activations_loader):
                activations = [{'Y': Variable(data[:, s, :])}
                               for s in range(self.num_subjects)]
                for acts in activations:
                    if tfa.CUDA and use_cuda:
                        acts['Y'] = acts['Y'].cuda()
                trs = (batch * batch_size, None)
                trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                optimizer.zero_grad()
                q = probtorch.Trace()
                variational(q, self.generative.embedding, times=trs,
                            num_particles=num_particles)
                p = probtorch.Trace()
                generative(p, times=trs, guide=q, observations=activations)

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
            variational.cpu()
            generative.cpu()

        return np.vstack([free_energies, lls])

    def results(self, subject):
        hyperparams = self.variational.hyperparams.state_vardict()

        z_f = hyperparams['embedding']['factors']['mu'][subject]
        z_f_embedded = self.generative.embedding.embedder(z_f)

        factors = self.generative.embedding.factors_generator(z_f_embedded)
        factors_shape = (self.num_factors, 4)
        if len(factors.shape) > 1:
            factors_shape = (-1,) + factors_shape
        factors = factors.view(*factors_shape)
        if len(factors.shape) > 2:
            centers = factors[:, :, 0:3]
            log_widths = factors[:, :, 3]
        else:
            centers = factors[:, 0:3]
            log_widths = factors[:, 3]

        z_w = hyperparams['embedding']['weights']['mu'][subject]
        weights = self.generative.embedding.weights_generator(z_w)

        return {
            'weights': weights[0:self.voxel_activations[subject].shape[0], :],
            'factors': tfa_models.radial_basis(self.voxel_locations[subject],
                                               centers.data, log_widths.data),
            'factor_centers': centers.data,
            'factor_log_widths': log_widths.data,
        }

    def embeddings(self):
        hyperparams = self.variational.hyperparams.state_vardict()

        return {
            'factors': hyperparams['embedding']['factors']['mu'],
            'weights': hyperparams['embedding']['weights']['mu'],
        }

    def plot_factor_centers(self, subject, filename=None, show=True,
                            trace=None):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_f_std_dev = hyperparams['embedding']['factors']['sigma'][subject]

        if trace:
            z_f = trace['z_f%d' % subject].value
            if len(z_f.shape) > 1:
                if z_f.shape[0] > 1:
                    z_f_std_dev = z_f.std(0)
                z_f = z_f.mean(0)
        else:
            z_f = hyperparams['embedding']['factors']['mu'][subject]

        z_f_embedded = self.generative.embedding.embedder(z_f)

        factors = self.generative.embedding.factors_generator(z_f_embedded)
        factors_shape = (self.num_factors, 4)
        if len(factors.shape) > 1:
            factors_shape = (-1,) + factors_shape
        factors = factors.view(*factors_shape)
        if len(factors.shape) > 2:
            factor_centers = factors[:, :, 0:3]
            factor_log_widths = factors[:, :, 3]
        else:
            factor_centers = factors[:, 0:3]
            factor_log_widths = factors[:, 3]

        factor_uncertainties = z_f_std_dev.norm().expand(self.num_factors, 1)

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

    def plot_original_brain(self, subject=None, filename=None, show=True,
                            plot_abs=False, t=0):
        if subject is None:
            subject = np.random.choice(self.num_subjects, 1)[0]
        image = nilearn.image.index_img(self._images[subject], t)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, subject=None, filename=None, show=True,
                            plot_abs=False, t=0):
        if subject is None:
            subject = np.random.choice(self.num_subjects, 1)[0]

        results = self.results(subject)

        reconstruction = results['weights'].data @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations[subject].numpy(),
                              self._templates[subject])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(image_slice, plot_abs=plot_abs)

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e',
            np.linalg.norm(
                (reconstruction - self.voxel_activations[subject]).numpy()
            )
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def scatter_factor_embedding(self, filename=None, show=True):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_f = hyperparams['embedding']['factors']['mu'].data

        tasks = self._tasks
        if tasks is None or len(tasks) == 0:
            tasks = list(range(self.num_subjects))
        palette = dict(zip(tasks, utils.compose_palette(len(tasks))))
        subject_colors = np.array([palette[task] for task in tasks])

        plt.scatter(x=z_f[:, 0], y=z_f[:, 1], c=subject_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()

    def scatter_weights_embedding(self, t=None, filename=None, show=True):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_f = hyperparams['embedding']['weights']['mu'].data
        if t is not None:
            z_f = z_f[:, t, :]
        else:
            z_f = z_f.mean(1)

        tasks = self._tasks
        if tasks is None or len(tasks) == 0:
            tasks = list(range(self.num_subjects))
        palette = dict(zip(tasks, utils.compose_palette(len(tasks))))
        subject_colors = np.array([palette[task] for task in tasks])

        plt.scatter(x=z_f[:, 0], y=z_f[:, 1], c=subject_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
