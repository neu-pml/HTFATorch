"""Perform hierarchical topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import collections
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

import hypertools as hyp
import nibabel as nib
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, query, mask, num_factors=tfa_models.NUM_FACTORS):
        self.num_factors = num_factors
        self.mask = mask
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self.voxel_locations = self._blocks[0].locations
        for block in self._blocks:
            block.unload_locations()
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
              batch_size=64, use_cuda=True, blocks_batch_size=4):
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
        rv_occurrences = collections.defaultdict(int)
        measure_occurrences = True

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = list(range(len(activations_loader)))

            for (batch, data) in enumerate(activations_loader):
                epoch_free_energies[batch] = 0.0
                block_batches = utils.chunks(list(range(self.num_blocks)),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    activations = [{'Y': Variable(data[:, b, :])}
                                   for b in block_batch]
                    if tfa.CUDA and use_cuda:
                        for acts in activations:
                            acts['Y'] = acts['Y'].cuda()
                        for b in block_batch:
                            dec.module.likelihoods[b].voxel_locations =\
                                self.voxel_locations.cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    enc(q, times=trs, num_particles=num_particles,
                        blocks=block_batch)
                    p = probtorch.Trace()
                    dec(p, times=trs, guide=q, observations=activations,
                        blocks=block_batch)

                    def block_rv_weight(node):
                        result = 1.0
                        if measure_occurrences:
                            rv_occurrences[node] += 1
                        if 'Weights' not in node and 'Y' not in node:
                            result /= rv_occurrences[node]
                        return result
                    free_energy = tfa.hierarchical_free_energy(
                        q, p,
                        rv_weight=block_rv_weight,
                        num_particles=num_particles
                    )

                    free_energy.backward()
                    optimizer.step()
                    epoch_free_energies[batch] += free_energy

                    if tfa.CUDA and use_cuda:
                        del activations
                        for b in block_batch:
                            locs = dec.module.likelihoods[b].voxel_locations
                            dec.module.likelihoods[b].voxel_locations =\
                                locs.cpu()
                            del locs
                        torch.cuda.empty_cache()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).sum(0)
            free_energies[epoch] = free_energies[epoch].sum(0)

            measure_occurrences = False

            end = time.time()
            msg = tfa.EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)

        if tfa.CUDA and use_cuda:
            dec.cpu()
            enc.cpu()

        return np.vstack([free_energies])

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
            hyp.plot(self.voxel_locations.numpy(), 'k.')
        else:
            for b in range(self.num_blocks):
                hyp.plot(self.voxel_locations.numpy(), 'k.')

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
            self.voxel_locations, factor_centers.data,
            factor_log_widths.data
        )
        times = (0, self.voxel_activations[block].shape[0])
        reconstruction = weights[times[0]:times[1], :].data @ factors

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
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

    def scatter_factor_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None, figsize=None,embedding=TSNE):

        factor_centers_map = self.enc.hyperparams.block__factor_centers__mu.data.numpy()
        factor_widths_map = self.enc.hyperparams.block__factor_log_widths__mu.data.numpy()
        factors_map = np.concatenate((np.expand_dims(factor_widths_map, 2), factor_centers_map), axis=2)
        factors_map = np.reshape(factors_map, newshape=(self.num_blocks, self.num_factors * 4))
        X = StandardScaler().fit_transform(factors_map)
        if embedding == 'TSNE':
            z_f = TSNE(n_components=2).fit_transform(X)
        else:
            z_f = PCA(n_components=2).fit_transform(X)


        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        all_labels = np.unique(all_labels)
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_fs = [z_f[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_fs = np.stack(z_fs)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('Embedded Dimension_1')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('Embedded Dimension_2')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Factor Embeddings')
        ax.scatter(x=z_fs[:, 0], y=z_fs[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()

    def scatter_weight_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None, figsize=None,embedding='TSNE'):

        weight_map = self.enc.hyperparams.block__weights__mu.data.numpy()
        weight_map = np.reshape(weight_map, newshape=(self.num_blocks, self.num_factors * weight_map.shape[1]))
        X = StandardScaler().fit_transform(weight_map)
        if embedding == 'TSNE':
            z_w = TSNE(n_components=2).fit_transform(X)
        else:
            z_w = PCA(n_components=2).fit_transform(X)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        all_labels = np.unique(all_labels)
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_ws = [z_w[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_ws = np.stack(z_ws)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('Embedded Dimension_1')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('Embedded Dimension_2')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Weight Embeddings')
        ax.scatter(x=z_ws[:, 0], y=z_ws[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            plt.show()



    def decoding_accuracy(self):
        """
        :return: accuracy: a dict containing decoding accuracies for each task [activity,isfc,mixed]
        """
        W = self.enc.hyperparams.block__weights__mu.data
        keys = ['rest','task']
        group = {key: [] for key in keys}
        accuracy = {key:[] for key in keys}
        isfc_accuracy = {key:[] for key in keys}
        mixed_accuracy = {key:[] for key in keys}
        for n in range(len(self._blocks)):
            if self._blocks[n].task == 'rest':
                group['rest'].append(W[n,:,:])
            else:
                group['task'].append(W[n,:,:])
        group['rest'] = np.rollaxis(np.dstack(group['rest']),-1)
        group['task'] = np.rollaxis(np.dstack(group['task']),-1)

        for key in keys:
            G1 = group[key][:int(group[key].shape[0]/2),:,:]
            G2 = group[key][int(group[key].shape[0]/2):,:,:]
            accuracy[key].append(utils.get_decoding_accuracy(G1,G2,5))
            accuracy[key].append(utils.get_isfc_decoding_accuracy(G1,G2,5))
            accuracy[key].append(utils.get_mixed_decoding_accuracy(G1,G2,5))

        return accuracy



