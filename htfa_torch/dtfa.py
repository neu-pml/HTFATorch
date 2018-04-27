"""Perform deep topographic factor analysis on fMRI data"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import datetime
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
    def __init__(self, query, mask, num_factors=tfa_models.NUM_FACTORS,
                 embedding_dim=2):
        self.num_factors = num_factors
        self.mask = mask
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self.voxel_locations = [block.locations for block in self._blocks]
        self._templates = [block.filename for block in self._blocks]
        self._tasks = [block.task for block in self._blocks]

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        b = np.random.choice(self.num_blocks, 1)[0]
        centers, widths, weights = utils.initial_hypermeans(
            self.voxel_activations[b].numpy().T, self.voxel_locations[b].numpy(), self.num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths * torch.ones(self.num_factors),
        }

        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, hyper_means, self.num_factors,
            self.num_blocks, self.num_times, embedding_dim
        )
        self.variational = dtfa_models.DeepTFAGuide(hyper_means,
                                                    self.num_factors,
                                                    self.num_blocks,
                                                    self.num_times,
                                                    embedding_dim)

    def sample(self, posterior_predictive=False, num_particles=1):
        q = probtorch.Trace()
        if posterior_predictive:
            self.variational(q, self.generative.embedding,
                             num_particles=num_particles)
        p = probtorch.Trace()
        self.generative(p, guide=q,
                        observations=[q for b in range(self.num_blocks)])
        return p, q

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True, checkpoint_steps=None,
              blocks_batch_size=4):
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
                            generative.module.likelihoods[b].voxel_locations =\
                            generative.module.likelihoods[b].voxel_locations.cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    variational(q, self.generative.embedding, times=trs,
                                num_particles=num_particles, blocks=block_batch)
                    p = probtorch.Trace()
                    generative(p, times=trs, guide=q, observations=activations,
                               blocks=block_batch)

                    free_energy = tfa.free_energy(q, p,
                                                  num_particles=num_particles)

                    free_energy.backward()
                    optimizer.step()
                    epoch_free_energies[batch] += free_energy

                    if tfa.CUDA and use_cuda:
                        del activations
                        for b in block_batch:
                            locs = generative.module.likelihoods[b].voxel_locations
                            generative.module.likelihoods[b].voxel_locations =\
                                locs.cpu()
                            del locs
                        torch.cuda.empty_cache()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).sum(0)
            free_energies[epoch] = free_energies[epoch].sum(0)

            end = time.time()
            msg = tfa.EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)
            if checkpoint_steps is not None and epoch % checkpoint_steps == 0:
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
                logging.info('Saving checkpoint...')
                self.save_state(path='.', tag=checkpoint_name)

        if tfa.CUDA and use_cuda:
            variational.cpu()
            generative.cpu()

        return np.vstack([free_energies])

    def results(self, block):
        template_centers = hyperparams['template']['factor_centers']['mu']
        template_log_widths = hyperparams['template']['factor_log_widths']['mu']
        hyperparams = self.variational.hyperparams.state_vardict()

        z_f = hyperparams['embedding']['factors']['mu'][block]
        z_f_embedded = self.generative.embedding.embedder(z_f)

        factors = self.generative.embedding.factors_generator(z_f_embedded)
        factors_shape = (self.num_factors, 4)
        if len(factors.shape) > 1:
            factors_shape = (-1,) + factors_shape
        factors = factors.view(*factors_shape)
        if len(factors.shape) > 2:
            centers = factors[:, :, 0:3] + template_centers
            log_widths = factors[:, :, 3] + template_log_widths
        else:
            centers = factors[:, 0:3] + template_centers
            log_widths = factors[:, 3] + template_log_widths

        z_w = hyperparams['embedding']['weights']['mu'][block]
        weights = self.generative.embedding.weights_generator(
            self.generative.embedding.embedder(z_w)
        )

        return {
            'weights': weights[0:self.voxel_activations[block].shape[0], :],
            'factors': tfa_models.radial_basis(self.voxel_locations[block],
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

    def plot_factor_centers(self, block, filename=None, show=True,
                            trace=None):
        hyperparams = self.variational.hyperparams.state_vardict()

        template_centers = hyperparams['template']['factor_centers']['mu']
        template_log_widths = hyperparams['template']['factor_log_widths']['mu']

        z_f_std_dev = hyperparams['embedding']['factors']['sigma'][block]

        if trace:
            z_f = trace['z_f%d' % block].value
            if len(z_f.shape) > 1:
                if z_f.shape[0] > 1:
                    z_f_std_dev = z_f.std(0)
                z_f = z_f.mean(0)
        else:
            z_f = hyperparams['embedding']['factors']['mu'][block]

        z_f_embedded = self.generative.embedding.embedder(z_f)
        factors_std_dev = self.generative.embedding.factors_generator(
            self.generative.embedding.embedder(z_f_std_dev)
        )

        factors = self.generative.embedding.factors_generator(z_f_embedded)
        factors_shape = (self.num_factors, 4)
        if len(factors.shape) > 1:
            factors_shape = (-1,) + factors_shape
        factors = factors.view(*factors_shape)
        factors_std_dev = factors_std_dev.view(*factors_shape)
        if len(factors.shape) > 2:
            factor_centers = factors[:, :, 0:3].clone()
            factor_log_widths = factors[:, :, 3].clone()
            factors_std_dev = factors_std_dev[:, :, 0:3].clone()
        else:
            factor_centers = factors[:, 0:3].clone()
            factor_log_widths = factors[:, 3].clone()
            factors_std_dev = factors_std_dev[:, 0:3].clone()
        factor_centers = factor_centers + template_centers
        factor_log_widths = torch.log(torch.exp(factor_log_widths) +
                                      torch.exp(template_log_widths))

        factor_uncertainties =\
            hyperparams['template']['factor_centers']['sigma']
        factor_uncertainties = factor_uncertainties + factors_std_dev

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            template_centers.data.numpy(),
            node_color=utils.uncertainty_palette(factor_uncertainties.data),
            node_size=np.exp(factor_log_widths.data.numpy() - np.log(2))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0):
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        image = nilearn.image.index_img(self._images[block], t)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0):
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]

        results = self.results(block)

        reconstruction = results['weights'].data @ results['factors']

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

    def scatter_factor_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_f = hyperparams['embedding']['factors']['mu'].data

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_fs = [z_f[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_fs = torch.stack(z_fs)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=(3.75, 2.75))
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^F_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^F_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Factor Embeddings')
        ax.scatter(x=z_fs[:, 0], y=z_fs[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            fig.show()

    def scatter_weights_embedding(self, t=None, labeler=None, filename=None,
                                  show=True, xlims=None, ylims=None):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_w = hyperparams['embedding']['weights']['mu'].data
        if t is not None:
            z_w = z_w[:, t, :]
        else:
            z_w = z_w.mean(1)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = [l for l in labels if l is not None]
        palette = dict(zip(all_labels, utils.compose_palette(len(all_labels))))

        z_ws = [z_w[b] for b in range(self.num_blocks) if labels[b] is not None]
        z_ws = torch.stack(z_ws)
        block_colors = [palette[labels[b]] for b in range(self.num_blocks)
                        if labels[b] is not None]

        fig = plt.figure(1, figsize=(3.75, 2.75))
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^W_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^W_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Weight Embeddings')
        ax.scatter(x=z_ws[:, 0], y=z_ws[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            fig.show()

    def common_name(self):
        return os.path.commonprefix([os.path.basename(b.filename)
                                     for b in self._blocks])

    def save_state(self, path='.', tag=''):
        name = self.common_name() + tag
        variational_state = self.variational.state_dict()
        torch.save(variational_state,
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.generative.state_dict(),
                   path + '/' + name + '.dtfa_model')

    def save(self, path='.'):
        name = self.common_name()
        torch.save(self.variational.state_dict(),
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.generative.state_dict(),
                   path + '/' + name + '.dtfa_model')
        with open(path + '/' + name + '.dtfa', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_state(self, basename):
        model_state = torch.load(basename + '.dtfa_model')
        self.generative.load_state_dict(model_state)

        guide_state = torch.load(basename + '.dtfa_guide')
        self.variational.load_state_dict(guide_state)

    @classmethod
    def load(cls, basename):
        with open(basename + '.dtfa', 'rb') as pickle_file:
            dtfa = pickle.load(pickle_file)
        dtfa.load_state(basename)

        return dtfa
