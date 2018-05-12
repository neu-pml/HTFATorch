"""Perform deep topographic factor analysis on fMRI data"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections
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
            block.unload_locations()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self._blocks[-1].load()
        self.voxel_locations = self._blocks[-1].locations
        self._templates = [block.filename for block in self._blocks]
        self._tasks = [block.task for block in self._blocks]

        self.weight_normalizers = None

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        subjects = list(set([b.subject for b in self._blocks]))
        tasks = list(set([b.task for b in self._blocks]))
        block_subjects = [subjects.index(b.subject) for b in self._blocks]
        block_tasks = [tasks.index(b.task) for b in self._blocks]

        b = np.random.choice(self.num_blocks, 1)[0]
        self._blocks[b].load()
        centers, widths, weights = utils.initial_hypermeans(
            self._blocks[b].activations.numpy().T, self._blocks[b].locations.numpy(),
            num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }
        self._blocks[b].unload()

        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, block_subjects, block_tasks,
            self.num_factors, self.num_blocks, self.num_times, embedding_dim,
            hyper_means
        )
        self.variational = dtfa_models.DeepTFAGuide(self.num_factors,
                                                    len(subjects), len(tasks),
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
                                     lr=learning_rate, weight_decay=1e-2)
        variational.train()
        generative.train()

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
                            generative.module.likelihoods[b].voxel_locations =\
                                self.voxel_locations.cuda()
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    variational(q, self.generative.embedding, times=trs,
                                num_particles=num_particles, blocks=block_batch)
                    p = probtorch.Trace()
                    generative(p, times=trs, guide=q, observations=activations,
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
                            locs = generative.module.likelihoods[b].voxel_locations
                            generative.module.likelihoods[b].voxel_locations =\
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
            if checkpoint_steps is not None and epoch % checkpoint_steps == 0:
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
                logging.info('Saving checkpoint...')
                self.save_state(path='.', tag=checkpoint_name)

        if tfa.CUDA and use_cuda:
            variational.cpu()
            generative.cpu()

        now = datetime.datetime.now()
        checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
        logging.info('Saving checkpoint...')
        self.save_state(path='.', tag=checkpoint_name)

        return np.vstack([free_energies])

    def results(self, block):
        hyperparams = self.variational.hyperparams.state_vardict()
        subject = self.generative.embedding.subjects[block]
        task = self.generative.embedding.tasks[block]

        subject_embed = hyperparams['subject']['mu'][subject]
        subject_embed = subject_embed.expand(
            self.num_times[block], *subject_embed.shape
        )
        task_embed = hyperparams['task']['mu'][task][0:self.num_times[block]]
        factors_embed = hyperparams['factors']['mu'][subject]
        weights_embed = torch.cat((subject_embed, task_embed), dim=-1)

        weight_params =\
            self.generative.embedding.weights_generator(weights_embed)
        weight_params = weight_params.view(weight_params.shape[0], 2,
                                           self.num_factors)
        weight_mus = weight_params[:, 0]
        weight_sigmas = self.generative.embedding.softplus(weight_params[:, 1])
        factor_params =\
            self.generative.embedding.factors_generator(factors_embed)
        factor_params = factor_params.view(self.num_factors, 4)
        factor_centers = factor_params[:, 0:3]
        factor_log_widths = factor_params[:, 3]

        return {
            'weights': {
                'mu': weight_mus,
                'sigma': weight_sigmas,
            },
            'factors': tfa_models.radial_basis(self.voxel_locations,
                                               factor_centers.data,
                                               factor_log_widths.data),
            'factor_centers': factor_centers.data,
            'factor_log_widths': factor_log_widths.data,
        }

    def normalize_weights(self):
        def weights_generator(run, subject):
            run_blocks = [b for b in range(len(self._blocks))
                          if self._blocks[b].run == run and\
                             self._blocks[b].subject == subject]
            for rb in run_blocks:
                weights = self.results(rb)['weights']['mu'].data
                yield weights.contiguous().view(-1)
        runs = list(set([(b.run, b.subject) for b in self._blocks]))
        runs.sort(key=lambda p: p[0])
        self.weight_normalizers = runs.copy()
        for (i, (run, subject)) in enumerate(runs):
            weights = list(weights_generator(run, subject))
            idw = utils.normalize_tensors(weights, percentiles=(10, 90))
            absw = utils.normalize_tensors(weights, absval=True,
                                           percentiles=(30, 70))
            self.weight_normalizers[i] = (idw, absw)

        return self.weight_normalizers

    def plot_factor_centers(self, block, filename=None, show=True,
                            colormap='cold_white_hot', t=None, labeler=None,
                            uncertainty_opacity=False):
        if labeler is None:
            labeler = lambda b: b.task
        results = self.results(block)
        hyperparams = self.variational.hyperparams.state_vardict()
        subject = self.generative.embedding.subjects[block]

        factor_params = self.generative.embedding.softplus(
            self.generative.embedding.factors_generator(
                hyperparams['factors']['sigma'][subject]
            )
        )
        factors_std_dev = factor_params.view(self.num_factors, 4)[:, 0:3]

        _, brain_center_std_dev = utils.brain_centroid(self.voxel_locations)
        brain_center_std_dev = brain_center_std_dev.expand(
            self.num_factors, 3
        )
        weights = results['weights']['mu']
        if t is not None:
            weights = weights[t]
        else:
            weights = weights.mean(0)

        if self.weight_normalizers is None:
            self.normalize_weights()
        runs = list(set([(b.run, b.subject) for b in self._blocks]))
        subject_run = (self._blocks[block].run, self._blocks[block].subject)
        idnorm, absnorm = self.weight_normalizers[runs.index(subject_run)]

        if uncertainty_opacity:
            alphas = utils.uncertainty_alphas(factors_std_dev.data,
                                              scalars=brain_center_std_dev)
        else:
            alphas = utils.intensity_alphas(torch.abs(weights.data),
                                            normalizer=absnorm)

        palette = utils.scalar_map_palette(weights.data.numpy(), alphas,
                                           colormap, normalizer=idnorm)

        centers_palette = utils.scalar_map_palette(weights.data.numpy(), None,
                                                   colormap, normalizer=idnorm)
        centers_sizes = np.repeat([50], self.num_factors)
        sizes = torch.exp(results['factor_log_widths']).numpy()

        centers = results['factor_centers'].numpy()

        plot = niplot.plot_connectome(
            np.eye(self.num_factors * 2),
            np.vstack([centers, centers]),
            node_color=np.vstack([palette, centers_palette]),
            node_size=np.vstack([sizes, centers_sizes]),
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block]))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0, labeler=None):
        if labeler is None:
            labeler = lambda b: b.task
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        image = utils.cmu2nii(self.voxel_activations[block].numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs,
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block]))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0, labeler=None):
        if labeler is None:
            labeler = lambda b: b.task
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]

        results = self.results(block)

        reconstruction = results['weights']['mu'].data @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs,
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block]))
        )

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(
                (self.voxel_activations[block] - reconstruction).numpy()
            ),
            np.linalg.norm(self.voxel_activations[block].numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def scatter_factor_embedding(self, labeler=None, filename=None, show=True,
                                 xlims=None, ylims=None, figsize=(3.75, 2.75),
                                 colormap='Set1'):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_f = hyperparams['factors']['mu'].data

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = np.unique([l for l in labels if l is not None])
        palette = dict(zip(all_labels,
                           utils.compose_palette(len(all_labels),
                                                 colormap=colormap)))

        subjects = list(set([block.subject for block in self._blocks]))
        z_fs = [z_f[subjects.index(b.subject)] for b in self._blocks
                if labeler(b) is not None]
        z_fs = torch.stack(z_fs)
        block_colors = [palette[labeler(b)] for b in self._blocks
                        if labeler(b) is not None]

        fig = plt.figure(1, figsize=figsize)
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

    def scatter_subject_embedding(self, labeler=None, filename=None, show=True,
                                  xlims=None, ylims=None, figsize=(3.75, 2.75),
                                  colormap='Set1'):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_p = hyperparams['subject']['mu'].data

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = np.unique([l for l in labels if l is not None])
        palette = dict(zip(all_labels,
                           utils.compose_palette(len(all_labels),
                                                 colormap=colormap)))

        subjects = list(set([block.subject for block in self._blocks]))
        z_ps = torch.stack(
            [z_p[subjects.index(b.subject)] for b in self._blocks
             if labeler(b) is not None]
        )
        block_colors = [palette[labeler(b)] for b in self._blocks
                        if labeler(b) is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^P_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^P_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Participant Embeddings')
        ax.scatter(x=z_ps[:, 0], y=z_ps[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            fig.show()

    def scatter_task_embedding(self, t=None, labeler=None, filename=None,
                               show=True, xlims=None, ylims=None,
                               figsize=(3.75, 2.75),
                               colormap='Set1'):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_s = hyperparams['task']['mu'].data
        if t is not None:
            z_s = z_s[t]
        else:
            z_s = z_s.mean(1)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = np.unique([l for l in labels if l is not None])
        palette = dict(zip(all_labels,
                           utils.compose_palette(len(all_labels),
                                                 colormap=colormap)))

        tasks = list(set([block.task for block in self._blocks]))
        z_ss = torch.stack(
            [z_s[tasks.index(b.task)] for b in self._blocks
             if labeler(b) is not None]
        )
        block_colors = [palette[labeler(b)] for b in self._blocks
                        if labeler(b) is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^S_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^S_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Stimulus Embeddings')
        ax.scatter(x=z_ss[:, 0], y=z_ss[:, 1], c=block_colors)
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
