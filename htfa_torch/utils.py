#!/usr/bin/env python
"""Utilities for topographic factor analysis"""

__author__ = 'Eli Sennesh'
__email__ = 'e.sennesh@northeastern.edu'

import flatdict
import math
import os
import warnings

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.cm as cm
    import matplotlib.colors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as sd
import scipy.special as spspecial
import scipy.stats as stats
from sklearn.cluster import KMeans
import torch
from torch import distributions
import probtorch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import nibabel as nib
from nilearn.input_data import NiftiMasker

def brain_centroid(locations):
    brain_center = torch.mean(locations, 0).unsqueeze(0)
    brain_center_std_dev = torch.sqrt(
        torch.var(locations, 0).unsqueeze(0)
    )
    return brain_center, brain_center_std_dev

def initial_radial_basis(location, center, widths):
    """The radial basis function used as the shape for the factors"""
    # V x 3 -> 1 x V x 3
    location = np.expand_dims(location, 0)
    # K x 3 -> K x 1 x 3
    center = np.expand_dims(center, 1)
    #
    delta2s = (location - center) ** 2
    widths = np.expand_dims(widths, 1)
    return np.exp(-delta2s.sum(2) / (widths))

def initial_hypermeans(activations, locations, num_factors):
    """Initialize our center, width, and weight parameters via K-means"""
    kmeans = KMeans(init='k-means++',
                    n_clusters=num_factors,
                    n_init=10,
                    random_state=100)
    kmeans.fit(locations)
    initial_centers = kmeans.cluster_centers_
    initial_widths = 2.0 * math.pow(np.nanmax(np.std(locations, axis=0)), 2)
    F = initial_radial_basis(locations, initial_centers, initial_widths)
    F = F.T

    # beta = np.var(voxel_activations)
    trans_F = F.T.copy()
    initial_weights = np.linalg.solve(trans_F.dot(F), trans_F.dot(activations))

    return initial_centers, np.log(initial_widths), initial_weights.T

def plot_losses(losses):
    epochs = range(losses.shape[1])

    free_energy_fig = plt.figure(figsize=(10, 10))

    plt.plot(epochs, losses[0, :], 'b-', label='Data')
    plt.legend()

    free_energy_fig.tight_layout()
    plt.title('Free-energy / -ELBO change over training')
    free_energy_fig.axes[0].set_xlabel('Epoch')
    free_energy_fig.axes[0].set_ylabel('Free-energy / -ELBO (nats)')

    plt.show()

def full_fact(dimensions):
    """
    Replicates MATLAB's fullfact function (behaves the same way)
    """
    vals = np.asmatrix(range(1, dimensions[0] + 1)).T
    if len(dimensions) == 1:
        return vals
    else:
        after_vals = np.asmatrix(full_fact(dimensions[1:]))
        independents = np.asmatrix(np.zeros((np.prod(dimensions), len(dimensions))))
        row = 0
        for i in range(after_vals.shape[0]):
            independents[row:(row + len(vals)), 0] = vals
            independents[row:(row + len(vals)), 1:] = np.tile(
                after_vals[i, :], (len(vals), 1)
            )
            row += len(vals)
        return independents

def nii2cmu(nifti_file, mask_file=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = nib.load(nifti_file)
        mask = NiftiMasker(mask_strategy='background')
        if mask_file is None:
            mask.fit(nifti_file)
        else:
            mask.fit(mask_file)

    header = image.header
    sform = image.get_sform()
    voxel_size = header.get_zooms()

    voxel_activations = np.float64(mask.transform(nifti_file)).transpose()

    vmask = np.nonzero(np.array(np.reshape(
        mask.mask_img_.dataobj,
        (1, np.prod(mask.mask_img_.shape)),
        order='C'
    )))[1]
    voxel_coordinates = full_fact(image.shape[0:3])[vmask, ::-1] - 1
    voxel_locations = np.array(np.dot(voxel_coordinates, sform[0:3, 0:3])) + sform[:3, 3]

    return {'data': voxel_activations, 'R': voxel_locations}

def cmu2nii(activations, locations, template):
    image = nib.load(template)
    sform = image.affine
    coords = np.array(
        np.dot(locations - sform[:3, 3],
               np.linalg.inv(sform[0:3, 0:3])),
        dtype='int'
    )
    data = np.zeros(image.shape[0:3] + (activations.shape[0],))

    for i in range(activations.shape[0]):
        for j in range(locations.shape[0]):
            x, y, z = coords[j, 0], coords[j, 1], coords[j, 2]
            data[x, y, z, i] = activations[i, j]

    return nib.Nifti1Image(data, affine=sform)

def load_collective_dataset(data_files, mask):
    datasets = [list(load_dataset(data, mask=mask)) for data in data_files]
    min_time = min([dataset[0].shape[0] for dataset in datasets])
    for dataset in datasets:
        difference = dataset[0].shape[0] - min_time
        if difference > 0:
            start_cut = difference // 2
            end_cut = difference - start_cut
            length = dataset[0].shape[0]
            dataset[0] = dataset[0][start_cut:length-end_cut, :]

    activations = [d[0] for d in datasets]
    locations = [d[1] for d in datasets]
    names = [d[2] for d in datasets]
    templates = [d[3] for d in datasets]

    for d in datasets:
        acts = d[0]
        locs = d[1]
        del acts
        del locs

    return activations, locations, names, templates

def load_dataset(data_file, mask=None, zscore=True):
    name, ext = os.path.splitext(data_file)
    if ext == 'mat':
        dataset = sio.loadmat(data_file)
        template = None
    else:
        dataset = nii2cmu(data_file, mask_file=mask)
        template = data_file
    _, name = os.path.split(name)
    # pull out the voxel activations and locations
    if zscore:
        data = stats.zscore(dataset['data'], axis=1, ddof=1)
    else:
        data = dataset['data']
    activations = torch.Tensor(data).t()
    locations = torch.Tensor(dataset['R'])

    del dataset

    return activations, locations, name, template

def generate_group_activities(group_data,window_size = 10):
    """
    :param group_data: n_subjects x n_times x n_voxels (or factors) activation data for all subjects
    :return: activation_vectors: times (depending on window size) x n_voxels (or factors) vectors of mean activation
    """
    n_times = group_data.shape[1]
    n_nodes = group_data.shape[2]
    n_windows = n_times-window_size+1
    activation_vectors = np.empty(shape = (n_windows,n_nodes))
    for w in range(0,n_windows):
        window = group_data[:,w:w+window_size,:]
        activation_vectors[w,:] = window.numpy().mean(axis=(0,1))

    return activation_vectors


def get_covariance(group_data, window_size=5):
    """
    :param data: n_subjects x n_times x n_nodes
    :param windowsize: number of observations to include in each sliding window (set to 0 or don't specify if all
                           timepoints should be used)
    :return: n_subjets x number-of-features by number-of-features covariance matrix
    """
    n_times = group_data.shape[1]
    n_nodes = group_data.shape[2]
    n_windows = n_times - window_size + 1
    cov = np.empty(shape=(n_windows, n_nodes, n_nodes))
    for w in range(0, n_windows):
        window = group_data[:, w:w + window_size, :]
        window = np.mean(window,axis=1)
        cov[w, :, :] = np.cov(window.T)

    return cov

def calculate_kl(mean1,cov1,mean2,cov2):
    cov1 = cov1 + 1e-3*np.eye(cov1.shape[0])
    cov2 = cov2 + 1e-3*np.eye(cov2.shape[0])
    d = len(mean1)
    cov2inv = np.linalg.inv(cov2)
    kl  = np.log(np.linalg.det(cov2)) - np.log(np.linalg.det(cov1)) - d + np.trace(np.dot(cov2inv,cov1)) + \
          (mean2-mean1).T.dot(cov2inv).dot(mean2-mean1)
    return kl/2
def get_correlation_matrix(pattern_G1,pattern_G2):

    activity_correlation_matrix = np.empty((pattern_G1.shape[0], pattern_G2.shape[0]))
    for i in range(pattern_G1.shape[0]):
        for j in range(pattern_G2.shape[0]):
            activity_correlation_matrix[i, j] = stats.pearsonr(pattern_G1[i], pattern_G2[j])[0]

    return activity_correlation_matrix

def get_decoding_accuracy(G1,G2,window_size=5,hist=True):
    """
    :param G1: Split Half Group G1 (group_size x n_times x n_nodes)
    :param G2: Split Half Group G2 (group_size x n_times x n_nodes
    :return: time labels of G1 as predicted by max corr with G2
    """
    activity_pattern_G1 = generate_group_activities(torch.Tensor(G1), window_size=window_size)
    activity_pattern_G2 = generate_group_activities(torch.Tensor(G2), window_size=window_size)
    activity_correlation_matrix = get_correlation_matrix(activity_pattern_G1,activity_pattern_G2)
    time_labels = np.argmax(activity_correlation_matrix, axis=1)
    decoding_accuracy=[]
    if hist:
        for i in range(5):
            temp = np.sum(time_labels+i == np.arange(activity_pattern_G1.shape[0]))
            decoding_accuracy.append(temp)
            if i!=0:
                temp = np.sum(time_labels-i == np.arange(activity_pattern_G1.shape[0]))
                decoding_accuracy.append(temp)
    else:
        decoding_accuracy = np.sum(time_labels == np.arange(activity_pattern_G1.shape[0]))
    decoding_accuracy = np.array(decoding_accuracy)/activity_pattern_G1.shape[0]

    return decoding_accuracy,activity_correlation_matrix

def get_isfc_decoding_accuracy(G1,G2,window_size=5,hist=True):
    """
    :param G1: Split Half Group G1 (group_size x n_times x n_nodes)
    :param G2: Split Half Group G2 (group_size x n_times x n_nodes
    :return: time labels of G1 as predicted by max corr with G2
    """
    isfc_pattern_G1 = dynamic_ISFC(G1, windowsize=window_size)
    isfc_pattern_G2 = dynamic_ISFC(G2, windowsize = window_size)
    activity_correlation_matrix = get_correlation_matrix(isfc_pattern_G1,isfc_pattern_G2)
    time_labels = np.argmax(activity_correlation_matrix, axis=1)
    decoding_accuracy = []
    if hist:
        for i in range(5):
            temp = np.sum(time_labels+i == np.arange(isfc_pattern_G1.shape[0]))
            decoding_accuracy.append(temp)
            if i!=0:
                temp = np.sum(time_labels-i == np.arange(isfc_pattern_G1.shape[0]))
                decoding_accuracy.append(temp)
    else:
        decoding_accuracy = np.sum(time_labels == np.arange(isfc_pattern_G1.shape[0]))
    decoding_accuracy = np.array(decoding_accuracy)/isfc_pattern_G1.shape[0]

    return decoding_accuracy,activity_correlation_matrix


def get_mixed_decoding_accuracy(isfc_correlation_matrix,activity_correlation_matrix,mixing_prop=0.5,hist=True):
    """
    :param G1: Split Half Group G1 (group_size x n_times x n_nodes)
    :param G2: Split Half Group G2 (group_size x n_times x n_nodes
    :return: time labels of G1 as predicted by max corr with G2
    """
    activity_correlation_matrix = mixing_prop*activity_correlation_matrix +\
                                  (1-mixing_prop)*isfc_correlation_matrix
    time_labels = np.argmax(activity_correlation_matrix, axis=1)
    decoding_accuracy = []
    if hist:
        for i in range(5):
            temp = np.sum(time_labels+i == np.arange(activity_correlation_matrix.shape[0]))
            decoding_accuracy.append(temp)
            if i!=0:
                temp = np.sum(time_labels-i == np.arange(activity_correlation_matrix.shape[0]))
                decoding_accuracy.append(temp)
    else:
        decoding_accuracy = np.sum(time_labels == np.arange(activity_correlation_matrix.shape[0]))
    decoding_accuracy = np.array(decoding_accuracy)/activity_correlation_matrix.shape[0]

    return decoding_accuracy

def get_kl_decoding_accuracy(G1, G2, window_size=5,hist=True):
    means_G1 = generate_group_activities(torch.Tensor(G1), window_size=window_size)
    means_G2 = generate_group_activities(torch.Tensor(G2), window_size=window_size)
    cov_G1 = get_covariance(G1, window_size=window_size)
    cov_G2 = get_covariance(G2, window_size=window_size)
    kl = np.empty(shape=(means_G1.shape[0],means_G2.shape[0]))
    for t in range(means_G1.shape[0]):
        for t2 in range(means_G2.shape[0]):
            kl[t,t2] = calculate_kl(means_G1[t,:],cov_G1[t,:,:],means_G2[t2,:],cov_G2[t2,:,:])
    time_labels = np.argmin(kl, axis=1)
    decoding_accuracy = []
    if hist:
        for i in range(5):
            temp = np.sum(time_labels+i == np.arange(means_G1.shape[0]))
            decoding_accuracy.append(temp)
            if i!=0:
                temp = np.sum(time_labels-i == np.arange(means_G1.shape[0]))
                decoding_accuracy.append(temp)
    else:
        decoding_accuracy = np.sum(time_labels == np.arange(means_G1.shape[0]))
    decoding_accuracy = np.array(decoding_accuracy)/means_G1.shape[0]

    return decoding_accuracy


def dynamic_ISFC(data, windowsize=5):
        """
        :param data: n_subjects x n_times x n_nodes
        :param windowsize: number of observations to include in each sliding window (set to 0 or don't specify if all
                           timepoints should be used)
        :return: number-of-features by number-of-features isfc matrix
        reference: http://www.nature.com/articles/ncomms12141
        code based on https://github.com/brainiak/brainiak/blob/master/examples/factoranalysis/htfa_tutorial.ipynb
        """
        def r2z(r):
            return 0.5 * (np.log(1 + r) - np.log(1 - r))

        def z2r(z):
            return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

        def vectorize(m):
            np.fill_diagonal(m, 0)
            return sd.squareform(m,checks=False)
        assert len(data) > 1

        ns = data.shape[1]
        vs = data.shape[2]

        n = np.min(ns)
        if windowsize == 0:
            windowsize = n

        assert len(np.unique(vs)) == 1
        v = vs

        isfc_mat = np.zeros([ns - windowsize + 1, int((v ** 2 - v) / 2)])
        for n in range(0, ns - windowsize + 1):
            next_inds = range(n, n + windowsize)
            for i in range(0, data.shape[0]):
                mean_other_data = np.zeros([len(next_inds), v])
                for j in range(0, data.shape[0]):
                    if i == j:
                        continue
                    mean_other_data = mean_other_data + data[j,next_inds, :]
                mean_other_data /= (data.shape[0] - 1)
                next_corrs = np.array(r2z(1 - sd.cdist(data[i,next_inds, :].T, mean_other_data.T, 'correlation')))
                isfc_mat[n, :] = isfc_mat[n, :] + vectorize(next_corrs + next_corrs.T)
            isfc_mat[n, :] = z2r(isfc_mat[n, :] / (2 * data.shape[0]))

        isfc_mat[np.where(np.isnan(isfc_mat))] = 0
        return isfc_mat

def vardict(existing=None):
    vdict = flatdict.FlatDict(delimiter='__')
    if existing:
        for (k, v) in existing.items():
            vdict[k] = v
    return vdict

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

def register_vardict(vdict, module, parameter=True):
    for (k, v) in vdict.iteritems():
        if parameter:
            module.register_parameter(k, Parameter(v))
        else:
            module.register_buffer(k, v)

def unsqueeze_and_expand(tensor, dim, size, clone=False):
    if clone:
        tensor = tensor.clone()

    shape = list(tensor.shape)
    shape.insert(dim, size)
    return tensor.unsqueeze(dim).expand(*shape)

def unsqueeze_and_expand_vardict(vdict, dim, size, clone=False):
    result = vardict(vdict)

    for (k, v) in result.iteritems():
        result[k] = unsqueeze_and_expand(v, dim, size, clone)

    return result

def populate_vardict(vdict, populator, *dims):
    for k in vdict.iterkeys():
        vdict[k] = populator(*dims)
    return vdict

def gaussian_populator(*dims):
    return {
        'mu': torch.zeros(*dims),
        'sigma': torch.ones(*dims)
    }

def uncertainty_alphas(uncertainties, scalars=None):
    return np.float64(1.0) - intensity_alphas(uncertainties, scalars)

def normalize_tensors(seq, absval=False, percentiles=None):
    flat = torch.cat([t.view(-1) for t in seq], dim=0)
    if absval:
        flat = torch.abs(flat)
    flat = flat.numpy()
    if percentiles is not None:
        left, right = percentiles
        result = matplotlib.colors.Normalize(np.percentile(flat, left),
                                             np.percentile(flat, right),
                                             clip=True)
    else:
        result = matplotlib.colors.Normalize(clip=True)
        result.autoscale_None(flat)

    return result

def intensity_alphas(intensities, scalars=None, normalizer=None):
    if scalars is not None:
        intensities = intensities / scalars
    if len(intensities.shape) > 1:
        intensities = intensities.norm(p=2, dim=1)
    if normalizer is None:
        normalizer = matplotlib.colors.Normalize()
    result = normalizer(intensities.numpy())
    if normalizer.clip:
        result = np.clip(result, 0.0, 1.0)
    return result

def scalar_map_palette(scalars, alphas=None, colormap='Set2', normalizer=None):
    scalar_map = cm.ScalarMappable(normalizer, colormap)
    colors = scalar_map.to_rgba(scalars, norm=True)
    if alphas is not None:
        colors[:, 3] = alphas
    return colors

def compose_palette(length, alphas=None, colormap='Set2'):
    return scalar_map_palette(np.linspace(0, 1, length), alphas, colormap)

def uncertainty_palette(uncertainties, scalars=None, colormap='Set2'):
    alphas = uncertainty_alphas(uncertainties, scalars=scalars)
    return compose_palette(uncertainties.shape[0], alphas=alphas,
                           colormap=colormap)

def palette_legend(labels, colors):
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in
               range(len(colors))]
    plt.legend(handles=patches)

def isnan(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return tensor != tensor

def hasnan(tensor):
    return isnan(tensor).any()

class TFADataset(torch.utils.data.Dataset):
    def __init__(self, activations):
        self._activations = activations
        self._num_subjects = len(self._activations)
        self.num_times = min([acts.shape[0] for acts in self._activations])

    def __len__(self):
        return self.num_times

    def __getitem__(self, i):
        return torch.stack([acts[i] for acts in self._activations], dim=0)

def chunks(chunkable, n):
    for i in range(0, len(chunkable), n):
        yield chunkable[i:i+n]

def inverse(func, args):
    result = {}
    for arg in args:
        val = func(arg)
        if val in result:
            result[val] += [arg]
        else:
            result[val] = [arg]
    return result
