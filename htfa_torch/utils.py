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
    import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

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

    return {'data': voxel_activations, 'R': voxel_locations}, image

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

def load_dataset(data_file, mask=None):
    name, ext = os.path.splitext(data_file)
    if ext == '.nii':
        dataset, image = nii2cmu(data_file, mask_file=mask)
        template = data_file
    else:
        dataset = sio.loadmat(data_file)
        image = template = None
    _, name = os.path.split(name)
    # pull out the voxel activations and locations
    zscored_data = stats.zscore(dataset['data'], axis=1, ddof=1)
    activations = torch.Tensor(zscored_data).t()
    locations = torch.Tensor(dataset['R'])

    del dataset

    return activations, image, locations, name, template

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
            module.register_buffer(k, Variable(v))

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

def uncertainty_palette(uncertainties):
    if len(uncertainties.shape) > 1:
        uncertainties = np.array([
            [u] for u in np.linalg.norm((uncertainties**-2).numpy(), axis=1)
        ])
    else:
        uncertainties = uncertainties.numpy()
    uncertainties = uncertainties / (1.0 + uncertainties)
    palette = np.array(sns.color_palette("RdBu", uncertainties.shape[0]))
    return np.concatenate([palette, uncertainties], axis=1)

def isnan(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return tensor != tensor

def hasnan(tensor):
    return isnan(tensor).any()
