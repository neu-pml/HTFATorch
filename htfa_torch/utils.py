#!/usr/bin/env python
"""Utilities for topographic factor analysis"""

__author__ = 'Eli Sennesh'
__email__ = 'e.sennesh@northeastern.edu'

import warnings

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib
from nilearn.input_data import NiftiMasker

from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

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

    return nib.Nifti1Image(data[:, :, :, 0], affine=sform)

class GaussianTower(nn.Module):
    """A class for handling Gaussians recursively parameterized by other
       Gaussians.  We can treat the leaves as hyperparameters (variational or
       not), and name or clamp any element of the tree, all while maintaining a
       sane naming scheme."""
    def __init__(self, initializers, name=None, levels=1,
                 variational=False):
        super(__class__, self).__init__()
        self.name = name
        self.levels = levels
        if levels == 1:
            if variational:
                self.register_parameter('hyper', Parameter(initializers[0]()))
            else:
                self.register_buffer('hyper', Variable(initializers[0]()))
        else:
            left = initializers[0:len(initializers) // 2]
            right = initializers[len(initializers) // 2:len(initializers)]
            self.mu = GaussianTower(left, name='mu', levels=levels - 1,
                                    variational=variational)
            self.sigma = GaussianTower(right, name='sigma', levels=levels - 1,
                                       variational=variational)

    def forward(self, f, model, clamp):
        if self.levels == 1:
            return f(self.hyper)
        else:
            mu = self.mu(f, model, clamp)
            sigma = self.sigma(f, model, clamp)
            if clamp:
                value = clamp[self.name]
            else:
                value = None
            return model.normal(mu, sigma, name=self.name, value=value)

def batch_weights(weights, start, end, num_samples, num_factors):
    result = weights[start:end, :]
    if num_samples > 1:
        return result.expand(num_samples, end - start, num_factors)
    else:
        return result
