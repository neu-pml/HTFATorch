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
from . import tfa_models
from . import utils

class HierarchicalTopographicFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_files, num_factors=tfa_models.NUM_FACTORS):
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

        self.enc = htfa_models.HTFAGuide(self.voxel_activations,
                                         self.voxel_locations,
                                         self.num_factors)
        self.dec = htfa_models.HTFAModel(self.voxel_locations, self.num_subjects,
                                         self.num_times, self.num_factors)
        if tfa.CUDA:
            self.enc = torch.nn.DataParallel(self.enc)
            self.dec = torch.nn.DataParallel(self.dec)

    def train(self, num_steps=10, learning_rate=LEARNING_RATE,
              log_level=logging.WARNING, batch_size=64,
              num_samples=tfa_models.NUM_SAMPLES):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

    def results(self):
        """Return the inferred parameters"""
        pass

    def save(self, out_dir='.'):
        '''Save a HierarchicalTopographicFactorAnalysis'''
        with open(out_dir + '/' + self._name + '.htfa', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        '''Load a saved HierarchicalTopographicFactorAnalysis from a file'''
        with open(filename, 'rb') as file:
            return pickle.load(file)
