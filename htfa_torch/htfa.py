import numpy as np
import torch
import probtorch
import torch.nn as nn

class HTFAEnconder(nn.module):
    """variational guide for HTFA"""
    def __init__(self,num_factors,num_subjects):
        self.mean_mean_weight = Parameter(torch.zeros(num_factors,))
        self.std_mean_weight = Parameter(torch.rand(num_factors,))
        self.mean_std_weight = Parameter(torch.zeros(num_factors,))
        self.std_std_weight = Parameter(torch.rand(num_factors,))
        
        self.mean_mean_factor = Parameter(torch.zeros(num_factors,3))
        self.std_mean_factor = Parameter(torch.rand(num_factors,3))
        self.mean_width_factor = Parameter(torch.zeros(num_factors,3))
        self.std_width_factor = Parameter(torch.rand(num_factors,3))
        
        self.mean_weight = Parameter(torch.zeros(num_factors,num_subjects))
        self.std_weight = Parameter(torch.rand(num_factors,num_subjects))
        
        
        
        
    def forward(self):
        q = probtorch.Trace()
        mean_weight = self.mean_weight
        std_weight =  self.std_weight
        mean_mean_factor = self.mean_mean_factor
        std_mean_factor = self.std_mean_factor
        mean_width_factor = self.mean_width_factor
        std_width_factor = self.std_width_factor
        
        q.normal(mean_weight.expand(num_times,num_factors,num_subjects),
                         std_weight.expand(num_times,num_factors,num_subjects),
                         name='weights')
        q.normal(mean_mean_factor.expand(num_factors,3,num_subjects),
                              std_mean_factor.expand(num_factors,3,num_subjects),
                              name='mean_factor')
        q.normal(mean_width_factor.expand(num_factors,3,num_subjects),
                               std_width_factor.expand(num_factors,3,num_subjects),
                               name='width_factor')
        
        return q


class HTFADecoder(nn.Module):
    """Generative Model for HTFA"""
    def __init__(self,num_subjects,num_times,num_voxels,num_factors = NUM_FACTORS):
        self._num_subjects = num_subjects
        self._num_factors = num_factors
        self._num_voxels = num_voxels
        
        ## outermost layer of hyperparameters, values are placeholders, dimensions should be correct
        ## dimensions K,
        self.register_buffer('hyper_mean_mean_mean_weight',Variable(torch.zeros(self._num_factors,)))
        self.register_buffer('hyper_std_mean_mean_weight',Variable(0.5 * torch.ones(self._num_factors,)))
        self.register_buffer('hyper_mean_std_mean_weight',Variable(torch.zeros(self._num_factors,)))
        self.register_buffer('hyper_std_std_mean_weight',Variable(0.5 * torch.ones(self._num_factors,)))
        
        self.register_buffer('hyper_mean_mean_std_weight',Variable(torch.zeros(self._num_factors,)))
        self.register_buffer('hyper_std_mean_std_weight',Variable(0.5 * torch.ones(self._num_factors,)))
        self.register_buffer('hyper_mean_std_std_weight',Variable(torch.zeros(self._num_factors,)))
        self.register_buffer('hyper_std_std_std_weight',Variable(0.5 * torch.ones(self._num_factors,)))
        
        ## dimensions K,3
        self.register_buffer('hyper_mean_mean_mean_factor',Variable(brain_center.expand
                                                                      (self._num_factors,3)))
        self.register_buffer('hyper_std_mean_mean_factor',Variable(0.5 * torch.ones
                                                                     (self._num_factors,3)))
        self.register_buffer('hyper_mean_std_mean_factor',Variable(brain_center_std_dev.expand
                                                                     (self._num_factors,3)))
        self.register_buffer('hyper_std_std_mean_factor',Variable(0.5 * torch.ones
                                                                    (self._num_factors,3)))
        
        ## dimensions K,3        
        self.register_buffer('hyper_mean_mean_width_factor',Variable(brain_center.expand
                                                                      (self._num_factors,3)))
        self.register_buffer('hyper_std_mean_width_factor',Variable(0.5 * torch.ones
                                                                     (self._num_factors,3)))
        self.register_buffer('hyper_mean_std_width_factor',Variable(brain_center_std_dev.expand
                                                                     (self._num_factors,3)))
        self.register_buffer('hyper_std_std_width_factor',Variable(0.5 * torch.ones
                                                                    (self._num_factors,3)))
        
        
        
    def forward(self):
        
        p = probtorch.Trace()
        
        #second from outermost layer
        mean_mean_weight = p.normal(self.hyper_mean_mean_mean_weight,
                                    self.hyper_std_mean_mean_weight)
        std_mean_weight = p.normal(self.hyper_mean_std_mean_weight,
                                   self.hyper_std_std_mean_weight)
        mean_std_weight = p.normal(self.hyper_mean_mean_std_weight,
                                  self.hyper_std_mean_std_weight)
        std_std_weight = p.normal(self.hyper_mean_std_std_weight,
                                 self.hyper_std_std_std_weight)
        
        mean_mean_factor = p.normal(self.hyper_mean_mean_mean_factor,
                                   self.hyper_std_mean_mean_factor)
        std_mean_factor = p.normal(self.hyper_mean_std_mean_factor,
                                  self.hyper_std_std_mean_factor)
        mean_width_factor = p.normal(self.hyper_mean_mean_width_factor,
                                  self.hyper_std_mean_width_factor)
        std_width_factor = p.normal(self.hyper_mean_std_width_factor,
                                 self.hyper_std_std_width_factor)
        #third from outermost layer
        ## dimensions K,S
        mean_weight = p.normal(mean_mean_weight.expand(num_factors,num_subjects),
                               std_mean_weight(num_factors,num_subjects))
        std_weight = p.normal(mean_std_weight.expand(num_factors,num_subjects),
                             std_std_weight.expand(num_factors,num_subjects))
        ## dimensions K,3,S
        mean_factor = p.normal(mean_mean_factor.expand(num_factors,3,num_subjects),
                              std_mean_factor.expand(num_factors,3,num_subjects),
                              value=q['mean_factor'],
                              name='mean_factor')
        width_factor = p.normal(mean_width_factor.expand(num_factors,3,num_subjects),
                               std_width_factor.expand(num_factors,3,num_subjects),
                               value=q['width_factor'],
                               name='width_factor')
        
        #fourth from outermost layer
        ##dimensions N,K,S
        weights = p.normal(mean_weight.expand(num_times,num_factors,num_subjects),
                         std_weight.expand(num_times,num_factors,num_subjects),
                         value=q['weights'],
                         name='weights')
        ##dimensions K,V,S
        factors = radial_basis(locations,mean_factors,mean_widths,num_voxels)
        
        #joint
        ##dimensions N,V
        p.normal(torch.matmul(weights,factors), self._voxel_noise,
                value=activations, name='Y')
        
        return p
        