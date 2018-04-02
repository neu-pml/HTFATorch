"""Sketch of Deep TFA architecture"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

from collections import defaultdict
import torch
import probtorch

# NOTE: I am writing this as a model relative to PyTorch master, 
# which no longer requires explicit wrapping in Variable(...)

class DeepTFA(torch.nn.Module):
    def __init__(self, N=50, T=200, D=2, E=2, K=24):
        # generative model
        self.p_z_w_mean = torch.zeros(D)
        self.p_z_w_std = torch.ones(D)
        self.w = torch.nn.Sequential(
                    torch.nn.Linear(D, K/2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(K/2, K))
        self.q_z_f_mean = torch.zeros(E)
        self.q_z_f_std = torch.ones(E)
        self.h_f = torch.nn.Sequential(
                        torch.nn.Linear(E, K/2),
                        torch.nn.ReLU())
        self.x_f = torch.nn.Linear(K/2, 3*K)
        self.log_rho_f = torch.nn.Linear(K/2, K)
        self.sigma_y = Parameter(1.0)
        # variational parameters
        self.q_z_w_mean = Parameter(torch.zeros(N, D))
        self.q_z_w_std = Parameter(torch.ones(N, D))
        self.q_z_f_mean = Parameter(torch.zeros(N, T, E))
        self.q_z_f_std = Parameter(torch.ones(N, T, E))

    def forward(self, x, y, n, t):
        p = probtorch.Trace()
        q = probtorch.Trace()
        z_w = q.normal(self.q_z_w_mean[n, t],
                       self.q_z_w_std[n, t],
                       name='z_w')
        z_w = p.normal(self.p_z_w_mean,
                       self.p_z_w_std,
                       name='z_w')
        w = self.w(z_w)
        z_f = q.normal(self.q_z_w_mean[n],
                       self.q_z_w_std[n],
                       name='z_f')
        z_f = p.normal(self.z_f_mean,
                       self.z_f_std,
                       value=q['z_f']
                       name='z_f')
        x_f = self.x_f(z_f)
        rho_f = torch.exp(self.log_rho_f(z_f))
        f = rbf(x, x_f, rho_f)
        y = p.normal(w * f, 
                     self.sigma_y, 
                     value='y', 
                     name='y')
        return p, q
