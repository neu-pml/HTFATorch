import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import probtorch
import scipy.io as sio
import numpy as np

CUDA = torch.cuda.is_available()

NUM_SAMPLES = 100  # placeholder values
LEARNING_RATE = 1e-7

dataset = sio.loadmat('s0.mat')
voxelActivations = torch.Tensor(dataset['data'])  # data
voxelActivations = torch.transpose(voxelActivations, 0, 1)
voxelLocations = torch.Tensor(dataset['R'])   # voxel locations
del dataset
NUM_TRs = voxelActivations.shape[0]  # no.of TRs
NUM_VOXELS = voxelActivations.shape[1]  # no. of voxels
NUM_FACTORS = 50  # no. of factors

brainCenter = torch.mean(voxelLocations, 0).unsqueeze(0)
sourceCenterVariance = 10 * torch.var(voxelLocations, 0).unsqueeze(0)  # kapp_u
sourceWeightVariance = 2  # kappa_w
sourceWidthVariance = 3
voxelNoise = 0.1  # sigma2_y


class Encoder(nn.Module):
    def __init__(self, NUM_TRs=NUM_TRs, NUM_FACTORS=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self.MeanWeight = Parameter(torch.randn((NUM_TRs, NUM_FACTORS)))
        self.SigmaWeight = Parameter(torch.randn((NUM_TRs, NUM_FACTORS)))
        self.MeanFactorCenter = Parameter(torch.randn((NUM_FACTORS, 3)))
        self.SigmaFactorCenter = Parameter(torch.randn((NUM_FACTORS, 3)))
        self.MeanLogFactorWidth = Parameter(torch.randn((NUM_FACTORS)))
        self.SigmaLogFactorWidth = Parameter(torch.randn((NUM_FACTORS)))

    def forward(self, NUM_SAMPLES=NUM_SAMPLES):
        q = probtorch.Trace()
        MeanWeight = self.MeanWeight.expand(NUM_SAMPLES, NUM_TRs, NUM_FACTORS)
        SigmaWeight = self.SigmaWeight.expand(NUM_SAMPLES, NUM_TRs, NUM_FACTORS)
        MeanFactorCenter = self.MeanFactorCenter.expand(NUM_SAMPLES, NUM_FACTORS, 3)
        SigmaFactorCenter = self.SigmaFactorCenter.expand(NUM_SAMPLES, NUM_FACTORS, 3)
        MeanLogFactorWidth = self.MeanLogFactorWidth.expand(NUM_SAMPLES, NUM_FACTORS)
        SigmaLogFactorWidth = self.SigmaLogFactorWidth.expand(NUM_SAMPLES, NUM_FACTORS)
        Weights = q.normal(MeanWeight, SigmaWeight, name='Weights')  #W
        FactorCenters = q.normal(MeanFactorCenter, SigmaFactorCenter, name='FactorCenters')  #M
        LogFactorWidths = q.normal(MeanLogFactorWidth, SigmaLogFactorWidth, name='LogFactorWidths')  #L
        return q


class Decoder(nn.Module):
    def __init__(self, NUM_TRs=NUM_TRs, NUM_FACTORS=NUM_FACTORS, NUM_VOXELS=NUM_VOXELS):
        super(self.__class__, self).__init__()
        self.MeanWeight = Variable(torch.zeros(NUM_TRs, NUM_FACTORS))
        self.SigmaWeight = Variable(sourceWeightVariance * torch.ones(NUM_TRs, NUM_FACTORS))
        self.MeanFactorCenter = Variable(brainCenter.expand(NUM_FACTORS, 3)) #c is center of 3D brain image)
        self.SigmaFactorCenter = Variable(sourceCenterVariance.expand(NUM_FACTORS, 3)) #s
        self.MeanLogFactorWidth = Variable(torch.ones(NUM_FACTORS))
        self.SigmaLogFactorWidth = Variable(sourceWidthVariance * torch.ones(NUM_FACTORS))
        self.Snoise = Variable(voxelNoise * torch.ones(NUM_TRs, NUM_VOXELS))

    def forward(self, data=voxelActivations, R=voxelLocations, q=None):
        p = probtorch.Trace()
        Weights = p.normal(self.MeanWeight, 
                           self.SigmaWeight, 
                           value=q['Weights'], 
                           name='Weights')
        FactorCenters = p.normal(self.MeanFactorCenter,
                                 self.SigmaFactorCenter,
                                 value=q['FactorCenters'],
                                 name='FactorCenters')
        LogFactorWidths = p.normal(self.MeanLogFactorWidth, 
                                   self.SigmaLogFactorWidth, 
                                   value=q['LogFactorWidths'],
                                   name='LogFactorWidths')
        Factors = RBF(R, FactorCenters, LogFactorWidths)
        data = p.normal(torch.matmul(Weights, Factors), 
                        self.Snoise,
                        value=data,
                        name = 'Y')
        # p.loss(((Yhat - data).T)*(Yhat-data),name='y') ##do we need this?
        return p


# locations: V x 3
# centers: S x K x 3
# log_widths: S x K
def RBF(locations, centers, log_widths):
    # V x 3 -> S x 1 x V x 3
    locations = locations.expand(NUM_SAMPLES, NUM_VOXELS, 3).unsqueeze(1)
    # S x K x 3 -> S x K x 1 x 3  
    centers = centers.unsqueeze(2)
    # S x K x V x 3
    delta2s = (locations - centers)**2
    # S x K  -> S x K x 1
    log_widths = log_widths.unsqueeze(2)
    return torch.exp(-delta2s.sum(3) / torch.exp(log_widths))

def elbo(q, p):
    return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0)   #negative taken later

enc = Encoder()
dec = Decoder()

if CUDA:
    enc.cuda()
    dec.cuda()

optimizer = torch.optim.Adam(list(enc.parameters()), lr=LEARNING_RATE)

def train(data,R,enc,dec,optimizer,num_steps):
    enc.train()
    dec.train()
    losses = np.zeros(num_steps)
    for n in range(num_steps):
        optimizer.zero_grad()
        q = enc()
        if CUDA:
            data = data.cuda()
            R = R.cuda()
        p = dec(data = data,R =R, q = q)
        loss = -elbo(q, p)
        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
        losses[n] = loss.data.numpy()[0]
        print (losses[n])
        # if n > 1:
        #     assert(losses[-1] < losses[-2])
    return losses


losses = train(Variable(voxelActivations),
               Variable(voxelLocations),
               enc, dec, 
               optimizer, 
               num_steps=100)

if CUDA:
    q = enc()
    W = q['Weights'].value.data.cpu().numpy()
    M = q['FactorCenters'].value.data.cpu().numpy()
    L = q['LogFactorWidths'].value.data.cpu().numpy()

r =3






