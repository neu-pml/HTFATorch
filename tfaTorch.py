from scipy.stats import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import probtorch
import scipy.io as sio
import numpy as np

CUDA =torch.cuda.is_available()

NUM_SAMPLES = 10 #placeholder values
LEARNING_RATE = 0.1

dataset = sio.loadmat('s0.mat')
voxelActivations = torch.Tensor(dataset['data']) #data
voxelActivations = torch.transpose(voxelActivations,0,1)
voxelLocations = torch.Tensor(dataset['R'])  #voxel locations
del dataset
NUM_TRs = voxelActivations.shape[0]     #no.of TRs
NUM_VOXELS = voxelActivations.shape[1]     #no. of voxels
NUM_FACTORS = 50             #no. of factors


brainCenter = torch.mean(voxelLocations,0).unsqueeze(0)
sourceCenterVariance = 10*torch.var(voxelLocations,0).unsqueeze(0) #kapp_u
sourceWeightVariance = 2 #kappa_w
sourceWidthVariance = 3
voxelNoise = 0.1 #sigma2_y





class Encoder(nn.Module):
    def __init__(self,NUM_TRs=NUM_TRs,NUM_FACTORS=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self.MeanWeight = Parameter(torch.randn((NUM_TRs,NUM_FACTORS)))
        self.SigmaWeight = Parameter(torch.randn((NUM_TRs,NUM_FACTORS)))
        self.MeanFactorCenter = Parameter(torch.randn((NUM_FACTORS,3)))
        self.SigmaFactorCenter = Parameter(torch.randn((NUM_FACTORS,3)))
        self.MeanFactorWidth = Parameter(torch.randn((NUM_FACTORS)))
        self.SigmaFactorWidth = Parameter(torch.randn((NUM_FACTORS)))

    def forward(self,NUM_SAMPLES = NUM_SAMPLES):

        q = probtorch.Trace()
        MeanWeight = self.MeanWeight.expand(NUM_SAMPLES,NUM_TRs,NUM_FACTORS)
        SigmaWeight = self.SigmaWeight.expand(NUM_SAMPLES,NUM_TRs,NUM_FACTORS)
        MeanFactorCenter = self.MeanFactorCenter.expand(NUM_SAMPLES,NUM_FACTORS,3)
        SigmaFactorCenter = self.SigmaFactorCenter.expand(NUM_SAMPLES,NUM_FACTORS,3)
        MeanFactorWidth = self.MeanFactorWidth.expand(NUM_SAMPLES,NUM_FACTORS)
        SigmaFactorWidth = self.SigmaFactorWidth.expand(NUM_SAMPLES,NUM_FACTORS)
        Weights = q.normal(MeanWeight, SigmaWeight, name = 'Weights') #W
        FactorCenters = q.normal(MeanFactorCenter,SigmaFactorCenter, name = 'FactorCenters') #M
        FactorWidths = q.normal(MeanFactorWidth,SigmaFactorWidth,name = 'FactorWidths') #L


        return q

class Decoder(nn.Module):
    def __init__(self,NUM_TRs =NUM_TRs,NUM_FACTORS = NUM_FACTORS,NUM_VOXELS = NUM_VOXELS):
        super(self.__class__, self).__init__()
        self.MeanWeight = Parameter(torch.zeros((NUM_TRs, NUM_FACTORS)))
        self.SigmaWeight = Parameter(sourceWeightVariance*torch.ones((NUM_TRs,NUM_FACTORS)))
        self.MeanFactorCenter = Parameter((brainCenter.expand(NUM_FACTORS,3))*torch.ones((NUM_FACTORS,3))) #c is center of 3D brain image
        self.SigmaFactorCenter = Parameter((sourceCenterVariance.expand(NUM_FACTORS,3))*
                                          torch.ones((NUM_FACTORS,3))) #s
        self.MeanFactorWidth = Parameter(torch.ones((NUM_FACTORS)))
        self.SigmaFactorWidth = Parameter(sourceWidthVariance*torch.ones((NUM_FACTORS)))
        self.Snoise = Parameter(voxelNoise*torch.ones(NUM_TRs,NUM_VOXELS))

    def forward(self,data = voxelActivations,R = voxelLocations,q=None):
        p = probtorch.Trace()
        Weights = p.normal(self.MeanWeight,self.SigmaWeight,value = q['Weights'],name = 'Weights')
        FactorCenters = p.normal(self.MeanFactorCenter,self.SigmaFactorCenter,value = q['FactorCenters'],
                                 name = 'FactorCenters')
        FactorWidths = p.normal(self.MeanFactorWidth,self.SigmaFactorWidth,value = q['FactorWidths'],
                                name = 'FactorWidths')
        Factors = Function_RBF(R,FactorCenters,FactorWidths)
        data = p.normal(torch.matmul(Weights,Factors),self.Snoise,value = data,name = 'Y')
        # p.loss(((Yhat - data).T)*(Yhat-data),name='y') ##do we need this?

        return p



def Function_RBF(locations, centers, distances):
    locations = locations.unsqueeze(0)
    locations = locations.expand(NUM_SAMPLES,NUM_VOXELS,3)
    return torch.exp((((locations.unsqueeze(1) - centers.unsqueeze(2))**2).sum(3))/(-torch.exp(distances.unsqueeze(2))))

def elbo(q,p,NUM_SAMPLES = NUM_SAMPLES):
    return probtorch.objectives.montecarlo.kl(q,p,sample_dim=0)   #negative taken later

enc = Encoder()
dec = Decoder()

if CUDA:
    enc.cuda()
    dec.cuda()

optimizer = torch.optim.Adam(list(enc.parameters()),lr= LEARNING_RATE)

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
    return losses


losses = train(Variable(voxelActivations),Variable(voxelLocations),enc,dec,optimizer,num_steps=10)


if CUDA:
    q = enc()
    W = q['Weights'].value.data.cpu().numpy()
    M = q['FactorCenters'].value.data.cpu().numpy()
    L = q['FactorWidths'].value.data.cpu().numpy()

r =3






