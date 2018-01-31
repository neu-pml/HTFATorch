from scipy.stats import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import probtorch
import scipy.io as sio
import numpy as np

CUDA =torch.cuda.is_available()
NUM_EPOCH = 10 #placeholder values
NUM_BATCH = 45 #placeholder values
NUM_SAMPLES = 10 #placeholder values
LEARNING_RATE = 0.1

dataset = sio.loadmat('subj1.mat')
voxelActivations = torch.Tensor(dataset['data']) #data
voxelLocations = torch.Tensor(dataset['R'])  #voxel locations
del dataset
NUM_TRs = voxelActivations.shape[1]     #no.of TRs
NUM_VOXELS = voxelLocations.shape[0]     #no. of voxels
NUM_FACTORS = 50             #no. of factors


c = torch.mean(voxelLocations,0)#c
brainCenter = c.unsqueeze(0)
# brainCenter = torch.Tensor([[c[0]],[c[1]],[c[2]]])
sourceCenterVariance = torch.Tensor([1,1,1]).unsqueeze(0) #kappa_u
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
        data = p.normal(torch.matmul(Weights,Factors),self.Snoise.unsqueeze(0).expand(NUM_SAMPLES,NUM_TRs,NUM_VOXELS),value = data,name = 'Y')
        # p.loss(((Yhat - data).T)*(Yhat-data),name='y') ##do we need this?

        return p

#
# 3 x V
# 3 x K
#
# 3 x V x 1
# 3 x 1 x K
# 1 x 1 x K

def Function_RBF(locations, centers, distances):
    locations = locations.unsqueeze(0)
    locations = locations.expand(10,NUM_VOXELS,3)
    return torch.exp((((locations.unsqueeze(1) - centers.unsqueeze(2))**2).sum(3))/(-torch.exp(distances.unsqueeze(2))))

def elbo(q,p,NUM_SAMPLES = 10):
    return probtorch.objectives.montecarlo.kl(q,p,sample_dim=NUM_SAMPLES)   #negative taken later

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
        data = Variable(data)
        R = Variable(R)
        p = dec(data = data,R =R, q = q)
        loss = -elbo(p, q)
        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
        losses[n] = loss.data.numpy()[0]
    return losses


losses = train(voxelActivations,voxelLocations,enc,dec,optimizer,num_steps=10)

#
# for e in range(NUM_EPOCH):
#     train_data = list(voxelActivations,voxelLocations)
#     train_elbo = train(train_data,enc,dec)
#     #test_elbo = test(test_data,enc,dec)
#
#     print('[Epoch %d] Train: ELBO %.4e Test: ELBO %.4e' % (e, train_elbo))

if CUDA:
    q = enc()
    W = q['Weights'].value.data.cpu().numpy()
    M = q['FactorCenters'].value.data.cpu().numpy()
    L = q['FactorWidths'].value.data.cpud().numpy()






