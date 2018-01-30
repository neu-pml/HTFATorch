from scipy.stats import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import probtorch
import scipy.io as sio

CUDA =torch.cuda.is_available()
NUM_EPOCH = 10 #placeholder values
NUM_BATCH = 45 #placeholder values
NUM_SAMPLES = 10 #placeholder values

dataset = sio.loadmat('subj1.mat')
voxelActivations = torch.Tensor(dataset['data']) #data
voxelLocations = torch.Tensor(dataset['R'])  #voxel locations
del dataset
NUM_TRs = voxelActivations.shape[1]     #no.of TRs
NUM_VOXELS = voxelLocations.shape[0]     #no. of voxels
NUM_FACTORS = 50             #no. of factors


brainCenter = torch.mean(voxelActivations,0)  #c
sourceCenterVariance = torch.Tensor([1,1,1]) #kappa_u
sourceWeightVariance = 2 #kappa_w
sourceWidthVariance = 3
voxelNoise = 0.1 #sigma2_y





class Encoder(nn.Module):
    def __init__(self,NUM_TRs=NUM_TRs,NUM_FACTORS=NUM_FACTORS):
        super(self.__class__, self).__init__()
        self.MeanWeight = Parameter(torch.randn((NUM_TRs,NUM_FACTORS)))
        self.SigmaWeight = Parameter(torch.randn((NUM_TRs,NUM_FACTORS)))
        self.MeanFactorCenter = Parameter(torch.randn((3,NUM_FACTORS)))
        self.SigmaFactorCenter = Parameter(torch.randn((3,NUM_FACTORS)))
        self.MeanFactorWidth = Parameter(torch.randn((NUM_FACTORS)))
        self.SigmaFactorWidth = Parameter(torch.randn((NUM_FACTORS)))

    def forward(self):

        q = probtorch.Trace()

        Weights = q.normal(self.MeanWeight, self.SigmaWeight, name = 'Weights') #W
        FactorCenters = q.normal(self.MeanFactorCenter,self.MeanFactorCenter, name = 'FactorCenters') #M
        FactorWidths = q.normal(self.MeanFactorWidth,self.SigmaFactorWidth,name = 'FactorWidths') #L


        return q

class Decoder(nn.Module):
    def __init__(self,NUM_TRs,NUM_FACTORS,NUM_VOXELS):
        self.MeanWeight = torch.zeros((NUM_TRs, NUM_FACTORS))
        self.SigmaWeight = sourceWeightVariance*torch.ones((NUM_TRs,NUM_FACTORS))
        self.MeanFactorCenter = (brainCenter.expand(3,NUM_FACTORS))*torch.ones((3,NUM_FACTORS))  #c is center of 3D brain image
        self.SigmaFactorCenter = (sourceCenterVariance.expand(3,NUM_FACTORS))*torch.ones((3,NUM_FACTORS)) #s
        self.MeanFactorWidth = torch.ones((NUM_FACTORS))
        self.SigmaFactorWidth = sourceWidthVariance*torch.ones((NUM_FACTORS))
        self.Snoise = voxelNoise*torch.ones(NUM_VOXELS)

    def forward(self,data = voxelActivations,R = voxelLocations,q=None):
        p = probtorch.Trace()
        Weights = p.normal(self.MeanWeight,self.SigmaWeightw,value = q['Weights'],name = 'Weights')
        FactorCenters = p.normal(self.MeanFactorCenter,self.SigmaFactorCenter,value = q['FactorCenters'],
                                 name = 'FactorCenters')
        FactorWidths = p.normal(self.MeanFactorWidth,self.SigmaFactorWidth,value = q['FactorWidths'],
                                name = 'FactorWidths')
        Factors = Function_RBF(R,FactorCenters,FactorWidths)
        Yhat = p.normal(torch.matmul(Weights,Factors),self.Snoise,name = 'Yhat')
        p.loss(((Yhat - data).T)*(Yhat-data),name='y')

        return p

def elbo(q,p):
    return probtorch.objectives.montecarlo.kl(q,p)   #negative taken later

enc = Encoder()
dec = Decoder()

if CUDA:
    enc.cuda()
    dec.cuda()

optimizer = torch.optim.Adam(list(enc.parameters()),lr= LEARNING_RATE)

def train(data,enc,dec,optimizer):
    epoch_elbo = 0
    enc.train()
    dec.train()
    N = 0
    for counter,(voxacts,voxlocs) in enumerate(data):
        N+=NUM_BATCH
        if CUDA:
            voxacts = voxacts.cuda()
            voxlocs = voxlocs.cuda()
        voxacts = Variable(voxacts)
        voxlocs = Variable(voxlocs)
        optimizer.zero_grad()
        q = enc()
        p = dec(data = voxacts,R = voxlocs,q = q)
        loss = -elbo(q,p)
        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
        epoch_elbo -= loss.data.numpy()[0]

        return epoch_elbo/N

def test(data,enc,dec):
    enc.eval()
    dec.eval()

    epoch_elbo = 0
    N = 0

    for counter,(voxacts,voxlocs) in data:
        N+=NUM_BATCH
        if CUDA:
            voxacts = voxacts.cuda()
            voxlocs = voxlocs.cuda()

        voxacts = Variable(voxacts)
        voxlocs = Variable(voxlocs)
        q = enc()
        p = dec(data = voxacts,R = voxlocs,q =q)
        batch_elbo = elbo(q,p)

        if CUDA:
            batch_elbo = batch_elbo.cpu()

        epoch_elbo += batch_elbo.data.numpy()[0]

    return epoch_elbo/N




for e in range(NUM_EPOCH):
    train_elbo = train(train_data,enc,dec)
    #test_elbo = test(test_data,enc,dec)

    print('[Epoch %d] Train: ELBO %.4e Test: ELBO %.4e' % (e, train_elbo))

if CUDA:
    q = enc()
    W = q['Weights'].value.data.cpu().numpy()
    M = q['FactorCenters'].value.data.cpu().numpy()
    L = q['FactorWidths'].value.data.cpud().numpy()





