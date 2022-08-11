# Place recognition CNN+FlyLSH systems
# Author : Munch Quentin, 2022

import math
import numpy as np
import scipy
import scipy.linalg as linalg
import cv2
import imutils
import copy
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenCV image to normalized tensor
def img2Alex(img):
    normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input = torch.from_numpy(img)
    input = input.view((1, *input.size())).type(torch.FloatTensor)
    input = input.permute(0, 3, 1, 2)
    return normalization(input/255.0)

# K-WTA sparse activation [0||1] (NEED TO BE OPTIMIZED)
def KWTA(input, KSparsity):
    k = int(KSparsity*input.shape[1])
    topval = input.topk(k, dim=1)[0][:, -1]
    topval = topval.expand(input.shape[1], input.shape[0]).permute(1,0)
    comp = (input>=topval).to(input)
    ht = comp*input
    ht[ht==-0.0] = 0.0
    ht[ht!=0] = 1.0
    return ht

# Place descriptor CNN with denseFly LSH
class PRCNN(nn.Module):
    # Layer Init
    def __init__(self, InputShape, DescriptorSize, ConnectionProbability, Sparsity):
        super(PRCNN, self).__init__()
        # Parameters
        self.inputSize = InputShape
        self.desSize = DescriptorSize
        self.cp = ConnectionProbability
        self.ks = Sparsity
        # Init feature extractor
        self.features = models.alexnet(pretrained=True).features
        print(self.features)
        # create Bio-LSH layer (denseFly based)
        outputSize = self.features(torch.randn(1, self.inputSize[0], self.inputSize[1], self.inputSize[2])).size()
        self.LSH = nn.Linear(outputSize[1]*outputSize[2]*outputSize[3], self.desSize, False)
        # create binary weight matrix
        binWeight = torch.randn(self.desSize, outputSize[1]*outputSize[2]*outputSize[3])
        binWeight = (binWeight>1-self.cp).long()
        print(binWeight)
        self.LSH.weight.data = binWeight.type(torch.float)
    # Generate descriptor
    def forward(self, Input):
        # Extract features
        feat = self.features(Input)
        # Compute LSH descriptor
        des = KWTA(input=feat.view(1, -1), KSparsity=self.ks)
        return des

# Init network
net = PRCNN(InputShape=[3, 320, 640], DescriptorSize=8000, ConnectionProbability=0.18, Sparsity=0.05)
# TEST
sta1 = cv2.resize(cv2.imread('/home/main/Bureau/st1.jpg'), (640, 320))
sta2 = cv2.resize(cv2.imread('/home/main/Bureau/st2.jpg'), (640, 320))
adv1 = cv2.resize(cv2.imread('/home/main/Bureau/st3.jpg'), (640, 320))
# convert to tensor
sta1 = img2Alex(sta1)
sta2 = img2Alex(sta2)
adv1 = img2Alex(adv1)
# create descriptor
d1 = net(sta1)
d2 = net(sta2)
d3 = net(adv1)
# compute cosine similarity
cosineDist = nn.CosineSimilarity(dim=1, eps=1e-6)

s1 = cosineDist(d1, d2)
s2 = cosineDist(d1, d3)
s3 = cosineDist(d2, d3)
print(cosineDist(d3, d3))
print(s1, s2, s3)
