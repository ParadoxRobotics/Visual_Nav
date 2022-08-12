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

# Place descriptor CNN with denseFly LSH
class PRCNN(nn.Module):
    # Layer Init
    def __init__(self, InputShape, DescriptorSize, ConnectionProbability):
        super(PRCNN, self).__init__()
        # Parameters
        self.inputSize = InputShape
        self.desSize = DescriptorSize
        self.cp = ConnectionProbability
        # Init feature extractor
        self.features = models.alexnet(pretrained=True).features
        # create Bio-LSH layer (denseFly based)
        outputSize = self.features(torch.randn(1, self.inputSize[0], self.inputSize[1], self.inputSize[2])).size()
        self.LSH = nn.Linear(outputSize[1]*outputSize[2]*outputSize[3], self.desSize, False)
        # create binary weight matrix
        binWeight = torch.randn(self.desSize, outputSize[1]*outputSize[2]*outputSize[3])
        binWeightMask= ((binWeight>1-self.cp).long().type(torch.float))
        binWeight = torch.randn(self.desSize, outputSize[1]*outputSize[2]*outputSize[3])*binWeightMask
        self.LSH.weight.data = binWeight
    # Generate descriptor
    def forward(self, Input):
        # Extract features
        feat = self.features(Input)
        # Compute LSH descriptor
        des = torch.sign(F.relu(feat.contiguous().view(1, -1)))
        return des

# Init network
net = PRCNN(InputShape=[3, 64, 128], DescriptorSize=1024, ConnectionProbability=0.18).to(device)
# TEST
sta1 = cv2.resize(cv2.imread('/home/main/Bureau/st1.jpg'), (128, 64))
sta2 = cv2.resize(cv2.imread('/home/main/Bureau/st2.jpg'), (128, 64))
adv1 = cv2.resize(cv2.imread('/home/main/Bureau/robot.jpeg'), (128, 64))
# convert to tensor
sta1 = img2Alex(sta1)
sta2 = img2Alex(sta2)
adv1 = img2Alex(adv1)
# create descriptor
d1 = net(sta1.to(device))
d2 = net(sta2.to(device))
d3 = net(adv1.to(device))
# compute cosine similarity
cosineDist = nn.CosineSimilarity(dim=1, eps=1e-6)

s1 = cosineDist(d1, d2)
s2 = cosineDist(d1, d3)
s3 = cosineDist(d2, d3)
print(s1, s2, s3)
