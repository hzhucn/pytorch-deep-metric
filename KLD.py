# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
import torch.nn as nn
from torch.backends import cudnn
import models
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display, orth_reg
import DataSet
cudnn.benchmark = False

import torch
import torch.nn.functional as F
from torch.autograd import Variable

rand_data = torch.randn(3, 8)

a = Variable(torch.randn(3, 8))
b = Variable(torch.randn(3, 8))

criterion = nn.KLDivLoss(size_average=False)

def KLDiv(log_inputs, target):
    return torch.sum(target*(torch.log(target) - log_inputs))

def js_div(a, b):
    num = float(a.size()[0])
    criterion = nn.KLDivLoss(size_average=False)
    softmax_a = F.softmax(a)
    softmax_b = F.softmax(b)
    softmax_mean = (softmax_a + softmax_b) / 2

    lsm_a = F.log_softmax(a)
    lsm_b = F.log_softmax(b)

    div = (0.5/num)*(criterion(lsm_a, softmax_mean) + criterion(lsm_b, softmax_mean))
    return div


lsm_a = F.log_softmax(a)
# lsm_b = F.log_softmax(b)
sm_b = F.softmax(b)

print(criterion(lsm_a, sm_b))
print(KLDiv(lsm_a, sm_b))
