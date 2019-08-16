import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from packaging import version
from my_utils import *
from tqdm import tqdm

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x):
#         x    = Variable(x.cuda())
        x    = Variable(to_device(x))
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
#         y = Variable(y.cuda())
        y = Variable(to_device(y))
        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        tt = tqdm(train_loader)
        for i, (x,y) in enumerate(tt):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()
            
#             if version.parse(torch.__version__) < version.parse("0.4.0"):
#                 avg_loss = avg_loss+loss.data[0]
#             else:
            cur_loss = loss.item()
            avg_loss = avg_loss+cur_loss

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
#                 print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                tt.set_description('Epoch %d: avg Loss = %.2f'%(epoch, avg_loss/float(i+1)))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

