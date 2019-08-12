# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

from my_utils import *

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, recons_func = None):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support, recons_func=recons_func)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        ''' get the last output (score) from image or embedding
        '''
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def decoder_forward(self,x,is_feature=False):
        ''' get the reconstructed output from image or embedding
        '''
        x = Variable(to_device(x))
        x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        
        if is_feature:
            embedding = x
        else:
            embedding = self.feature.forward(x)
        
        decoded_img = self.recons_func(embedding)
        
        return decoded_img

    def set_forward_loss(self, x): # utilized by train_loop
        ''' compute task loss
        '''
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
#         y_query = Variable(y_query.cuda())
        y_query = Variable(to_device(y_query))

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )
    
    def decoder_loss(self, x):
        ''' the reconstruction loss
        '''
        if self.recons_func:
            decoded_img = self.decoder_forward(x)
            x = Variable(to_device(x))
            x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
#             print('decoded_img shape =',decoded_img.shape)
            loss = nn.MSELoss()(decoded_img,x) # TODO
        else:
            loss = 0
        return loss


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
