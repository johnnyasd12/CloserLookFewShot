# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate, AENet

from my_utils import *



class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        ''' get the last output (scores of query set) from image or embedding
        Returns:
            shape: (n_way*n_query, n_way)
        '''
        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x): # utilized by train_loop
        ''' compute task loss (by query set) given support and query set
        '''
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
#         if self.device is None:
#             y_query = Variable(to_device(y_query))
#         else:
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query )
    
    def forwardout2prob(self, forward_outputs):
        '''
        Args:
            forward_outputs: shape=(n_way*n_query, n_way)
        '''
        probs = nn.Softmax(dim=1)(forward_outputs)
        return probs
    
    def total_loss(self, x):
        return self.set_forward_loss(x)

class ProtoNetMinGram(ProtoNet):
    def __init__(self, model_func,  n_way, n_support, min_gram, lambda_gram):
        '''
        Args:
            min_gram: what to minimize? 'l2', 'l1' norm of Gram Matrix
            lambda_gram: coefficient of Gram Matrix loss
        '''
        if min_gram not in ['l1', 'l2', 'inf']:
            raise ValueError('Invalid min_gram: %s'%(min_gram))
        
        super(ProtoNetMinGram, self).__init__( model_func,  n_way, n_support)
        self.min_gram = min_gram
        self.lambda_gram = lambda_gram
        
    
    def total_loss(self, x):
        standard_loss = self.set_forward_loss(x)
        min_gram_loss = self.min_gram_loss(x)
        loss = standard_loss + self.lambda_gram*min_gram_loss
        return loss
        
    def min_gram_loss(self, x):
        # also in BaselineTrainMinGram
        x = x.cuda()
        N,C = x.size(0), x.size(1)
#         print('self.min_gram:', self.min_gram)
        if self.min_gram == 'l2':
            p = 2
        elif self.min_gram == 'l1':
            p = 1
        elif self.min_gram == 'inf':
            p = float('inf')
        gram_matrix = self.feature.get_hidden_gram(x) # shape = (N,C,C)
        gram_reshape = gram_matrix.view(N,-1) # N,C*C
        gram_norm = torch.norm(gram_reshape, p=p, dim=1) # N, 
        if p == float('inf'):
            gram_norm_square = gram_norm
        else:
            gram_norm_square = gram_norm**p
        loss = 1/N * torch.sum(gram_norm_square)
        return loss
        
class ProtoNetAE(AENet, ProtoNet): # TODO: self.recons_func = recons_func()
    def __init__(self, model_func,  n_way, n_support, recons_func = None, lambda_d = 1):
        super(ProtoNetAE, self).__init__( model_func,  n_way, n_support)
        self.recons_func = recons_func
        self.lambda_d = lambda_d
        self.encoder = self.feature
        from backbone import LambdaLayer # trick to get identity function as extractor after encoder
        self.extractor = LambdaLayer(lambda x: x)
        
    def set_forward_with_decoded_img(self,x,is_feature = False):
        ''' get the last output (scores of query set) from image or embedding
        Returns:
            shape: (n_way*n_query, n_way)
        '''
        z_support, z_query, encodings  = self.parse_feature_with_encoding(x,is_feature)
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        decoded_imgs = self.recons_func(encodings)
        
        return scores, decoded_imgs


# class ProtoNetAE(ProtoNet): # TODO: self.recons_func = recons_func()
#     def __init__(self, model_func,  n_way, n_support, recons_func = None, lambda_d = 1):
#         super(ProtoNetAE, self).__init__( model_func,  n_way, n_support)
#         self.recons_func = recons_func
#         self.lambda_d = lambda_d
    
#     def total_loss(self, x):
#         return self.set_forward_loss(x) + self.reconstruct_loss(x)*self.lambda_d
    
#     def decoder_forward(self,x,is_feature=False):
#         ''' get the reconstructed output from image or embedding
#         '''
# #         x = Variable(to_device(x))
#         x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        
#         if is_feature:
#             embedding = x
#         else:
#             embedding = self.feature.forward(x)
        
#         decoded_img = self.recons_func(embedding)
        
#         return decoded_img

#     def reconstruct_loss(self, x):
#         ''' the reconstruction loss
#         '''
#         if self.recons_func:
#             if self.device is None:
# #                 x = Variable(to_device(x)) # TODO: done: switch this two rows?
#                 x = Variable(x.cuda())
#             else:
#                 x = Variable(x.cuda())
            
#             decoded_img = self.decoder_forward(x) # TODO: done: switch this two rows?
# #             print('ProtoNetAE: \nx.shape:', x.shape, '\nx.min:', x.min(), '\nx.max:', x.max())
# #             print('decoded_img.min:', decoded_img.min(), '\ndecoded_img.max:', decoded_img.max())
#             x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
# #             print('decoded_img shape =',decoded_img.shape)
#             loss = nn.MSELoss()(decoded_img,x) # TODO
#         else:
#             loss = 0
#         return loss

class ProtoNetAE2(ProtoNetAE):
    def __init__(self, model_func,  n_way, n_support, recons_func = None, lambda_d = 1, extract_layer = 2, is_color = True):
        super(ProtoNetAE2, self).__init__( model_func,  n_way, n_support, 
                                          recons_func = recons_func, lambda_d = lambda_d)
        self.encoder = self.feature.trunk[:extract_layer] # TODO: changed when different architecture
        self.extractor = self.feature.trunk[extract_layer:]
        self.is_color = is_color

# class ProtoNetAE2(ProtoNetAE):
#     def __init__(self, model_func,  n_way, n_support, recons_func = None, lambda_d = 1, extract_layer = 2, is_color = True):
#         super(ProtoNetAE2, self).__init__( model_func,  n_way, n_support, 
#                                           recons_func = recons_func, lambda_d = lambda_d)
#         self.encoder = self.feature.trunk[:extract_layer] # TODO: changed when different architecture
#         self.extractor = self.feature.trunk[extract_layer:]
#         self.is_color = is_color
        
#     def decoder_forward(self, x, is_feature = False):
#         x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        
#         assert is_feature == False, "decoder_forward: is_feature must be False. "
#         if is_feature: # TODO: if is_feature
#             pass # aaaahhhhhh
#         else:
#             if not self.is_color:
#                 x = x[:,0:1,:,:]
#             embedding = self.encoder.forward(x)
        
#         decoded_img = self.recons_func(embedding)
        
#         return decoded_img

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    assert x.size(1) == y.size(1)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
