import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

from my_utils import *

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax", finetune_dropout_p=None):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type
        # BUGFIX: BaselinFinetune has no attribute 'loss_fn'
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        
        self.finetune_dropout_p = finetune_dropout_p

    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation

    def set_forward_adaptation(self,x,is_feature = True):
        # almost the same with MetaTemplate, except linear_clf & loss_function part
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())
        
        if self.finetune_dropout_p is not None:
            dropout = nn.Dropout(p=self.finetune_dropout_p)#.cuda()?

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        # BUGFIX: BaselineFinetune has no attribute 'loss_fn'
        loss_function = self.loss_fn
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()

                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                
                if self.finetune_dropout_p is not None:
                    z_batch = dropout(z_batch)
                
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
                
        scores = linear_clf(z_query)
        return scores


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        
    def forwardout2prob(self, forward_outputs):
        '''
        Args:
            forward_outputs: shape=(n_way*n_query, n_way)
        '''
        probs = nn.Softmax(dim=1)(forward_outputs)
        return probs

