import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

from packaging import version
from my_utils import *
from tqdm import tqdm

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func() # set feature backbone
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test
        print('model_function:\n',self.feature)
        self.device = None

    @abstractmethod
    def set_forward(self,x,is_feature):
        ''' get the last output (score) from image or embedding
        '''
        pass

    @abstractmethod
    def set_forward_loss(self, x): # utilized by train_loop
        ''' compute task loss
        '''
        pass
    
    @abstractmethod
    def total_loss(self, x):
        ''' compute whole objective function
        '''
        pass

    def forward(self,x):
        ''' get feature embedding Tensor???
        :param x: input image i think
        '''
        out  = self.feature.forward(x)
        return out
    
    def to(self, device):
        self.device = device
        return super(MetaTemplate, self).to(device)
    
    def parse_feature(self,x,is_feature): # utilized by set_forward
        ''' parsing xs or zs to support and query feature embedding
        '''
        
#         x = Variable(x.cuda())
        
#         print('x.shape =', x.shape, ', x.max() =', x.max(),' ,x.min() =', x.min())
#         print('0.min = %s, 0.max = %s' % (x[:,:,0,:,:].min(),x[:,:,0,:,:].max()))
#         print('1.min = %s, 1.max = %s' % (x[:,:,1,:,:].min(),x[:,:,1,:,:].max()))
#         print('2.min = %s, 2.max = %s' % (x[:,:,2,:,:].min(),x[:,:,2,:,:].max()))
        if self.device is None:
#             x = Variable(to_device(x))
            x = x.cuda()
        else:
            x = x.cuda()
        
        # x.size = n_way, (n_supp + n_que), 3, size, size (even for omniglot channel size is 3
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def parse_x(self, x, is_feature):
        ''' parsing xs into support and query x
        :param x: x_support & x_query before parsing
        :return: x_support, x_query, shape=[n_way, n_sample, dim]
        '''
        assert not is_feature, 'x must not be feature'
        
        
#         x = Variable(x.cuda())
        if self.device is None:
#             x = Variable(to_device(x))
            x = x.cuda()
        else:
            x = x.cuda()
        
        x_reshape = x.contiguous().view(self.n_way, self.n_support+self.n_query, -1)
        x_support = x_reshape[:, :self.n_support]
        x_query = x_reshape[:, self.n_support:]
        
        return x_support, x_query
        
    def correct(self, x):       
        ''' compute accuracy of query_set in an episode
        :param x: x_support & x_query before parse_feature
        :return: n_correct, n_query
        '''
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def correct_batch(self, x):
        ''' compute accuracy of query_set in an episode
        :param x: x_support & x_query before parse_feature
        :return: n_correct, n_query
        '''
        print('correct_batch: x.shape =', x.shape)
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        correct_bool = topk_ind[:,0] == y_query
        top1_correct = np.sum(correct_bool)
        print('plot support set...')
        
        print('plot correct query set...')
        
        print('plot misclassified query set...')
        return float(top1_correct), len(y_query)
    
    def sample_plt(self, x):
        '''
        :param x: x_support & x_query before parse_feature
        '''
        pass
    
    def train_loop(self, epoch, train_loader, optimizer ): # every epoch call this function
        print_freq = 10

        avg_loss=0
        tt = tqdm(train_loader)
        for i, (x,_ ) in enumerate(tt):
            # x.size = batch, 3, 28, 28 for omniglot (wtf?
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            # no need label to compute loss becuase x is ordered
            loss = self.total_loss(x) #self.set_forward_loss( x ) + recons_lambda*self.decoder_loss(x)
            loss.backward()
            optimizer.step()
            
            cur_loss = loss.item()
            avg_loss = avg_loss+cur_loss

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
#                 print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                tt.set_description('Epoch %d: avg Loss = %.2f'%(epoch, avg_loss/float(i+1)))
        
        avg_loss = avg_loss/float(i+1)
        return avg_loss
    
    def test_loop(self, test_loader, record = None):
        ''' not for MAML, MAML will override this function
        '''
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        tt = tqdm(test_loader, desc='Validation')
        for i, (x,_) in enumerate(tt): # episode
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/count_this * 100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        
#         y_support = Variable(y_support.cuda())
        if self.device is None:
#             y_support = Variable(to_device(y_support))
            y_support = Variable(y_support.cuda())
        else:
            y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
#         linear_clf = linear_clf.cuda()
        if self.device is None:
#             linear_clf = to_device(linear_clf)
            linear_clf = linear_clf.cuda()
        else:
            linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
#         loss_function = loss_function.cuda()
        if self.device is None:
#             loss_function = to_device(loss_function)
            loss_function = loss_function.cuda()
        else:
            loss_function = loss_function.cuda()
        
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
#                 selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                if self.device is None:
#                     selected_id = to_device(torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]))
                    selected_id = torch.from_numpy(rand_id[i:min(i+batch_size,support_size)]).cuda()
                else:
                    selected_id = torch.from_numpy(rand_id[i:min(i+batch_size,support_size)]).cuda()
                
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
