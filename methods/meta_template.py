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

class AENet:
    def parse_feature_with_encoding(self,x,is_feature): # utilized by set_forward
        ''' parsing xs or zs to support and query feature embedding
        Return:
            z_support: shape=(n_way, n_support,...)
            z_query: shape=(n_way, batch_size - n_support, ...)
        '''
        x = x.cuda()
        
        # x.size = n_way, (n_supp + n_que), 3, size, size (even for omniglot channel size is 3
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            # TODO: encoder forward
            encodings   = self.encoder.forward(x)
            z_all       = self.extractor.forward(encodings)
#             z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query, encodings
    
    
    def set_forward_loss(self, x): # utilized by train_loop
        ''' compute task loss (by query set) given support and query set
        '''
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
#         if self.device is None:
#             y_query = Variable(to_device(y_query))
#         else:
        y_query = Variable(y_query.cuda())
        scores, decoded_imgs = self.set_forward_with_decoded_img(x)
        
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)).cuda()
#         print('decoded_imgs shape =',decoded_imgs.shape)
        recons_loss = nn.MSELoss()(decoded_imgs,x) # TODO
    
        return self.loss_fn(scores, y_query ), recons_loss
    
    def total_loss(self, x):
        set_loss, recons_loss = self.set_forward_loss(x)
        return set_loss + recons_loss*self.lambda_d

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input)
#         self.feature    = model_func(dropout_p=dropout_p) # set feature backbone
        self.feature    = model_func() # set feature backbone
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test
        print('model_function:\n',self.feature)
        self.device = None

    @abstractmethod
    def set_forward(self,x,is_feature):
        ''' split x into support & query, then get the last output (score) from image or embedding
        '''
        pass

    @abstractmethod
    def set_forward_loss(self, x): # utilized by train_loop
        ''' compute task loss (by query set) given support & query set
        '''
        pass
    
    @abstractmethod
    def total_loss(self, x):
        ''' compute whole objective function
        '''
        pass

    def forwardout2loss(self, scores):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())
        loss = self.loss_fn(scores, y_query )
        return loss
    
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
        Return:
            z_support: shape=(n_way, n_support,...)
            z_query: shape=(n_way, batch_size - n_support, ...)
        '''
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
#         if self.device is None:
# #             x = Variable(to_device(x))
#             x = x.cuda()
#         else:
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

    def train_loop(self, epoch, train_loader, optimizer, compute_acc=True): # every epoch call this function
        print_freq = 10

        sum_loss=0
        # for compute_acc
        if compute_acc:
            acc_all = []
        
        tt = tqdm(train_loader)
        for i, (x,_ ) in enumerate(tt):
            # x.size = batch, 3, 28, 28 for omniglot
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            
            # compute loss
            loss = self.total_loss(x) # no need label to compute loss because x is ordered
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            sum_loss = sum_loss+cur_loss

            # compute acc
            if compute_acc:
                correct_this, count_this = self.correct(x)
                acc = correct_this/count_this * 100
                acc_all.append(acc)
            
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                description_str = 'Epoch %d: avg Loss = %.2f'%(epoch, sum_loss/float(i+1))
                if compute_acc:
#                     avg_acc = np.asarray(acc_all)
#                     avg_acc = np.mean(avg_acc)
#                     description_str += ' , avg Acc = %.2f%%'%(avg_acc)
                    avg_acc = np.asarray(acc_all)
                    n_avg = 100
                    if i > n_avg:
                        avg_acc = np.mean(avg_acc[-n_avg:])
                    else:
                        avg_acc = np.mean(avg_acc)
                    description_str += ' , avg Acc = %.2f%%'%(avg_acc)
                tt.set_description(description_str)
        
        avg_loss = sum_loss/float(i+1)
        
        if compute_acc:
            avg_acc = np.mean(acc_all)
            return avg_acc, avg_loss
        else:
            return avg_loss
    
    def test_loop(self, val_loader, record = None):
        ''' not for MAML, MAML will override this function
        '''
        acc_all = []
        
        iter_num = len(val_loader) 
        tt = tqdm(val_loader, desc='Validation')
        for i, (x,_) in enumerate(tt): # for each episode
            # x.shape = (N,K,C,H,W), N-way, K-shot
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/count_this * 100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
#         print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        print('Val Acc = %4.2f%% +- %4.2f%% with %d episodes.' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num), iter_num))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i:min(i+batch_size,support_size)]).cuda()
                
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
