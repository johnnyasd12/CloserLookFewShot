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
    def __init__(self, model_func, num_class, n_way, n_support, loss_type = 'softmax', change_way = True):
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
        self.DBval = False; #only set True for CUB dataset, see issue #31
#         self.DBval = True; #only set True for CUB dataset, see issue #31
        
        # only for validation fine-tuning
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input)
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    def forward(self,x):
        x    = Variable(x.cuda())
#         x    = Variable(to_device(x))
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
#         y = Variable(to_device(y))
        return self.loss_fn(scores, y )
    
    def total_loss(self, x, y):
        loss = self.forward_loss(x, y)
        return loss
    
    def pred(self, x):
        scores = self.forward(x) # (batch_size, num_classes)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def correct(self, x, y):
        preds = self.pred(x)
        y = y.cuda()
        n_data = y.size(0)
        n_correct = (preds==y).sum().cpu().numpy()
        return n_correct, n_data
        
    
    def train_loop(self, epoch, train_loader, optimizer, compute_acc=True):
        print_freq = 10
        sum_loss=0
        # for compute_acc
        if compute_acc:
            acc_all = []

        tt = tqdm(train_loader)
        for i, (x,y) in enumerate(tt):
            optimizer.zero_grad()
#             loss = self.forward_loss(x, y) # TODO: 
            loss = self.total_loss(x, y)
            loss.backward()
            optimizer.step()
            
#             if version.parse(torch.__version__) < version.parse("0.4.0"):
#                 sum_loss = sum_loss+loss.data[0]
#             else:
            cur_loss = loss.item()
            sum_loss = sum_loss+cur_loss
            # compute acc
            if compute_acc:
                correct_this, count_this = self.correct(x, y)
                acc = correct_this/count_this * 100
                acc_all.append(acc)


            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
#                 print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                avg_loss = sum_loss/float(i+1)
                description_str = 'Epoch %d: avg Loss = %.2f'%(epoch, avg_loss)
                if compute_acc:
                    avg_acc = np.asarray(acc_all)
                    n_avg = 100
                    if i > n_avg:
                        avg_acc = np.mean(avg_acc[-n_avg:])
                    else:
                        avg_acc = np.mean(avg_acc)
                    description_str += ' , avg Acc = %.2f%%'%(avg_acc)
                tt.set_description(description_str)
        
        avg_loss = avg_loss/float(i+1)
        
        if compute_acc:
            avg_acc = np.mean(acc_all)
            return avg_acc, avg_loss
        else:
            return avg_loss
                     
    def test_loop(self, val_loader):
        if self.DBval:
            return self.analysis_loop(val_loader)
        else:
#             return -1   #no validation, just save model during iteration
            return self.finetune_loop(val_loader)

    def finetune_loop(self, val_loader):
        # almost the same with test_loop in MetaTemplate except self.change_way
        acc_all = []
        n_episodes = len(val_loader) 
        tt = tqdm(val_loader, desc='Validation')
        for i, (x,_) in enumerate(tt): # for each episode
            # x.shape = (N,K,C,H,W), N-way, K-shot
            self.n_query = x.size(1) - self.n_support
#             if self.change_way:
            self.n_way  = x.size(0)
            correct_this, count_this = self.finetune_correct(x)
            acc_all.append(correct_this/count_this * 100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
#         print('%d Test Acc = %4.2f%% +- %4.2f%%' %(n_episodes,  acc_mean, 1.96* acc_std/np.sqrt(n_episodes)))
        print('Val Acc = %4.2f%% +- %4.2f%% with %d episodes.' %(acc_mean, 1.96* acc_std/np.sqrt(n_episodes), n_episodes))

        return acc_mean
    
    def finetune_correct(self, x):
        scores = self.set_forward_adaptation(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward_adaptation(self, x, is_feature = False):
        # just for finetuning
        # almost the same with MetaTemplate, except linear_clf & loss_function part
        # exactly the same with BaselineFinetune, except the assertion. 
        
        # freeze feature network
        # reset to trainable
        for par in self.feature.parameters():
            par.requires_grad = False
        
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

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
        # an epoch is just go through support set once
        n_epoch = 20 #100
        for epoch in range(n_epoch):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()

                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
                
        scores = linear_clf(z_query)
        
        # reset to trainable
        for par in self.feature.parameters():
            par.requires_grad = True
        
        return scores
        
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
    
    def analysis_loop(self, val_loader, record = None):
        class_file  = {}
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var).data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(feats, labels):
                if l not in class_file.keys():
                    class_file[l] = []
                class_file[l].append(f)

        for cl in class_file:
            class_file[cl] = np.array(class_file[cl])
        
        DB = DBindex(class_file)
        print('DB index = %4.2f' %(DB))
        return 1/DB #DB index: the lower the better

class BaselineTrainMinGram(BaselineTrain):
    def __init__(self, model_func, num_class, n_way, n_support, loss_type, min_gram, lambda_gram, change_way = True):
        if min_gram not in ['l1', 'l2', 'inf']:
            raise ValueError('Invalid min_gram: %s'%(min_gram))
        super(BaselineTrainMinGram, self).__init__(
            model_func=model_func, 
            num_class=num_class, loss_type=loss_type, 
            n_way=n_way, n_support=n_support, # just for validation fine-tune
            change_way=change_way
        )
        self.min_gram = min_gram
        self.lambda_gram = lambda_gram
    
    def total_loss(self, x, y):
        standard_loss = self.forward_loss(x, y)
        min_gram_loss = self.min_gram_loss(x)
        loss = standard_loss + self.lambda_gram*min_gram_loss
        return loss
    
    def min_gram_loss(self, x):
        # also in ProtoNetMinGram
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

def DBindex(cl_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

