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
#         self.DBval = False; #only set True for CUB dataset, see issue #31
        self.DBval = True; #only set True for CUB dataset, see issue #31

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
#         print('preds:', preds)
#         print('y:', y)
#         print('n_correct:', n_correct)
#         print('n_data:', n_data)
        return n_correct, n_data
        
    
    def train_loop(self, epoch, train_loader, optimizer, compute_acc=True):
        print_freq = 10
        avg_loss=0
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
#                 avg_loss = avg_loss+loss.data[0]
#             else:
            cur_loss = loss.item()
            avg_loss = avg_loss+cur_loss
            # compute acc
            if compute_acc:
                correct_this, count_this = self.correct(x, y)
                acc = correct_this/count_this * 100
                acc_all.append(acc)


            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
#                 print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                description_str = 'Epoch %d: avg Loss = %.2f'%(epoch, avg_loss/float(i+1))
                if compute_acc:
                    avg_acc = np.asarray(acc_all)
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
            return -1   #no validation, just save model during iteration

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
    def __init__(self, model_func, num_class, loss_type, min_gram, lambda_gram):
        if min_gram not in ['l1', 'l2', 'inf']:
            raise ValueError('Invalid min_gram: %s'%(min_gram))
        super(BaselineTrainMinGram, self).__init__(
            model_func=model_func, 
            num_class=num_class, loss_type=loss_type)
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

