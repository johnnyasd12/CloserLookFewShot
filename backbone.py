# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from my_utils import *
import logging

# to reconstruct image back
torch.autograd.set_detect_anomaly(True)
def img_standardize(img, normalize_param = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])):
    img = img + 1 # DO NOT DO in-place modification
    img = img / 2 # DO NOT DO in-place modification
    
    means = normalize_param['mean']
    stds = normalize_param['std']
    
    for channel in range(3):
        # here do the in-place operation but its okay because here is already not original images
        img[:,channel,:,:] = img[:,channel,:,:] - means[channel]#.sub(normalize_param['mean'][channel])
        img[:,channel,:,:] = img[:,channel,:,:] / stds[channel]#.div(normalize_param['std'][channel])
    return img

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features, gpu_id): # TODO: initialize gpu_id in MAML
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None
        if gpu_id:
            self.device = torch.device('cuda:'+str(gpu_id))
        else:
            self.device = None

    def forward(self, x):
#         running_mean = torch.zeros(x.data.size()[1]).cuda()
#         running_var = torch.ones(x.data.size()[1]).cuda()
        if self.device is None:
            running_mean = to_device(torch.zeros(x.data.size()[1]))
            running_var = to_device(torch.ones(x.data.size()[1]))
        else:
            running_mean = torch.zeros(x.data.size()[1]).to(self.device)
            running_var = torch.ones(x.data.size()[1]).to(self.device)
        
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out


class MyDropout(nn.Module):
    def __init__(self, n_features, p, inplace: bool = False):
        '''
        Args:
            n_features (int): number of channels or features
            p (float): dropout probability (1-p = keep_prob)
            inplace (bool): haven't implement yet
        '''
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.n_features = n_features
        self.p = p # 1-p = keep_prob
    
    def get_random_mask(self, n_samples, fix_num_drop=False):
        # get mask Tensor without grad, return shape: (n_samples, n_features)
        # p is dropout prob (not keep_prob)
        n_features = self.n_features
        if fix_num_drop:
            # TODO: fix_n_drop
            mask = None
        else:
            mask = torch.Tensor(n_samples,n_features).uniform_(0,1)>self.p
        mask = Variable(mask.type(torch.cuda.FloatTensor), requires_grad=False)
        return mask
    
    def get_reshaped_random_mask(self, x): # different between dropout and dropout2d
        # So that we have `(n_samples, n_features)` numbers of Bernoulli(1-p) samples
        n_samples = x.shape[0]
        mask = self.get_random_mask(n_samples)
        # no need to reshape for normal dropout, only dropout2d need extra reshape
        return mask
    
    def forward(self, x):
        n_features = self.n_features # also equals x.shape[1] as well as dropout2d case
        if not self.training: # eval() mode
            return x
        else: # if train() mode
            mask = self.get_reshaped_random_mask(x) # shape: (n_samples, n_features), dropout2d shape: (N,C,H,W)
            return torch.mul(mask,x) * 1/(1-self.p) # inverse dropout



class CustomDropout(MyDropout):
    def __init__(self, n_features, p, inplace: bool = False):
        '''
        Args:
            n_features (int): number of channels or features
            p (float): dropout probability (1-p = keep_prob)
            inplace (bool): haven't implement yet
        '''
        super(CustomDropout, self).__init__(n_features=n_features, p=p, inplace=inplace)
        self.eval_mask = None # only used when sampling subnet, shape: (1, n_features)
    
    def get_reshaped_eval_mask(self, x): # different between dropout and dropout2d
        n_samples = x.shape[0]
        mask = self.eval_mask # shape: (1, n_features)
        mask = mask.repeat(n_samples,1) # shape: (n_samples, n_features)
        return mask
        
    def forward(self, x):
#         n_samples = x.shape[0]
        n_features = self.n_features # also equals x.shape[1] as well as dropout2d case
        if not self.training: # eval() mode
            if self.eval_mask is not None:
                mask = self.get_reshaped_eval_mask(x) # shape: (n_samples, n_features), dropout2d shape: (N,C,H,W)
                return torch.mul(mask,x) * 1/(1-self.p) # inverse dropout for eval() mode
            else: # if self.eval_mask is None
                return x
        else: # if train() mode
            mask = self.get_reshaped_random_mask(x) # shape: (n_samples, n_features), dropout2d shape: (N,C,H,W)
            # Multiply output by multiplier as described in the paper [1]
            return torch.mul(mask,x) * 1/(1-self.p) # inverse dropout
        
    def set_random_eval_mask(self): # this is the same for ALL examples
        random_mask = self.get_random_mask(n_samples=1)
        self.eval_mask = random_mask
    
    def get_mask_comb(self):
        # dropout_p generally equals to self.p
        mask_comb = []
        n_comb = int(1//self.p) # e.g. 1//0.33 = 3
        n_drop_features = int(self.n_features*self.p) # e.g. 20*0.33 = 6
        remain_feature_ids = list(range(self.n_features))
        for i in range(n_comb):
            sampled_feature_ids = np.random.choice(remain_feature_ids, size=n_drop_features, replace=False)
#             print('sampled_feature_ids:', sampled_feature_ids)
            mask_np = np.ones((1, self.n_features))
            mask_np[0][sampled_feature_ids] = 0
#             print('mask_np:', mask_np)
            mask = torch.Tensor(mask_np)
            mask = Variable(mask.type(torch.cuda.FloatTensor), requires_grad=False)
#             print('mask:', mask)
            mask_comb.append(mask)
            for idx in sampled_feature_ids:
                remain_feature_ids.remove(idx)
        return mask_comb
    
    def reset_eval():
        self.eval_mask = None
        self.eval()


class MyDropout2D(MyDropout):
    def __init__(self, n_features, p, inplace: bool = False):
        '''
        Args:
            n_features (int): number of channels or features
            p (float): dropout probability (1-p = keep_prob)
            inplace (bool): haven't implement yet
        '''
        super(MyDropout2D, self).__init__(n_features=n_features, p=p, inplace=inplace)
        
    def get_reshaped_random_mask(self, x): # different between dropout and dropout2d
        # So that we have `(n_samples, n_features)` numbers of Bernoulli(1-p) samples
        n_samples = x.shape[0]
        c = x.shape[1] # also is n_features
        h = x.shape[2]
        w = x.shape[3]
        mask = self.get_random_mask(n_samples) # (N, C)
        mask = mask.view(n_samples,c,1,1) # (N, C, 1, 1)
        mask = mask.repeat(1,1,h,w) # (N, C, H, W)
        return mask
        
class CustomDropout2D(MyDropout2D, CustomDropout):
    def __init__(self, n_features, p, inplace: bool = False):
        '''
        Args:
            n_features (int): number of channels or features
            p (float): dropout probability (1-p = keep_prob)
            inplace (bool): haven't implement yet
        '''
        
        # these 2 lines might have some problems that call __init__() of MyDropout twice, to avoid the problem, maybe we could just call CustomDropout.__init__()???
        super(CustomDropout2D, self).__init__(n_features=n_features, p=p, inplace=inplace)
        CustomDropout.__init__(self, n_features=n_features, p=p, inplace=inplace)
        
    def get_reshaped_eval_mask(self, x): # different between dropout and dropout2d
        n_samples = x.shape[0]
        c = x.shape[1] # also is n_features
        h = x.shape[2]
        w = x.shape[3]
        mask = self.eval_mask # shape: (1, C)
        mask = mask.repeat(n_samples,1) # shape: (N, C)
        mask = mask.view(n_samples,c,1,1) # (N, C, 1, 1)
        mask = mask.repeat(1,1,h,w) # (N, C, H, W)
        return mask


class CustomDropoutNet:
    def record_active_dropout(self):
        self.active_dropout_ls = []
        for module in self.modules():
            if isinstance(module, CustomDropout):
                if module.p != 0: # becuz not all of CustomDropout module are active
                    self.active_dropout_ls.append(module)
    
    def sample_random_subnet(self):
        # traverse all over the nn.Modules to get CustomDropout
        has_custom_dropout = False if len(self.active_dropout_ls)==0 else True
        assert has_custom_dropout, "there should be CustomDropout module to sample random subnet"
        assert not self.training, "should be in eval() mode when calling function"
        for module in self.active_dropout_ls:
            module.set_random_eval_mask()
        
    def reset_dropout(self):
        for module in self.active_dropout_ls:
            module.eval_mask = None


class CustomDropoutBlock:
    def after_standard_init(self, n_features, dropout_p):
        if dropout_p == 0:
            self.dropout = None
        else:
            self.dropout = CustomDropout2D(n_features=n_features, p=dropout_p)
    
    def after_standard_forward(self, inputs):
        if self.dropout is None:
            outputs = inputs
        else:
            outputs = self.dropout(inputs)
        return outputs

def feat2gram(feat, normalize=True):
    '''
    feat shape: (N,C,H,W)
    gram shape: (N,C,C)
    '''
    # input shape: (N,C,H,W)
    N,C,H,W = feat.size()
    feat = feat.view(N,C,H*W) # N,C,H*W
    feat_t = feat.transpose(1,2) # N,H*W,C
    feat_gram = torch.bmm(feat, feat_t) # batch-wise matmul -> N,C,C
    if normalize:
        feat_gram = feat_gram / (2*C*H*W) # would be squared in loss
    return feat_gram


class MinGramDropoutNet:
    '''
    should implement:
        self.trunk_to_gram_block (nn.Sequential): the trunk that outputs feature map
    '''
    def min_gram_init(self, gram_bid):
        self.gram_bid = gram_bid
    
    def get_feature_map_for_gram(self, x, dropout=True):
        # TODO: dropout argument can be removed becuz already handled in self.trunk_to_gram_block
        if len(x.size())==5:
            # meta learning dims
            N, K, C, H, W = x.size() # N-way, K-shot
            x = x.view(N*K, C, H, W)
        elif len(x.size())==4:
            # baseline dims
            N, C, H, W = x.size()
        
        if self.indim == 1:
            x = x[:,0:1,:,:]
#         if dropout:
        return self.trunk_to_gram_block.forward(x)
#         else:
#             # TODO: remove dropout from block
#             raise ValueError("Haven't implement get_feature_map_for_gram() for dropout=False.")
    
    def get_hidden_gram(self, x):
#         print('self.gram_bid:', self.gram_bid)
        if self.gram_bid is None:
            raise ValueError('should not get_hidden_gram since self.gram_bid is None.')
        elif self.gram_bid == 'after_dropout':
            feat = self.get_feature_map_for_gram(x, dropout=True)
        elif self.gram_bid == 'before_dropout':
            feat = self.get_feature_map_for_gram(x, dropout=False)
        gram = feat2gram(feat)
        return gram


# class GramBlock:
#     '''
#     Attributes:
#         self.should_out_gram (bool)
        
#     '''
#     def after_standard_init(self, should_out_gram):
#         self.should_out_gram = should_out_gram
    
#     def additional_forward(self, inputs):
#         if self.should_out_gram:
#             pass
#         else:
#             outputs = inputs
#         return outputs


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1, dropout_p=0.):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)
        
        self.parametrized_layers = [self.C, self.BN, self.relu]
        
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

#         if dropout_p != 0:
#             self.dropout = CustomDropout2D(n_features=outdim, p=dropout_p)
#             self.parametrized_layers.append(self.dropout)
        
        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)
        
        # for CustomDropout
        CustomDropoutBlock.after_standard_init(self, n_features=outdim, dropout_p=dropout_p)


    def forward(self,x):
        out = self.trunk(x)
        out = CustomDropoutBlock.after_standard_forward(self, out)
        return out

# Simple Block for Decoder of ResAE18
class DeSimpleBlock(nn.Module):
    maml = False
    def __init__(self, indim, outdim, double_res):
        super(DeSimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            raise ValueError('DeSimpleBlock do not support maml.')
        else:
            self.CT1 = nn.ConvTranspose2d(indim, indim, kernel_size=3, stride=1, 
                                          padding=1, output_padding=0, bias=False)
            self.BN1 = nn.BatchNorm2d(indim)
            self.relu1 = nn.ReLU(inplace=True)
            self.CT2 = nn.ConvTranspose2d(indim, outdim, kernel_size=3, 
                                          stride=2 if double_res else 1, padding=1, 
                                          output_padding=1 if double_res else 0, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
            self.relu2 = nn.ReLU(inplace=True)
            
            self.parametrized_layers = [self.CT1, self.CT2, self.BN1, self.BN2]
            
            for layer in self.parametrized_layers:
                init_layer(layer)
    
    def forward(self, x):
        out = self.CT1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.CT2(out)
        out = self.BN2(out)
        out = self.relu2(out)
        return out

class LambdaLayer(nn.Module):
    # to do some hack in ResNet
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res, dropout_p=0): # half_res means output size would be half
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml: # no need check this so far
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False) # ResNet18: 
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        # to do the trunk for min_gram, should be in true order
        self.layers_wo_shortcut = [ # no shortcut, no activation
            self.C1, self.BN1, self.relu1, 
            self.C2, self.BN2
        ]
        self.trunk_wo_shortcut = nn.Sequential(*self.layers_wo_shortcut)

        self.half_res = half_res # half_res means output size would be half

        # setting shortcut. need a 1x1 convolution if in_dim!=out_dim
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

#         self.dropout = None
#         if dropout_p != 0:
#             self.dropout = CustomDropout2D(n_features=outdim, p=dropout_p)
# #             self.parametrized_layers.append(self.dropout)
        
        for layer in self.parametrized_layers:
            init_layer(layer)
        
        # for CustomDropout
        CustomDropoutBlock.after_standard_init(self, n_features=outdim, dropout_p=dropout_p)
        # for minimizing Gram
        self.trunk = LambdaLayer(self.trunk_forward) # hack to simulate original block (without dropout)

    def trunk_forward(self, x): 
        # a hack to simulate trunk behavior in ConvNetS
        out = self.trunk_wo_shortcut(x)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        
        out = out + short_out
        out = self.relu2(out)
        return out
        
    def forward(self, x):
#         out = self.C1(x)
#         out = self.BN1(out)
#         out = self.relu1(out)
#         out = self.C2(out)
#         out = self.BN2(out)

#         out = self.trunk_wo_shortcut(x)
#         short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        
#         out = out + short_out
#         out = self.relu2(out)
        out = self.trunk(x)
        
        out = CustomDropoutBlock.after_standard_forward(self, out)
#         if self.dropout != None:
#             out = self.dropout(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module): # utilized by ResNet50, ResNet101
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module, CustomDropoutNet):
    def __init__(self, depth, flatten = True, dropout_p=0., dropout_block_id=3, more_to_drop=None): # CUB/miniImgnet Conv input = 84*84*3
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth): 
            ''' input = 3*84*84
            -> [64*84*84 -> 64*42*42]
            -> [64*42*42 -> 64*21*21]
            -> [64*21*21 -> 64*10*10]
            -> [64*10*10 -> 64*5*5]
            '''
            # if the 1st block then input is image, otherwise 64 from pre-block
#             indim = 3 if i == 0 else 64 
            # BUGFIX for more_to_drop
            indim = 3 if i == 0 else outdim
            outdim = 64
            
            # CustomDropout
            dropout_cond = i==dropout_block_id # whether this layer should dropout
            block_dropout_p = dropout_p if dropout_cond else 0.
            # more_to_drop
            if more_to_drop=='double' and dropout_cond:
                outdim = outdim*2
            
            #only pooling for first 4 layers
            B = ConvBlock(indim, outdim, pool = ( i <4 ), 
                          dropout_p=block_dropout_p)
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        
        # BUGFIX for more_to_drop
        self.final_feat_dim = outdim*5*5
#         self.final_feat_dim = 1600 # output = 64*5*5
        
        # for CustomDropout
        self.record_active_dropout()

    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNetS(nn.Module, CustomDropoutNet, MinGramDropoutNet): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True, dropout_p=0., dropout_block_id=3, more_to_drop=None, gram_bid = None):
        '''
        Args:
            dropout_block_id: could be {0|1|2|3}
            more_to_drop: could be {None|'double'}
            gram_bid (str|int): which block (index) should output Gram Matrix, follows dropout_bid if 'dropout'
        '''
        super(ConvNetS,self).__init__()
        trunk = []
        # TODO: trunk.append only select 1 channel
        for i in range(depth):
            '''input = 1*28*28 (see self.forward)
            TODO: compute the dimension, modify the decoder
            -> [64*28*28 -> 64*14*14]
            -> [64*14*14 -> 64*7*7]
            -> [64*7*7 -> 64*3*3]
            -> [64*3*3 -> 64*1*1]
            '''
            # BUGFIX for more_to_drop
            indim = 1 if i == 0 else outdim
            outdim = 64
            # CustomDropout
            dropout_cond = i==dropout_block_id # whether this layer should dropout
            block_dropout_p = dropout_p if dropout_cond else 0.
            # more_to_drop
            if more_to_drop=='double' and dropout_cond:
                outdim = outdim*2
            # for Gram Matrix block
            if gram_bid is None:
                gram_cond = False
            else:
                gm_bid = dropout_block_id if 'dropout' in gram_bid else gram_bid
                gram_cond = i==gm_bid # whether this block should output Gram Matrix
            #only pooling for first 4 layers
            B = ConvBlock(indim, outdim, pool = ( i <4 ), dropout_p=block_dropout_p) 
            trunk.append(B)
            
            # for Gram Matrix block
            if gram_cond: # currently assume only 1 block should output Gram matrix
                gram_trunk = trunk.copy()
                target_block = gram_trunk.pop() # remove & get last one
                target_block_b4_dropout = target_block.trunk
                gram_trunk.append(target_block_b4_dropout)
                self.trunk_to_gram_block = nn.Sequential(*gram_trunk)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        
        # BUGFIX for more_to_drop
        self.final_feat_dim = outdim
        
        # for CustomDropout
        self.record_active_dropout()
        # Gram matrix
        self.indim = 1 # BUGFIX for get_feature_map_for_gram
        self.min_gram_init(gram_bid)

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension (OOOOOMMMMMGGGG finally i see this NOW
        out = self.trunk(out)
        return out


class DeConvNetS(nn.Module): # for AE, input: flattened 64*1*1
    def __init__(self):
        super(DeConvNetS, self).__init__() # BUGFIX: not sure if correct (padding, output_padding, Tanh())
        self.decoder = nn.Sequential( # input: b, 64, 1, 1
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, output_padding=(1,1)),  # b, 64, 3, 3
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 3, 3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, output_padding=(1,1)),  # b, 64, 7, 7
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 7, 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 14, 14
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 14, 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 28, 28
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh() # BUGFIX: see how image is input to the model
        )
        
    def forward(self,x):
        out = x.view(x.size(0),64,1,1)
        out = self.decoder(out)
        out = out.repeat(1,3,1,1) # repeat for channel dimension. NOTE: NOT act like numpy.repeat
        out = img_standardize(out) # if don't want to standardize, then should cancel the normalize in get_composed_transform for omniglot
        return out

class DeConvNetS2(nn.Module):
    def __init__(self):
        super(DeConvNetS2, self).__init__()
        self.decoder = nn.Sequential( # input 64, 7, 7
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 14, 14
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 14, 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 28, 28
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh() # BUGFIX: see how image is input to the model
        )
        
    def forward(self,x):
        out = x.view(x.size(0),64,7,7)
        out = self.decoder(out)
        out = out.repeat(1,3,1,1) # repeat for channel dimension. NOTE: NOT act like numpy.repeat
        out = img_standardize(out)
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,5,5]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class DeResNet(nn.Module):
    maml = False
    def __init__(self, block, list_of_num_blocks, list_of_out_dims, flattened=True, indim=512):
        super(DeResNet,self).__init__()
        if flattened: # flattened input = 512
#             CT0 = nn.ConvTranspose2d(512, 512, kernel_size=7) # 512*7*7
#             bn0 = nn.BatchNorm2d(512)
            # TODO: upsample
            upsample0 = nn.Upsample(size=(7,7))
            
            
            relu = nn.ReLU()
#         else: # not flattened input = 512*7*7
#             raise ValueError('DeResNet only support flattened input. ')
        
#         init_layer(CT0) # useless
#         init_layer(bn0)
        if flattened:
#             trunk = [CT0, bn0]
            trunk = [upsample0]
        else:
            trunk = []
        
#         indim = 512
        n_stages = len(list_of_out_dims)
#         list_of_out_dims = [512, 256, 128, 64]
#         list_of_num_blocks = [2, 2, 2, 2]
#         block = DeSimpleBlock
        if self.maml:
            raise ValueError('DeResNet18 do not support maml.')
        else:
            for i in range(n_stages): # 4 stages
                for j in range(list_of_num_blocks[i]): # every stage is 2 for ResNet18
                    double_res = (i<=2) and (j==list_of_num_blocks[i]-1)
                    B = block(indim, list_of_out_dims[i], double_res)
                    trunk.append(B)
                    indim = list_of_out_dims[i] # NOT SURE
        CT1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, 
                                 padding=1, output_padding=1) # 64*56*56 -> 64*112*112
        bn = nn.BatchNorm2d(64)
        CT2 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, 
                                 padding=3, output_padding=1) # 64*112*112 -> 3*224*224
        tanh = nn.Tanh()
        
        init_layer(CT1) # useless
        init_layer(CT2) # useless
        init_layer(bn)
        trunk += [CT1, bn, CT2, tanh]
        self.trunk = nn.Sequential(*trunk)
        self.flattened = flattened
        
    def forward(self, x):
        if self.flattened:
            out = x.view(x.size(0), 512, 1, 1)
        else:
            out = x
        out = self.trunk(out)
        out = img_standardize(out)
#         print('out.shape =', out.shape, ', out.max() =', out.max(),' ,out.min() =', out.min())
#         print('out0.min = %s, out0.max = %s' % (out[:,0,:,:].min(),out[:,0,:,:].max()))
#         print('out1.min = %s, out1.max = %s' % (out[:,1,:,:].min(),out[:,1,:,:].max()))
#         print('out2.min = %s, out2.max = %s' % (out[:,2,:,:].min(),out[:,2,:,:].max()))
        return out

class ResNet(nn.Module, CustomDropoutNet, MinGramDropoutNet):
    maml = False #Default
    def __init__(self,block,list_of_num_blocks, list_of_out_dims, flatten = True, 
                dropout_p=0, dropout_block_id=3, more_to_drop=None, gram_sid=None): # not flatten only RelationNet?
        '''
        gram_sid is actually "gram_stage_id" in ResNet
        '''
        # list_of_num_blocks specifies number of blocks in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__() # input 224*224
        assert len(list_of_num_blocks)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False) # 64*112*112 (1)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False) # 64*112*112 (1)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 64*56*56 (1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4): # 4 stages
            for j in range(list_of_num_blocks[i]): # every stage is 2 for ResNet18
                ''' for ResNet 18:
                list_of_num_blocks = [2, 2, 2, 2], so num_layer is 2 for every stage
                block = SimpleBlock
                list_of_out_dims = [64, 128, 256, 512]
                
                SimpleBlock():
                    conv1(indim, outdim, kernel=3, stride=2 if half_res else 1, pad=1)
                    bn1()
                    conv2(outdim, outdim, kernel=3, stride=1, pad=1)
                    bn2()
                    if indim != outdim:
                        shortcut_layer = conv12(indim, outdim, kernel=1, stride=2 if half_res else 1)
                    else:
                        shortcut_layer = identity
                
                block 0-0: half_res=False,  (k=3, s=1, p=1) + (k=3, s=1, p=1),  64*56*56
                block 0-1: half_res=False, 64*56*56
                block 1-0: half_res=True, (k=3, s=2, p=1) + (k=3, s=1, p=1), 128*28*28 (1)
                block 1-1: half_res=False, 128*28*28
                block 2-0: half_res=True, 256*14*14 (1)
                block 2-1: half_res=False, 256*14*14
                block 3-0: half_res=True, 512*7*7 (1)
                block 3-1: half_res=False, 512*7*7
                '''
                half_res = (i>=1) and (j==0) # only stage 2 and 3's first block?
                # for CustomDropout
                dropout_cond = i==dropout_block_id # whether this layer should dropout
                block_dropout_p = dropout_p if dropout_cond else 0.
                # more_to_drop
                if more_to_drop=='double' and dropout_cond:
                    list_of_out_dims[i] = list_of_out_dims[i]*2 +1 # BUGFIX: +1 to avoid indim==outdim then affect 'half_res' 
                # for Gram Matrix
                if gram_sid is None:
                    gram_cond = False
                else:
                    gm_sid = dropout_block_id if 'dropout' in gram_sid else gram_sid
                    gram_cond = i==gm_sid # whether this block should output Gram Matrix
                
                B = block(indim, list_of_out_dims[i], half_res, dropout_p=block_dropout_p)
#                 B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
                
                # for Gram Matrix block
                is_last_block_of_stage = j==list_of_num_blocks[i]-1
                if gram_cond and is_last_block_of_stage: 
                    gram_trunk = trunk.copy()
                    target_block = gram_trunk.pop() # remove & get last one
                    target_block_b4_dropout = target_block.trunk
                    gram_trunk.append(target_block_b4_dropout)
                    self.trunk_to_gram_block = nn.Sequential(*gram_trunk)

        if flatten:
            avgpool = nn.AvgPool2d(7) # 512*1*1
            trunk.append(avgpool)
            trunk.append(Flatten()) # 512 for ResNet18
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7] # 512*7*7 for ResNet18 (RelationNet?)

        self.trunk = nn.Sequential(*trunk)
        
        # for CustomDropout
        self.record_active_dropout()
        # Gram matrix
        self.indim = 3 # BUGFIX for get_feature_map_for_gram
        self.min_gram_init(gram_sid)

    def forward(self,x):
        out = self.trunk(x)
        return out


class DeConvNet(nn.Module): # for AE, input: flattened 64*5*5
    def __init__(self):
        super(DeConvNet, self).__init__() # BUGFIX: not sure if correct (padding, output_padding, Tanh())
        self.decoder = nn.Sequential( # input: b, 64, 5, 5
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 10, 10
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 10, 10
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, output_padding=(1,1)),  # b, 64, 21, 21
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 21, 21
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 42, 42
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 42, 42
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 84, 84
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # b, 3, 84, 84
            nn.Tanh() # BUGFIX: see how image is input to the model
        )
        
    def forward(self,x):
        out = x.view(x.size(0),64,5,5)
        out = self.decoder(out)
        out = img_standardize(out)
        return out

class DeFCNet(nn.Module): # for AE
    def __init__(self):
        super(DeFCNet, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64*5*5,500), 
            nn.ReLU(inplace=True), 
            nn.Linear(500,3*84*84),
            nn.Tanh()
        )
    
    def forward(self,x):
        out = self.decoder(x)
        out = out.view(out.size(0),3,84,84)
        out = img_standardize(out)
        return out

class DeConvNet2(nn.Module):
    def __init__(self):
        super(DeConvNet2, self).__init__()
        self.decoder = nn.Sequential( # input 64, 21, 21
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 42, 42
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 42, 42
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # b, 64, 84, 84
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # b, 3, 84, 84
            nn.Tanh()
        )
        
    def forward(self,x):
        out = x.view(x.size(0),64,21,21)
        out = self.decoder(out)
        out = img_standardize(out)
        return out

# def Conv4():
#     return ConvNet(4)

# def Conv4Drop(dropout_p=0.):
#     return ConvNet(4,dropout_p=dropout_p)

def Conv4(dropout_p=0., dropout_block_id=3, more_to_drop=None):
    return ConvNet(4,dropout_p=dropout_p, dropout_block_id=dropout_block_id, more_to_drop=more_to_drop)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

# def Conv4S():
#     return ConvNetS(4)

# def Conv4SDrop(dropout_p=0.):
#     return ConvNetS(4,dropout_p=dropout_p)

def Conv4S(dropout_p=0., dropout_block_id=3, more_to_drop=None, gram_bid=None):
    return ConvNetS(
        4, dropout_p=dropout_p, dropout_block_id=dropout_block_id, 
        more_to_drop=more_to_drop, 
        gram_bid=gram_bid)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10(flatten=True, dropout_p=0, dropout_block_id=10, more_to_drop=None, gram_bid=None):
    # WTF i dunno why SimpleBlock cost less memory
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, 
                 dropout_p=dropout_p, dropout_block_id=dropout_block_id, 
                  more_to_drop=more_to_drop, gram_sid=gram_bid)
#     return ResNet(BottleneckBlock, [1,1,1,1],[64,128,256,512], flatten)

def DeResNet10(flatten=True):
    return DeResNet(DeSimpleBlock, [1,1,1,1], [512,256,128,64], flatten, indim=512)

def DeResNet10_2(flatten=False):
    return DeResNet(DeSimpleBlock, [1,1], [128,64], flatten, indim=128)

def ResNet18(flatten = True, dropout_p=0, dropout_block_id=10):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, flatten, 
                 dropout_p=dropout_p, dropout_block_id=dropout_block_id)
#     return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def DeResNet18(flatten=True):
    return DeResNet(DeSimpleBlock, [2,2,2,2], [512,256,128,64], flatten, indim=512)

def ResNet34( flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)




