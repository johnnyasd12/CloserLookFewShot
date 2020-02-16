import torch
import numpy as np
import random
import configs
import datetime

import time
import os
import tensorflow as tf

# global_datasets = [] # for multi-processsing

def describe(obj, obj_str): # support ndarray, tf.Tensor, dict, iterable
    # exec('global '+obj_str)
    # exec('obj = '+obj_str)
    obj_type = type(obj)
    print('='*5
        , obj_str
        , obj_type
        , '='*5
        , end = ' '
        )
    if obj_type == np.ndarray:
        print('\nshape:',obj.shape)
        print('(min, max) = ('+str(obj.min())+', '+str(obj.max())+')')
        print('(mean, median) = ('+str(obj.mean())+', '+str(np.median(obj))+')')

    elif tf.contrib.framework.is_tensor(obj):
        obj_shape = obj.get_shape().as_list() # get_shape().num_elements() can get sum
        print(obj_shape)
    elif isinstance(obj, dict):
        print()
        for key, content in obj.items():
            print(key, ':', content)
    else:
        try:
            iterator = iter(obj)
        except TypeError:
            # not iterable
            print(obj)
        else:
            # iterable
            print(len(obj))
    print('='*10)

class Timer2:
    def __init__(self, name, enabled, verbose=1): 
        # TODO: verbose (then delete enabled)
        # TODO: accumulate the time interval of all iterations
        '''
        :param verbose: 0 to print NOTHING, 1 to print common, 2 to print detail
        '''
#         self.name = name
        self.start_t = {}
        self.last_t = {}
        self.start_t['name'] = name
        self.start_t['time'] = time.time()
        self.start_t['clock'] = time.clock()
        self.start_t['process_time'] = time.process_time()
        self.last_t = self.start_t.copy()
        
        self.end_t = {}
        self.duration = {}
        self.duration['from'] = name
        self.enabled = enabled
        self.verbose = verbose
        self.duration_list = [] # TODO: 
        if self.enabled:
            print(name, 'started')
    
    def __call__(self, name):
        self.keys = ['time', 'clock', 'process_time']
        self.end_t['name'] = name
        self.end_t['time'] = time.time()
        self.end_t['clock'] = time.clock()
        self.end_t['process_time'] = time.process_time()
        
        self.duration['from'] = self.last_t['name']
        self.duration['to'] = name
        for k in self.keys:
            self.duration[k] = self.end_t[k] - self.last_t[k]
        
        if self.enabled:
#         print('since last time:', self.duration)
            print('Duration from "' + self.duration['from'] + '" to "' + self.duration['to'] +'": ')
            print('time:', self.duration['time'], ', process time:', self.duration['process_time'])
        self.last_t = self.end_t.copy()

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    This class is modified from https://github.com/Bjarten/early-stopping-pytorch/blob/63d59203ffe29fe111296c705c4eb0958922eaf7/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): Procedure for determining the best score.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.mode = mode

        if self.mode == 'min':
            self.criterion = np.less
            self.delta = - delta
            self.best_score = np.Inf

            self.vocab = {'score': 'loss', 'comportement': 'decreased'}

        elif self.mode == 'max':
            self.criterion = np.greater
            self.delta = delta
            self.best_score = np.NINF

            self.vocab = {'score': 'metric', 'comportement': 'increased'}

        else:
            raise ValueError(
                "mode only takes as value in input 'min' or 'max'")

    def __call__(self, score, model):
        """Determines if the score is the best and saves the model if so.
           Also manages early stopping.
        Arguments:
            score (float): Value of the metric or loss.
            model: Pytorch model
        """
        if np.isinf(self.best_score):
            self.best_score = score
#             self.save_checkpoint(score, model)

        elif self.criterion(score, self.best_score + self.delta):

            self.best_score = score
#             self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
#             print(
#                 f'EarlyStopping counter: {self.counter} out of {self.patience}'
#             )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        '''Saves the model when the score satisfies the criterion.'''
        if self.verbose:
            score_name = self.vocab['score']
            comportement = self.vocab['comportement']
            print(
                f'Validation {score_name} {comportement} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), 'checkpoint.pt')

def plot_PIL(img):
    pass



def feature_evaluation(cl_feature_dict_ls, model, params, n_way = 5, n_support = 5, n_query = 15, recons_func = None):
    ''' (for test.py) sample ONE episode to do evaluation
    :param cl_feature_dict_ls: list of dictionary (len=1 if n_test_candidates is None), keys=label_idx, values = all extracted features
    :param recons_func: temporary no use
    :return: accuracy (%)
    '''
    adaptation = params.adaptation
    
    if params.n_test_candidates is None: # common setting
        class_list = cl_feature_dict_ls[0].keys()
        select_class = random.sample(class_list,n_way) # needed to be func input
        z_all  = []
        cl_feature_dict = cl_feature_dict_ls[0]
        for cl in select_class:
            img_feat = cl_feature_dict[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist() # get shuffled idx inside one-class data
            z_all.append( [ np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] ) # stack each batch
        z_all = torch.from_numpy(np.array(z_all)) # z_support & z_query
        
        model.n_query = n_query
        if adaptation:
            scores  = model.set_forward_adaptation(z_all, is_feature = True)
        else:
            scores  = model.set_forward(z_all, is_feature = True)
        pred = scores.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc = np.mean(pred == y)*100 
    else: # n_test_candidates setting
        assert params.n_test_candidates == len(cl_feature_dict_ls), "features & params mismatch."
        
        class_list = cl_feature_dict_ls[0].keys()
        select_class = random.sample(class_list,n_way)
        perm_ids_dict = {} # store the permutation indices of each class
        sub_acc_ls = [] # store sub_query set accuracy of each candidate
        
        # get shuffled data idx in each class (of all features?)
        for cl in select_class:
#             tmp_cl_feature_dict = cl_feature_dict_ls[0]
#             img_feat = tmp_cl_feature_dict[cl]
            # I think len(img_feat) is always n_support+n_query so i don't write len(img_feat)
            perm_ids = np.random.permutation(n_support+n_query).tolist()
            perm_ids_dict[cl] = perm_ids
        
        for n in range(params.n_test_candidates): # for each candidate
            z_all  = []
            cl_feature_dict = cl_feature_dict_ls[n]
            for cl in select_class: # for each class
                img_feat = cl_feature_dict[cl]
                perm_ids = perm_ids_dict[cl]
                # stack each batch
                z_all.append( [ np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
            z_all = np.array(z_all)
            z_all = torch.from_numpy(z_all) # z_support & z_query
                        
            # reset back
            model.n_support = n_support
            model.n_query = n_query
            z_support, z_query  = model.parse_feature(z_all,is_feature=True)# shape:(n_way, n_data, *feature_dims)
            z_support   = z_support.contiguous() # shape = (n_way, n_shot, *feature_dims)
            z_support_cpu = z_support.data.cpu().numpy()
            
            # TODO: tunable n_sub_support
            n_sub_support = n_support//2 # 5//2 = 2
            n_sub_query = n_support - n_sub_support # 5-2 = 3
            model.n_support = n_sub_support
            model.n_query = n_sub_query
            
#             perm_id = np.random.permutation(n_support).tolist()
#             z_supp_perm = np.array([z_support_cpu[i,perm_id,:] for i in range(z_support.size(0))])
#             print('z_supp_perm.shape:',z_supp_perm.shape) # (n_way, n_shot, *feature_dims)
#             z_supp_perm = torch.Tensor(z_supp_perm).cuda() # support set, permutation is for the in-class samples
            if model.change_way:
#                 model.n_way  = z_supp_perm.size(0)
                model.n_way  = z_support.size(0)
            y_sub_query = np.repeat(range( model.n_way ), n_sub_query ) # sub_query set label
#             y_sub_query = torch.from_numpy(y_sub_query)
            
            if adaptation:
#                 scores  = model.set_forward_adaptation(z_supp_perm, is_feature = True)
                scores  = model.set_forward_adaptation(z_support, is_feature = True)
            else:
#                 scores  = model.set_forward(z_supp_perm, is_feature = True)
                scores  = model.set_forward(z_support, is_feature = True)
            pred = scores.data.cpu().numpy().argmax(axis = 1)
            sub_acc = np.mean(pred == y_sub_query)*100
            sub_acc_ls.append(sub_acc)
            
        n_ensemble = 1 if params.frac_ensemble == None else int(params.frac_ensemble*params.n_test_candidates)
        # reset back
        model.n_support = n_support
        model.n_query = n_query
        
        # TODO: argsort
        sub_acc_ls = np.array(sub_acc_ls)
        sorted_candidate_id = np.argsort(-sub_acc_ls) # in descent order
        elected_candidate_id = sorted_candidate_id[:n_ensemble]
        # repeat procedure of common setting to get query prediction

        acc = None
    
    return acc

def set_gpu_id(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def to_device(tensor):
    if configs.gpu_id:
        device = torch.device('cuda:'+str(configs.gpu_id))
        tensor = tensor.to(device)
    else:
        tensor = tensor.cuda()
    return tensor

def get_time_now():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def cl_file_to_z_all(cl_feature_dict, n_way, n_support, n_query):
    class_list = cl_feature_dict.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_feature_dict[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all)) # z_support & z_query
    return z_all