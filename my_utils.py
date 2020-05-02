import torch
import numpy as np
import random
import configs
import datetime

import time
import os
import tensorflow as tf

from methods.meta_template import MetaTemplate
from methods.protonet import ProtoNet
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
# global_datasets = [] # for multi-processsing
import logging

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
        
    def end(self, summary=False):
        
        if summary:
            self.summary()
    
    def summary(self):
        pass

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


# def most_frequent(List): 
#     return max(set(List), key = List.count)

def feature_evaluation(cl_feature_each_candidate, model, params, n_way = 5, n_support = 5, n_query = 15, recons_func = None):
    ''' (for test.py) sample ONE episode to do evaluation
    :param cl_feature_each_candidate: list of dictionary (len=1 if n_test_candidates is None), keys=label_idx, values = all extracted features
    :param recons_func: temporary no use
    :return: accuracy (%)
    '''
#     def select_class_with_sanity(class_list, cl_feature_each_candidate): # no need this after BUGFIX
#         select_class = random.sample(class_list,n_way)
#         # sanity check
#         while True:
#             sanity = True
#             for cl in select_class:
#                 for cl_feature_dict in cl_feature_each_candidate:
#                     img_feat = cl_feature_dict[cl]
#                     if len(img_feat)<n_support+n_query:
#                         sanity = False
# #                         print('select_class sanity check failed, len(img_feat) =',len(img_feat),'<',n_support+n_query,'resample classes...')
#                         select_class = random.sample(class_list,n_way)
#             if sanity:
#                 break
#         return select_class
    
    def get_all_perm_features(select_class, cl_feature_dict, perm_ids_dict):
        '''
        Return:
            z_all (ndarray): shape=(n_way, n_support+n_query, ???)
        '''
        z_all  = []
        for cl in select_class:
            img_feat = cl_feature_dict[cl]
            
            n_data = n_support+n_query
#             n_data = n_support+n_query if n_support+n_query<=len(img_feat) else len(img_feat)
            # get shuffled idx inside one-class data
            perm_ids = perm_ids_dict[cl] 
            # stack each batch
            z_all.append( [ np.squeeze(img_feat[perm_ids[i]]) for i in range(n_data) ] )
    
        z_all = np.array(z_all)
        return z_all
    
    def get_forward_outputs(model, z_all):
        ''' get query set forward_outputs
        Args:
            z_all (torch.Tensor): z_support & z_query
        Return:
            forward_outputs (torch.Tensor): query set forward_outputs (not probs)
        '''
        adaptation = params.adaptation
        if isinstance(model, ProtoNet):
            if adaptation:
                forward_outputs  = model.set_forward_adaptation(z_all, is_feature = True)
            else:
                forward_outputs  = model.set_forward(z_all, is_feature = True)
        elif isinstance(model, BaselineFinetune):
            forward_outputs = model.set_forward(z_all, is_feature = True)
        elif isinstance(model, BaselineTrain):# or isinstance(model, BaselineFinetune):
            raise ValueError('not support Baseline. ')
            forward_outputs = model.forward() # only support original data (not feature)
        else:
            raise ValueError('Unsupported method.')
        return forward_outputs
    
    def get_pred(model, z_all, prob=False):
        '''
        Args:
            z_all (torch.Tensor): z_support & z_query
        Return:
            pred (ndarray): query set prediction
                shape=(n_query*n_way, n_way) if prob
                shape=(n_query*n_way, ) if not prob
        '''
#         if isinstance(model, MetaTemplate):
#             if adaptation:
#                 forward_outputs  = model.set_forward_adaptation(z_all, is_feature = True)
#             else:
#                 forward_outputs  = model.set_forward(z_all, is_feature = True)
#         else:
#             raise ValueError('Unsupported method.')
        forward_outputs = get_forward_outputs(model, z_all)
        if prob:
            pred = model.forwardout2prob(forward_outputs)
            pred = pred.data.cpu().numpy()
        else:
            forward_outputs = forward_outputs.data.cpu().numpy()
            pred = forward_outputs.argmax(axis = 1)
        return pred
    
    def get_result(model, z_all, n_way, n_support, n_query, metric):
        
        # make model can parse feature correctly
        model.n_support = n_support
        model.n_query = n_query
        
        if metric == 'acc':
            pred = get_pred(model, z_all)
            y = np.repeat(range( n_way ), model.n_query )
            result = np.mean(pred == y)*100
        elif metric == 'loss':
#             if isinstance(model, MetaTemplate):
#                 forward_outputs = model.set_forward(z_all, is_feature=True)
#             elif isinstance(model, BaselineTrain) or isinstance(model, BaselineFinetune):
#                 raise ValueError('not support Baseline yet. ')
#                 forward_outputs = model.forward() # only support original data (not feature)
#             else:
#                 raise ValueError('Unsupported method.')
            forward_outputs = get_forward_outputs(model, z_all)
            result = model.forwardout2loss(forward_outputs)
        else:
            raise ValueError('Unknown metric: %s'%(metric))
        return result
    
    def get_result_loocv(model, z_all, n_way, metric, the_one='val'):
        '''
        Actually not really leave-one-out but "leave-one-out per-class"!!!
        Args:
            z_all (torch.Tensor): shape=(n_way, n_data, feature_dim) contain sub_support & sub_query set
            metric (str): 'acc' or 'loss'
        '''
        
        n_data_per_class = z_all.size(1) # not sure lol, usually 5, also n_fold
        k_fold = n_data_per_class
#         n_way = z_all.size(0)
        
        n_support_cv = 1 if the_one=='train' else n_data_per_class - 1
        n_query_cv = n_data_per_class - n_support_cv
        
        swap_the_one_per_class = 0 if the_one=='train' else n_data_per_class-1 # first or last
        # TODO: NO NEED THIS if all use get_results. (make model can parse feature correctly)
        # TODO: or NO NEED function args "n_support" & "n_query" if already here
        model.n_support = n_support_cv
        model.n_query = n_query_cv
        
        result_cv = [0]*k_fold
        original_ids = [k for k in range(k_fold)]
        
        for k in range(k_fold):
            # get swapped features
            swapped_ids = original_ids.copy()
            swapped_ids[swap_the_one_per_class] = k
            swapped_ids[k] = swap_the_one_per_class
            z_swapped = torch.index_select(z_all, 1, torch.LongTensor(swapped_ids).cuda())
#             z_swapped = z_swapped.contiguous() # try to fix bug but failed
        
#             if metric == 'acc':
#                 pred = get_pred(model, z_swapped)
#                 y = np.repeat(range( n_way ), n_query_cv )
#                 acc = np.mean(pred == y)*100
#                 result_cv[k] = acc
#             elif metric == 'loss':
            result = get_result(
                model=model, z_all=z_swapped, 
                n_way=n_way, n_support=n_support_cv, n_query=n_query_cv, 
                metric=metric
            )
            result_cv[k]=result
        
        return sum(result_cv)/k_fold
    
    
#     adaptation = params.adaptation
    
    if params.n_test_candidates is None: # common setting
        class_list = cl_feature_each_candidate[0].keys()
        cl_feature_dict = cl_feature_each_candidate[0] # list only have 1 element
        
        select_class = random.sample(class_list,n_way)

        
        perm_ids_dict = {} # permutation indices of each selected class
        # initialize perm_ids_dict
        for cl in select_class:
            # I think len(img_feat) is always n_support+n_query so i don't write len(img_feat) # nonono BUGBUGBUG
            img_feat = cl_feature_dict[cl]
            n_data = len(img_feat)
            perm_ids = np.random.permutation(n_data).tolist() # get shuffled idx inside one-class data
            perm_ids_dict[cl] = perm_ids
            
        z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
        z_all = torch.from_numpy(z_all) # z_support & z_query
        
        # here should be acc
        acc = get_result(
            model=model, z_all=z_all, 
            n_way=n_way, n_support=n_support, n_query=n_query, metric='acc')
    
    else: # n_test_candidates setting
        assert params.n_test_candidates == len(cl_feature_each_candidate), "features & params mismatch."
        
        class_list = cl_feature_each_candidate[0].keys()
        
        select_class = random.sample(class_list,n_way)
#         select_class = select_class_with_sanity(class_list, cl_feature_each_candidate) # no need to fix bug this way now 
        
        perm_ids_dict = {} # store the permutation indices of each class
        sub_result_each_candidate = [] # store sub_query set result of each candidate
        
        # get shuffled data idx in each class (of all features?)
        for cl in select_class:
            tmp_cl_feature_dict = cl_feature_each_candidate[0] # i think all candidates have the same n_data
            img_feat = tmp_cl_feature_dict[cl]
            # I think len(img_feat) is always n_support+n_query so i don't write len(img_feat) NONONO BUGBUGBUG
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            perm_ids_dict[cl] = perm_ids
        
        # here seems took most of the time cost
        for n in range(params.n_test_candidates): # for each candidate
            cl_feature_dict = cl_feature_each_candidate[n] # features of the candidate

            z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
            z_all = torch.from_numpy(z_all) # z_support & z_query
                        
            # reset back
            model.n_support = n_support
            model.n_query = n_query
            z_support, z_query  = model.parse_feature(z_all,is_feature=True)# shape:(n_way, n_data, *feature_dims)
            z_support   = z_support.contiguous() # TODO: what this means???
#             z_support_cpu = z_support.data.cpu().numpy()
            
            if model.change_way:
                model.n_way  = z_support.size(0)
            # TODO: tunable n_sub_support
            # leave-one-out (per-class) cross validation
            loopccv_the_one = 'val' # None, 'train', 'val'
            if loopccv_the_one is not None:
                sub_result = get_result_loocv(model=model, z_all=z_support, 
                                        n_way=n_way, the_one=loopccv_the_one, metric=params.candidate_metric)
            else: # testing without loopccv
                n_sub_support = 1 # 1 | n_support-1 | n_support//2, 1 seems better?
                n_sub_query = n_support - n_sub_support # those who are rest
                sub_result = get_result(
                    model=model, z_all=z_support, 
                    n_way=n_way, n_support=n_sub_support, n_query=n_sub_query, 
                    metric=params.candidate_metric)
            
            sub_result_each_candidate.append(sub_result)
            
        n_ensemble = 1 if params.frac_ensemble == None else int(params.frac_ensemble*params.n_test_candidates)
        
        # get ensemble ids
        sub_result_each_candidate = np.array(sub_result_each_candidate)
        if params.candidate_metric == 'acc':
            sorted_candidate_ids = np.argsort(-sub_result_each_candidate) # in descent order
        elif params.candidate_metric == 'loss':
            sorted_candidate_ids = np.argsort(sub_result_each_candidate) # in ascent order
        else:
            raise ValueError('Unknown candidate_metric: %s'%(metric))
        elected_ids = sorted_candidate_ids[:n_ensemble] # TODO: rename elected_candidate to winners/superiors
        all_preds = []
        
        # reset back
        model.n_support = n_support
        model.n_query = n_query
        # repeat procedure of common setting to get query prediction
        for elected_id in elected_ids:
            cl_feature_dict = cl_feature_each_candidate[elected_id]
            z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
            z_all = torch.from_numpy(z_all) # z_support & z_query
            
            if params.ensemble_strategy=='avg_prob':
                pred = get_pred(model, z_all, prob=True)
            elif params.ensemble_strategy=='vote':
                pred = get_pred(model, z_all)
            else:
                raise ValueError('Invalid ensemble_strategy: %s'%(params.ensemble_strategy))
            all_preds.append(pred)
        
        # all_preds shape=(n_ensemble, n_query*n_way) for 'vote'
        # all_preds shape=(n_ensemble, n_query*n_way, n_way) for 'avg_prob'
        all_preds = np.array(all_preds)
        
        if params.ensemble_strategy=='vote':
            all_preds = all_preds.T # shape:(n_query*n_way, n_ensemble) for 'vote'
            ensemble_preds = [np.argmax(np.bincount(preds)) for preds in all_preds]
            ensemble_preds = np.array(ensemble_preds)
        elif params.ensemble_strategy=='avg_prob':
            ensemble_preds = all_preds.mean(axis=0) # shape=(n_query*n_way, n_way)
#             print('avg_prob/ensemble_preds.shape (after mean)', ensemble_preds.shape)
            ensemble_preds = np.argmax(ensemble_preds, axis=1) # shape=(n_query*n_way)
#             print('avg_prob/ensemble_preds.shape (after argmax)', ensemble_preds.shape)
        
        y = np.repeat(range( n_way ), n_query )
        acc = np.mean(ensemble_preds == y)*100
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

def set_random_seed(seed): # TODO: keras, theano and so forth
#     if seed is None:
#         pass
#         tf.set_random_seed()
#         np.random.seed()
#         random.seed()
#         torch.manual_seed()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def del_keys(d: dict, keys):
    for key in keys:
        if key in d:
            del d[key]



