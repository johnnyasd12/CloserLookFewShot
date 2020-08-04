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
from methods.relationnet import RelationNet
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.matchingnet import MatchingNet
# global_datasets = [] # for multi-processsing
import logging

import matplotlib.pyplot as plt

# to multi-frac_ensemble
import copy

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
            print('Duration \nfrom \t"' + self.duration['from'] + '" \nto\t"' + self.duration['to'] +'": ')
            print('time:', self.duration['time'], ', process time:', self.duration['process_time'])
            print()
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

def feature_evaluation(cl_feature_each_candidate, model, params, n_way = 5, n_support = 5, n_query = 15, cl_filepath = None):
    ''' (for test.py) sample ONE episode to do evaluation
    :param cl_feature_each_candidate: list of dictionary (len=1 if n_test_candidates is None), keys=label_idx, values = all extracted features
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
    
    def get_task_paths(select_class, cl_feature_dict, perm_ids_dict):
        '''
        Return:
            task_paths (dict): keys = 'cl00_su00'|...|'cl04_qu14'
        '''
        task_paths = {}
        n_data_per_class = n_support + n_query
        
        for cl_idx, cl in enumerate(select_class): # for each class
            task_class_str = 'c'+str(cl_idx).zfill(2) # c00|01|02|03|04
            paths = cl_filepath[cl]
            # get shuffled idx inside one-class data
            perm_ids = perm_ids_dict[cl] # len = n_data_per_class
            
            for i in range(n_data_per_class):
                if i < n_support:
                    task_split_str = 'su' + str(i).zfill(2) # su00|01|02|03|04
                else:
                    task_split_str = 'qu' + str(i-n_support).zfill(2) # qu00|...|14
                task_str = task_class_str + '_' + task_split_str # c03_su02
                path = paths[perm_ids[i]]
                task_paths[task_str] = {'path':path, 'pred':None}
        
        return task_paths
    
#     def record_acc_and_task_pred(z_all, task_data):
    def record_acc_and_task_pred(pred_prob, task_data):
        n_data_per_class = n_support + n_query
        pred = pred_prob.argmax(axis = 1)
        # record paths and preds
        for cl_idx in range(n_way):
            task_class_str = 'c'+str(cl_idx).zfill(2) # c00|01|02|03|04
            for i in range(n_data_per_class):
                # record prediction (only for query set)
                if i >= n_support:
                    task_split_str = 'qu' + str(i - n_support).zfill(2) # qu00|...|14
                    task_str = task_class_str + '_' + task_split_str # c03_su02
                    pred_idx = cl_idx * n_query + (i - n_support)
                    task_data[task_str]['pred'] = pred[pred_idx]
                    task_data[task_str]['pred_prob'] = pred_prob[pred_idx]
        
        # which preds are correct
        y = np.repeat(range( n_way ), n_query )
        correct = pred == y
        acc = np.mean(correct)*100
        task_data['acc'] = acc
    
    def get_all_perm_paths(select_class, perm_ids_dict):
        path_all = []
        n_data_per_class = n_support + n_query
        for cl in select_class:
            paths = cl_filepath[cl]
            perm_ids = perm_ids_dict[cl]
            path_all.append(np.array([paths[perm_ids[i]] for i in range(n_data_per_class)]))
        path_all = np.array(path_all)
        return path_all
    
    def get_all_perm_features(select_class, cl_feature_dict, perm_ids_dict):
        '''
        Return:
            z_all (ndarray): shape=(n_way, n_support+n_query, ???)
        '''
        z_all  = []
#         path_all = []
        n_data_per_class = n_support + n_query
        for cl in select_class:
            img_feat = cl_feature_dict[cl]
#             paths = cl_filepath[cl]
            
#             n_data_per_class = n_support+n_query if n_support+n_query<=len(img_feat) else len(img_feat)
            
            # get shuffled idx inside one-class data
            perm_ids = perm_ids_dict[cl] 
            # stack each batch
            z_all.append( [ np.squeeze(img_feat[perm_ids[i]]) for i in range(n_data_per_class) ] )
#             if cl_filepath is not None:
#                 path_all.append(np.array([paths[perm_ids[i]] for i in range(n_data_per_class)]))
    
        z_all = np.array(z_all)
#         if cl_filepath is None:
        return z_all
#         else:
#             path_all = np.array(path_all)
#             return z_all, path_all
    
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
        elif isinstance(model, RelationNet):
            forward_outputs = model.set_forward(z_all, is_feature = True)
        elif isinstance(model, MatchingNet):
            forward_outputs = model.set_forward(z_all, is_feature = True)
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
        forward_outputs = get_forward_outputs(model, z_all)
        if prob:
            pred = model.forwardout2prob(forward_outputs)
            pred = pred.data.cpu().numpy()
        else:
            forward_outputs = forward_outputs.data.cpu().numpy()
            pred = forward_outputs.argmax(axis = 1)
        return pred
    
    def show_acc_w_data(model, z_all, path_all, n_way, n_support, n_query):
        
        # make model can parse feature correctly
        model.n_support = n_support
        model.n_query = n_query
        # get predictions and local labels
        prob = get_pred(model, z_all, prob=True)
        pred = prob.argmax(axis = 1)
        y = np.repeat(range(n_way), model.n_query)
        # which preds are correct
        correct = pred == y
        acc = np.mean(correct)*100
        
        ##### drawing #####
        n_col = n_way
        n_row = n_support + n_query
        unit_size = 50
        fig, axarr = plt.subplots(n_row, n_col, figsize=(unit_size, unit_size))
#         fig.tight_layout()
        for row in range(n_row): # for each data per class
            for col in range(n_col): # for each class
                idx = col*n_way + row
                path = path_all[col][row]
                img = plt.imread(path)
                axarr[row, col].imshow(img, aspect=1) # set aspect to avoid showing with actual size
        plt.show()
        
        return acc
    
    def get_cand_prob_diversities(candidate_supp_prob, measure):
        '''
        Args:
            candidate_supp_prob (ndarray): candidate prob arrays with shape: (n_test_candidates, n_way, n_shot, n_way)
            measure (str): 'abs_diff', 'abs_diff_ignore_true_class', 'cross_entropy'
        '''
        candidate_probs = candidate_supp_prob.reshape(params.n_test_candidates, n_way*n_shot*n_way)
        
        
        for i, supp_prob in enumerate(candidate_probs):
            i_candidate_diff = candidate_probs - supp_prob
            print('i_candidate_diff:', i_candidate_diff)
            hohohoho
        
        
    
    def get_result(model, z_all, n_way, n_support, n_query, metric, 
                   return_prob=False):
        '''
        Returns:
            prob (ndarray): shape=(n_way*n_query, n_way)???
        '''
        # make model can parse feature correctly
        model.n_support = n_support
        model.n_query = n_query
        
        forward_outputs = get_forward_outputs(model, z_all)
        if return_prob:
            prob = model.forwardout2prob(forward_outputs).data.cpu().numpy()
            pred = prob.argmax(axis = 1)
        else:
            pred = forward_outputs.data.cpu().numpy().argmax(axis=1)
        
        if metric == 'acc':
#             pred = get_pred(model, z_all)
            y = np.repeat(range( n_way ), model.n_query )
#             result = np.mean(pred == y)*100
            sample_accs = (pred == y)*100
            acc = np.mean(sample_accs)
            sample_results = sample_accs
            result = acc
        elif metric == 'loss':
#             result = model.forwardout2loss(forward_outputs)
            loss, sample_losses = model.forwardout2loss(forward_outputs, return_respective=True)
            sample_results = sample_losses.data.cpu().numpy()
            result = loss.data.cpu().numpy()
        else:
            raise ValueError('Unknown metric: %s'%(metric))
        
        ##### debug
#         print('my_utils/get_result()/sample_results:', type(sample_results), sample_results)
        
        if return_prob:
            return result, sample_results, prob
        else:
            return result, sample_results
    
    def get_result_loocv(model, z_all, n_way, metric, the_one='loopccv_one_val', return_all_probs=False):
        '''
        Actually not really leave-one-out but "leave-one-out per-class"!!!
        Args:
            z_all (torch.Tensor): shape=(n_way, n_data, feature_dim) contain sub_support & sub_query set
            metric (str): 'acc'|'loss'|'loss_bagging'
            the_one (str): 'loopccv_one_val'|'loopccv_one_train'
        '''
        
#         if 'loss' in metric: # 'loss_bagging'
#             metric = 'loss'
#         elif 'acc' in metric:
#             metric = 'acc'
        
        n_data_per_class = z_all.size(1) # not sure lol, usually 5, also n_fold
        k_fold = n_data_per_class
#         n_way = z_all.size(0)
        
        n_support_cv = 1 if the_one=='loopccv_one_train' else n_data_per_class - 1
        n_query_cv = n_data_per_class - n_support_cv
        
        swap_the_one_per_class = 0 if the_one=='loopccv_one_train' else n_data_per_class-1 # first or last
        # TODO: NO NEED THIS if all use get_results. (make model can parse feature correctly)
        # TODO: or NO NEED function args "n_support" & "n_query" if already here
        model.n_support = n_support_cv
        model.n_query = n_query_cv
        
        result_cv = [0]*k_fold
        original_ids = [k for k in range(k_fold)]
        
        if return_all_probs:
    #         supp_probs = [[None]*n_data_per_class for _ in range(n_way)] # (n_way, n_data_per_class, n_classes(=n_way))
            supp_probs = np.zeros((n_way, n_data_per_class, n_way)) # (n_way, n_data_per_class, n_classes(=n_way))
        supp_results = np.zeros((n_way, n_data_per_class))
#         print('supp_probs:', supp_probs)
        
        for k in range(k_fold): # k_fold = n_data_per_class
            # get swapped features
            swapped_ids = original_ids.copy()
            swapped_ids[swap_the_one_per_class] = k
            swapped_ids[k] = swap_the_one_per_class
            z_swapped = torch.index_select(z_all, 1, torch.LongTensor(swapped_ids).cuda())
        
            if return_all_probs:
                result, sample_results, prob = get_result(
                    model=model, z_all=z_swapped, 
                    n_way=n_way, n_support=n_support_cv, n_query=n_query_cv, 
                    metric=metric, return_prob = True
                )
            else:
                result, sample_results = get_result(
                    model=model, z_all=z_swapped, 
                    n_way=n_way, n_support=n_support_cv, n_query=n_query_cv, 
                    metric=metric, return_prob = False
                )
            result_cv[k]=result
#             print('get_result_loocv()/prob.shape:', prob.shape) # shape: (n_data(=n_way), n_classes(=n_way))
            for way in range(n_way):
                if return_all_probs:
                    supp_probs[way, k, :] = prob[way]
                supp_results[way, k] = sample_results[way]

#         print('supp_results after loop:', supp_results)
#         print('supp_probs after loop:', supp_probs)
#         print('supp_probs.sum(axis=2) after loop:', supp_probs.sum(axis=2)) # sum is 1, no problem
        
        if return_all_probs:
            return supp_results, supp_probs
#             return supp_results.sum()/(n_way*n_data_per_class), supp_probs
#             return sum(result_cv)/k_fold, supp_probs
        else:
            return sum(result_cv)/k_fold
    
    
    
    class_list = cl_feature_each_candidate[0].keys()
    select_class = random.sample(class_list,n_way)
    
    # get shuffled data idx in each class (of all features?)
    perm_ids_dict = {} # store the permutation indices of each class
    tmp_cl_feature_dict = cl_feature_each_candidate[0] # to compute n_data (all candidates should have the same n_data)
    for cl in select_class:
        img_feats = tmp_cl_feature_dict[cl]
        n_data = len(img_feats)
        perm_ids = np.random.permutation(n_data).tolist()
        perm_ids_dict[cl] = perm_ids

    # task_paths to draw_task (utilize save_features path)
    task_paths = get_task_paths(
        select_class=select_class, cl_feature_dict=tmp_cl_feature_dict, perm_ids_dict=perm_ids_dict)
    
    task_data = dict(**task_paths) # stores all info, e.g. acc, img_path, img_is_correct, etc.
    
    if params.n_test_candidates is None: # common setting
        cl_feature_dict = cl_feature_each_candidate[0] # list only have 1 element
            
        z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
#         if cl_filepath is not None:
#             path_all = get_all_perm_paths(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
        z_all = torch.from_numpy(z_all) # z_support & z_query
        
        # here should be acc
        # TODO: 5/7 
        if cl_filepath is None: # no need to draw tasks???
            acc = get_result(
                model=model, z_all=z_all, 
                n_way=n_way, n_support=n_support, n_query=n_query, metric='acc')
        else:
            # make model can parse feature correctly
            model.n_support = n_support
            model.n_query = n_query
            # get prediction
            pred_prob = get_pred(model, z_all = z_all, prob = True)
            record_acc_and_task_pred(pred_prob = pred_prob, task_data = task_data)
            
            return task_data
    
    
    
    else: # n_test_candidates setting
        assert params.n_test_candidates == len(cl_feature_each_candidate), "features & params mismatch."
        
        ################### choose those subnets to ensemble ###################
        
        ensemble_strategy = params.ensemble_strategy # 'avg_prob', 'bagging', 'adaboost', 'vote'
        
        is_single_exp = not isinstance(params.frac_ensemble, list)
        if is_single_exp:
            n_ensemble = 1 if params.frac_ensemble == None else int(params.frac_ensemble*params.n_test_candidates)
#         else:
#             n_frac_exps = len(params.frac_ensemble)
#             n_ensemble_ls = [frac*params.n_test_candidates for frac in params.frac_ensemble]
        
        if ensemble_strategy == 'adaboost':
            if not is_single_exp:
                raise ValueError('adaboost not support multi-frac_ensemble')
            ##### initialize data for adaboost #####
            elected_ids = []
            epsilons = np.zeros(n_ensemble)
            factors = np.zeros(n_ensemble)
            alphas = np.zeros(n_ensemble)
            
            candidate_sample_results = np.zeros((params.n_test_candidates, n_way, n_support))
            
            for cand_id in range(params.n_test_candidates): # for each candidate
                cl_feature_dict = cl_feature_each_candidate[cand_id] # features of the candidate

                z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
                z_all = torch.from_numpy(z_all) # z_support & z_query

                # reset back
                model.n_support = n_support
                model.n_query = n_query
                z_support, z_query  = model.parse_feature(z_all,is_feature=True)
                # z_support.shape:(n_way, n_support, *feature_dims)
                # z_query.shape:(n_way, n_query, *feature_dims)
                z_support   = z_support.contiguous()

                if model.change_way:
                    model.n_way  = z_support.size(0)
                
                split_sub_query = True
                
                if split_sub_query:
                    sample_results, sample_probs = get_result_loocv(
                        model=model, z_all=z_support, 
                        n_way=n_way, the_one='loopccv_one_val', 
                        metric=params.candidate_metric, return_all_probs=True) # to get sample_results
                    
                else: # use whole support set as sub_support set & sub_query set (acc too high to adaboost)
                    z_support_doubled = z_support.repeat(1,2,1) # copy alone n_shot dimension
                    n_sub_support = n_support # 1 | n_support-1 | n_support//2, 1 seems better?
                    n_sub_query = n_support # those who are rest
                    sub_result, sample_results = get_result(
                        model=model, z_all=z_support_doubled, 
                        n_way=n_way, n_support=n_sub_support, n_query=n_sub_query, 
                        metric=params.candidate_metric, return_prob=False)
                
                # sample_results.shape = (n_way*n_sub_query)(=(n_way*n_support))
                candidate_sample_results[cand_id] = sample_results.reshape(n_way,n_support)
                
            data_weights = np.ones((n_way, n_support)) * 1/(n_way*n_support) # initialize as 1/N_data
            if params.candidate_metric == 'acc': # should be acc
                # originally 100 or 0, transfer to 0/1 error
                candidate_sample_errors = 1 - candidate_sample_results/candidate_sample_results.max()
            else:
                raise ValueError('Unsupported candidate_metric for adaboost ensemble.')
            
            for t in range(n_ensemble):
                # compute new weighted errors (broadcast!!!
                candidate_weighted_sample_errors = candidate_sample_errors * data_weights / data_weights.sum()
                candidate_weighted_error = candidate_weighted_sample_errors.sum(axis=(1,2))
                elected_id = candidate_weighted_error.argmin(axis=0)
                elected_ids.append(elected_id)
                elected_sample_errors = candidate_sample_errors[elected_id]
                epsilons[t] = candidate_weighted_error[elected_id] # weighted error rate
                # TODO: should i ???
                eps = 1e-9
                factors[t] = np.sqrt((1-epsilons[t]+eps)/(epsilons[t]+eps))
                alphas[t] = np.log(factors[t])
                if epsilons[t] == 0:
#                     print('get perfect classification (that\'s bad!?)!!! break loop %d.'%(t))
                    break
                if 1-epsilons[t] <= 1/n_way:
                    print('get terrible classification worse than random guess !!! break loop %d.'%(t))
                    break

                multiply_weights = np.zeros_like(elected_sample_errors)
                multiply_weights[elected_sample_errors==1] = factors[t] # error
                multiply_weights[elected_sample_errors==0] = 1/factors[t] # correct
                data_weights = data_weights*multiply_weights

            ################### do the ensemble ###################
            alphas /= alphas.sum()

            # reset back
            model.n_support = n_support
            model.n_query = n_query
            # repeat procedure of common setting to get query prediction
            ensemble_probs = np.zeros((n_way*n_query, n_way)) # shape
            for t, elected_id in enumerate(elected_ids):
#                 elected_id = elected_ids[t]
                cl_feature_dict = cl_feature_each_candidate[elected_id]
                z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
                z_all = torch.from_numpy(z_all) # z_support & z_query
                prob = get_pred(model, z_all, prob=True)
                ensemble_probs += alphas[t] * prob
                
        
        elif ensemble_strategy in ['avg_prob', 'vote', 'ada_weight', 'bagging', 'bootstrap_avgrank']:
            
            if ensemble_strategy in ['bagging', 'bootstrap_avgrank']:
                
                ##### initialize indices and recorded results for bagging #####
                if ensemble_strategy == 'bagging':
                    if not is_single_exp:
                        raise ValueError('bagging not support multi-frac_ensemble')
                    n_resample = n_ensemble
                elif ensemble_strategy == 'bootstrap_avgrank':
                    n_resample = params.boot_n_resample if params.boot_n_resample is not None else 100

                candidate_resample_results = [] # recorded result, size: (n_cands, n_resample)
                n_sub_support_each_resampling = []
                resampled_ids_each_resample = []
                for _ in range(n_resample):
                    # 1 ~ n_support-1 or fixed as n_support-1???
                    if params.bag_n_sub_support is None:
                        n_sub_support = n_support-1 #np.random.choice(n_support-1) + 1 
                    elif params.bag_n_sub_support == 0:
                        n_sub_support = 1 + np.random.choice(n_support-1)
                    else:
                        n_sub_support = params.bag_n_sub_support
                    n_sub_query = n_support - n_sub_support
                    n_sub_support_each_resampling.append(n_sub_support)
                    resampled_id_each_way = []
                    for way in range(n_way):
                        resampled_id = np.random.choice(n_support, n_support, replace=False)
                        resampled_sub_support_id = np.random.choice(
                            resampled_id[:n_sub_support], n_sub_support, replace=True)
                        resampled_sub_query_id = np.random.choice(
                            resampled_id[n_sub_support:], n_sub_query, replace=True)
                        resampled_id[:n_sub_support] = resampled_sub_support_id
                        resampled_id[n_sub_support:] = resampled_sub_query_id
                        resampled_id_each_way.append(resampled_id)
                    resampled_ids_each_resample.append(resampled_id_each_way)
            
            if params.frac_ensemble == 1 and ensemble_strategy in ['avg_prob', 'vote', 'ada_weight']:
                # just use all the candidates, so no need to compute sub-performance
                elected_ids = np.array(range(params.n_test_candidates))
            else: 
                ##### initialize n_ensemble_ls in order
                if not is_single_exp:
                    n_frac_exps = len(params.frac_ensemble)
                    n_ensemble_ls = []
                    for frac in params.frac_ensemble:
                        if frac is None:
                            n_ensemble_ls.append(1)
                        else:
                            n_ensemble_ls.append(int(frac*params.n_test_candidates))
#                     params.frac_ensemble.sort()
                    n_ensemble_ls.sort()
#                     n_ensemble_ls = [frac*params.n_test_candidates for frac in params.frac_ensemble]
                    
                # validate sub-performance to choose from candidates
                # here seems took most of the time cost

                if ensemble_strategy in ['avg_prob', 'vote', 'ada_weight']:
                    sub_performance_each_candidate = [] # store sub_query set result of each candidate

                candidate_sample_results = np.zeros((params.n_test_candidates, n_way, n_support))
                # the following 2 lists seems useless currently
                supp_prob_each_candidate = [] # 7/7 store each support data (as sub-query) prediction of each candidate
                query_prob_each_candidate = [] # 7/7 store query set prediction of each candidate

                for cand_id in range(params.n_test_candidates): # for each candidate
                    cl_feature_dict = cl_feature_each_candidate[cand_id] # features of the candidate

                    z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
                    z_all = torch.from_numpy(z_all) # z_support & z_query

                    # reset back
                    model.n_support = n_support
                    model.n_query = n_query
                    z_support, z_query = model.parse_feature(z_all,is_feature=True)
                    # z_support.shape:(n_way, n_support, *feature_dims)
                    # z_query.shape:(n_way, n_query, *feature_dims)
                    z_support   = z_support.contiguous()

                    if model.change_way:
                        model.n_way  = z_support.size(0)
                    # TODO: tunable n_sub_support
                    # leave-one-out (per-class) cross validation

                    if ensemble_strategy in ['avg_prob', 'vote', 'ada_weight']:
                        val_mode = 'loopccv_one_val' # None, 'train', 'val'
                        if 'loopccv_one' in val_mode:
                            supp_cv_sample_results, supp_prob = get_result_loocv(
                                model=model, z_all=z_support, 
                                n_way=n_way, the_one=val_mode, 
                                metric=params.candidate_metric, return_all_probs=True)

                            candidate_sample_results[cand_id] = supp_cv_sample_results
                            sub_result = supp_cv_sample_results.sum() / (n_way*n_support)
                            supp_prob_each_candidate.append(supp_prob) # seems useless currently??
                        else: # testing without loopccv
                            n_sub_support = 1 # 1 | n_support-1 | n_support//2, 1 seems better?
                            n_sub_query = n_support - n_sub_support # those who are rest
                            sub_result, sample_results = get_result(
                                model=model, z_all=z_support, 
                                n_way=n_way, n_support=n_sub_support, n_query=n_sub_query, 
                                metric=params.candidate_metric)

                        sub_performance_each_candidate.append(sub_result)

                    elif ensemble_strategy in ['bagging', 'bootstrap_avgrank']:
                        ##### get results of each resampling #####
                        # Here seems consume most of the time
                        result_each_resample = []
                        
                        for resample_num in range(n_resample):
                            
                            n_sub_support = n_sub_support_each_resampling[resample_num]
                            n_sub_query = n_support - n_sub_support # TODO: maybe can use all (n_support)???

#                             z_support_resampled = z_support.data.cpu().numpy() # shape: (n_way, n_support, *n_dim) # do not transfer between torch Tensor and numpy array
                            z_support_resampled = z_support.detach().clone()
                            resampled_id_each_way = resampled_ids_each_resample[resample_num]
                            # here consume more time
                            for way in range(n_way): 
                                # sample sub_support set (with replacement) for each class
                                # other samples should be query data in each class
                                resampled_id = resampled_id_each_way[way]
                                way_z_resampled = z_support_resampled[way][resampled_id]
                                z_support_resampled[way] = way_z_resampled
#                                 z_support_resampled.scatter(way, index = torch.LongTensor([]).cuda(), src = way_z_resampled)

                            result, sample_results = get_result(
                                model=model, z_all=z_support_resampled, 
                                n_way=n_way, n_support=n_sub_support, n_query=n_sub_query, 
                                metric=params.candidate_metric, return_prob=False)

                            result_each_resample.append(result)

                        candidate_resample_results.append(result_each_resample)
                
                ##### get ensemble ids #####
                if params.ensemble_strategy in ['avg_prob', 'vote', 'ada_weight']:
                    sub_performance_each_candidate = np.array(sub_performance_each_candidate)
                    if params.candidate_metric == 'acc':
                        sorted_candidate_ids = np.argsort(-sub_performance_each_candidate) # in descent order
#                         if is_single_exp:
                        elected_ids = sorted_candidate_ids[:n_ensemble]
#                         else:
#                             exp_elected_ids = [0]*n_frac_exps
                        
                    elif params.candidate_metric == 'loss':
                        sorted_candidate_ids = np.argsort(sub_performance_each_candidate) # in ascent order
                        elected_ids = sorted_candidate_ids[:n_ensemble]
                    else:
                        raise ValueError('Unknown candidate_metric: %s'%(metric))
                    
                elif params.ensemble_strategy in ['bagging', 'bootstrap_avgrank']:
                    candidate_resample_results = np.asarray(candidate_resample_results) # shape: (n_test_candidates, n_ensemble)
                    if params.ensemble_strategy == 'bagging':
                        if params.candidate_metric == 'loss':
                            elected_ids = candidate_resample_results.argmin(axis=0)
                        elif params.candidate_metric == 'acc':
                            elected_ids = candidate_resample_results.argmax(axis=0)
                    elif params.ensemble_strategy == 'bootstrap_avgrank':
                        # trick to get order: argsort().argsort()
                        if params.candidate_metric == 'loss':
                            resamples_candidate_orders = candidate_resample_results.argsort(0).argsort(0)
                        elif params.candidate_metric == 'acc':
                            resamples_candidate_orders = (-candidate_resample_results).argsort(0).argsort(0)
                        candidate_mean_orders = resamples_candidate_orders.mean(axis=1) # ranking, the smaller the better
                        if is_single_exp:
                            elected_ids = candidate_mean_orders.argsort(axis=0)[:n_ensemble]
                        else:
                            sorted_candidate_ids = candidate_mean_orders.argsort(axis=0)
#                             elected_ids = sorted_candidate_ids
                        
            ################### do the ensemble ###################
            all_preds = []

            # reset back
            model.n_support = n_support
            model.n_query = n_query
            # repeat procedure of common setting to get query prediction
            selected_ids = elected_ids if is_single_exp else sorted_candidate_ids[:np.max(n_ensemble_ls)]
            for elected_id in selected_ids:
                cl_feature_dict = cl_feature_each_candidate[elected_id]
                z_all = get_all_perm_features(select_class=select_class, cl_feature_dict=cl_feature_dict, perm_ids_dict=perm_ids_dict)
                z_all = torch.from_numpy(z_all) # z_support & z_query

                if params.ensemble_strategy in ['avg_prob', 'ada_weight', 'bagging', 'bootstrap_avgrank']:
                    pred = get_pred(model, z_all, prob=True)
                elif params.ensemble_strategy=='vote':
                    raise ValueError('stop using ensemble_strategy: vote.')
                    pred = get_pred(model, z_all)
#                 elif params.ensemble_strategy=='bagging':
#                     pred = get_pred(model, z_all, prob=True)
                else:
                    raise ValueError('Invalid ensemble_strategy: %s'%(params.ensemble_strategy))
                all_preds.append(pred)
            

            # all_preds shape=(n_ensemble, n_query*n_way) for 'vote'
            # all_preds shape=(n_ensemble, n_query*n_way, n_way) for 'avg_prob'
            all_preds = np.array(all_preds)

            if params.ensemble_strategy=='vote':
                raise ValueError('stop using ensemble_strategy: vote.')
                all_preds = all_preds.T # shape:(n_query*n_way, n_ensemble) for 'vote'
                ensemble_preds = [np.argmax(np.bincount(preds)) for preds in all_preds]
                ensemble_preds = np.array(ensemble_preds)
            elif params.ensemble_strategy in ['avg_prob', 'bagging', 'bootstrap_avgrank']:
                if is_single_exp:
                    ensemble_probs = all_preds.mean(axis=0) # shape=(n_query*n_way, n_way)
                    ensemble_preds = np.argmax(ensemble_probs, axis=1) # shape=(n_query*n_way)
                else:
                    # TODO: 
                    exp_ensemble_probs = [None]*n_frac_exps
                    for frac_exp_id in range(n_frac_exps):
                        n_ensemble = n_ensemble_ls[frac_exp_id]
#                         try:
                        ensemble_probs = all_preds[:n_ensemble].mean(axis=0) # shape=(n_query*n_way, n_way)
#                         except TypeError:
#                             print('n_ensemble:', n_ensemble)
                        ensemble_preds = np.argmax(ensemble_probs, axis=1) # shape=(n_query*n_way) # useless???
                        exp_ensemble_probs[frac_exp_id] = ensemble_probs
                        
            elif params.ensemble_strategy == 'ada_weight':
                elected_sub_performances = sub_performance_each_candidate[elected_ids]
                if params.candidate_metric == 'acc':
                    # compute error rate
                    elected_sub_errors = (100 - elected_sub_performances)/100
                    eps = 1e-9
                    factors = (1-elected_sub_errors+eps)/(elected_sub_errors+eps)
                    alphas = np.log(factors)
                    alphas /= alphas.sum()
                    ensemble_probs = np.average(all_preds, weights=alphas, axis=0) # shape=(n_query*n_way, n_way)
                    ensemble_preds = np.argmax(ensemble_probs, axis=1) # shape=(n_query*n_way)
                else:
                    raise ValueError('Unsupported candidate_metric for ada_weight.')
            
            
        if params.ensemble_strategy == 'bootstrap_avgrank' and not is_single_exp:
            frac_task_data = []
            for frac_exp_id in range(n_frac_exps):
                tmp_task_data = copy.deepcopy(task_data)
                ensemble_probs = exp_ensemble_probs[frac_exp_id]
                record_acc_and_task_pred(pred_prob = ensemble_probs, task_data = tmp_task_data)
                frac_task_data.append(tmp_task_data)
            return frac_task_data
        else:
            record_acc_and_task_pred(pred_prob = ensemble_probs, task_data = task_data)
            return task_data

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



