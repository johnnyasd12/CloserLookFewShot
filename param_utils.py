import numpy as np
from argparse import Namespace
import itertools
import logging

def get_all_params_comb(possible_params: dict):
    '''
    Args:
        possible_params (dict): dictionary of all possible params, e.g. {'dropout_p':[0.25, 0.5], 'frac_ensemble':[0.1, 0.3]}

    Return:
        params_ls (List): list of all combination dictionary
    '''
    tune_names = [] # e.g. ['dropout_p', 'frac_ensemble']
    tune_lsls = [] # e.g. [[0.25, 0.5, 0.75], [0.1, 0.3, 0.5]]
    for key, ls in possible_params.items():
        tune_names.append(key)
        tune_lsls.append(ls)
    params_comb = list(itertools.product(*tune_lsls)) # e.g. [[0.25,0.1], [0.25,0.3], [0.25,0.5], ..., [0.75,0.5]]
    n_combs = len(params_comb)
    params_ls = [] # e.g. [{'dropout_p':0.25, 'frac_ensemble':0.1}, ..., {'dropout_p':0.75, 'frac_ensemble':0.5}]
    for i in range(n_combs):
        d = {}
        for j, key in enumerate(tune_names):
            d[key] = params_comb[i][j]
        params_ls.append(d)
    return params_ls

def get_args(params: dict):
    args = Namespace()
    for k,v in params.items():
        setattr(args, k, v)
    return args

def copy_args(args):
    copied_args = Namespace(**vars(args))
    return copied_args

def get_modified_args(base_args, extra_params: dict):
    new_args = Namespace(**vars(base_args)) # copy base arguments
    for k,v in extra_params.items():
        setattr(new_args, k, v)
    return new_args

def get_all_args_ls(base_args, possible_params: dict):
    params_comb_ls = get_all_params_comb(possible_params)
    all_args = []
    for params in params_comb_ls:
        args = get_modified_args(base_args, params)
        all_args.append(args)
    return all_args

def get_matched_df(params, df, possible_params={}):
    '''
    Args:
        possible_params (dict): dictionary of list
    '''
    base_cond = None
    for k,v in params.items():
        if k in list(df.columns):
            cond = df[k]==v if (v!=None and v==v) else df[k]!=df[k]
            base_cond = base_cond&cond if base_cond is not None else cond
#             logging.debug('get_matched_df()/ %s==%s:\n%s'%(k,v,cond))
        else:
            logging.warning('param_utils/get_matched_df/"%s" not in df.columns'%(k))
    
    df = df[base_cond]
    
#     logging.debug('get_matched_df()/df after params:\n%s'%(df))
    
#     logging.debug('possible_params: %s'%(possible_params))
    if len(possible_params)!=0:
        for k, possible_values in possible_params.items():
            if k in list(df.columns):
                k_cond = None
                for value in possible_values:
                    cond = df[k]==value if value != None else df[k]!=df[k]
    #                 logging.debug('cond: %s'%(cond))
                    k_cond = k_cond|cond if k_cond is not None else cond
    #             logging.debug('k_cond: %s'%(k_cond))
                df = df[k_cond]
            else:
                logging.warning('"%s" not in df.columns'%(k))
            
    return df
