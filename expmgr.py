import numpy as np
from io_utils import parse_args, args_sanity_check
from train import exp_train_val
from save_features import exp_save_features
from test import exp_test, record_to_csv
from param_utils import get_all_params_comb, get_all_args_ls, get_modified_args, copy_args, get_matched_df
import pandas as pd
import os

# to empty cache
import torch

# for better error message when encounter RuntimeError: CUDA error: device-side assert triggered
if False:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from my_utils import del_keys
import logging

import pickle

# to avoid UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure
# only for jupyter notebook?
# import matplotlib
# matplotlib.pyplot.ion()
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt # to draw in notebook? (refer to draw_utils.py)

# to draw gray images
import matplotlib.cm as cm
# from matplotlib import cm

# to draw task, and frac_ensemble multi-exps
import copy

class ExpManager:
    def __init__(self, base_params, train_fixed_params, test_fixed_params, general_possible_params, test_possible_params, pkl_postfix, record_folder):
        '''
        Args:
            base_params (dict): the core settings of the experiment
            train_fixed_params (dict): e.g. {'stop_epoch':700, 'gpu_id':0}
            test_fixed_params (dict): e.g. {'record_csv':'program_time.csv', 'gpu_id':0}
            general_possible_params (dict): dictionary of list of tunable parameters, different param would train different model. e.g. {'dropout_p':[0.25, 0.5]}
        '''
        self.base_params = base_params # comparison is based on this setting.
        self.possible_params = {'general':general_possible_params, 'test':test_possible_params} # general means generalize to train/save_features/test
        self.fixed_params = {'train':train_fixed_params, 'test':test_fixed_params}
        self.negligible_vars = ['gpu_id', 'csv_name'] # can be ignored when comparing results but in ArgParser
        self.negligible_vars += ['split', 'num_classes'] # not sure if should be ignored
        self.negligible_vars += ['debug'] # hack: ignore debug param
        self.dependent_vars = [
            'epoch', 'train_acc_mean', 'source_val_acc', 'val_acc_mean', 'val_acc_std', 'novel_acc_mean', 'novel_acc_std']
        
        self.results_pkl = [] # params as well as results restore in the list of dictionaries
        
        self.record_folder = record_folder
        print('record_folder:', record_folder)
        if not os.path.exists(record_folder):
            print('record_folder not exists, creating...')
            os.mkdir(record_folder)
        
        self.csv_path = os.path.join(self.record_folder, test_fixed_params['csv_name'])
        self.pkl_postfix = pkl_postfix
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
    
    def exp_grid(self, choose_by='val_acc_mean', mode='from_scratch'):
        '''
        Args:
            mode (str): 'from_scratch'|'resume'|'draw_tasks'|'tmp_pkl' # 'draw_tasks' could be deprecated???
        '''
        print('exp_grid() start.')
        print('self.base_params:', self.base_params)
        base_params = self.base_params.copy() # set to test_n_way = default(5)
        if mode == 'tmp_pkl':
            if 'test_n_way' in base_params:
                print('change base_params["test_n_way"] to be 5.')
                base_params['test_n_way'] = 5
        default_args = {} # the raw default args of the code
        default_args['train'] = parse_args('train', parse_str='')
        default_args['test'] = parse_args('test', parse_str='')
        possible_params = self.possible_params
        
        if mode == 'tmp_pkl':
            for key in possible_params['general']:
                if len(possible_params['general'][key]) != 1:
                    raise ValueError('general param length should be 1 for mode: tmp_pkl')
            for key in possible_params['test']:
                if len(possible_params['test'][key]) != 1:
                    raise ValueError('test param length should be 1 for mode: tmp_pkl')
        
        train_args = get_modified_args(default_args['train'], {**base_params, **self.fixed_params['train']})
        test_args = get_modified_args(default_args['test'], {**self.base_params, **self.fixed_params['test']}) # test_n_way should be 2
        all_general_possible_params = get_all_params_comb(possible_params=possible_params['general'])
        all_test_possible_params = get_all_params_comb(possible_params=possible_params['test'])
        
        csv_path = os.path.join(self.record_folder, self.fixed_params['test']['csv_name'])
        
        # TODO: refactor to many functions with argument: mode
        
        is_csv_exists = os.path.exists(csv_path)
        
        ########## deal with initial csv ##########
        if mode == 'from_scratch':
            if not is_csv_exists:
                # create empty csv
                print("Creating file: '%s' since not exist"%(csv_path))
                with open(csv_path, "w") as file:
                    empty_df = pd.DataFrame(columns=['dataset'])
                    empty_df.to_csv(file, header=True, index=False)

        pkl_postfix_str = '_' + self.pkl_postfix #+ '.pkl'
        pkl_postfix_str += '.pkl'

        pkl_path = csv_path.replace('.csv', pkl_postfix_str) # restore ALL experiments in this class
        is_pkl_exists = os.path.exists(pkl_path)
        
        ########## load pickle data ##########
        if mode == 'resume':
            print('loading self.results_pkl from:', pkl_path)
            if is_pkl_exists:
                with open(pkl_path, 'rb') as handle:
                    self.results_pkl = pickle.load(handle)
            else:
                logging.warning('No previous result pickle file: %s, only record self.results_pkl from scratch.'%(pkl_path))
            print('self.results_pkl at begin, len =', len(self.results_pkl))
        elif mode == 'tmp_pkl':
            if is_pkl_exists:
                print('pickle file: %s, already exists.'%(pkl_path))
                return
            
            
        for general_params in all_general_possible_params:
            
            ########## decide if should_train ##########
            if mode in ['resume', 'from_scratch']:
#             if mode == 'resume':# or mode == 'draw_tasks':
                print()
                print('='*20, 'Checking if already trained in:', csv_path, '='*20)
                print('base_params:', base_params)
                print('general_params:', general_params)
                # read csv
                loaded_df = pd.read_csv(csv_path)

                # check if ALL test_params has been experimented. if so, then do NOT even train model
                check_df = loaded_df.copy()
                check_param = {**base_params, **general_params}
                
                # include default columns if param not in dataframe
                default_train_args = parse_args('train', parse_str='')
                default_train_param = vars(default_train_args)
                for param in check_param:
                    if param not in check_df.columns:
                        print('param:', param)
                        if default_train_param[param] is None:
                            check_df[param] = np.nan
                        else:
                            check_df[param] = default_train_param[param]
                
                check_df = get_matched_df(check_param, check_df)
                num_experiments = len(check_df)
                
                should_train = num_experiments == 0
                
#             elif mode == 'from_scratch':
#                 should_train = True
            elif mode in ['draw_tasks', 'tmp_pkl']:
                should_train = False
            
            ########## training ##########
            if should_train:
                # release cuda memory
                torch.cuda.empty_cache()
                # train model
                print()
                print('='*20, 'Training', '='*20)
                print('general_params:', general_params)
                modified_train_args = get_modified_args(train_args, general_params)
                if mode == 'resume':
                    modified_train_args.resume = True
                
                args_sanity_check(modified_train_args, script='train')
                
                train_result = exp_train_val(modified_train_args)
            else:
                if mode == 'draw_tasks':
                    print('NO need to train when draw_tasks.')
                elif mode == 'resume':
                    print('NO need to train since already trained in record:', csv_path)
                    print('Summing up current record...')
                    choose_by = 'val_acc_mean'
                    top_k = None
                    self.sum_up_results(choose_by=choose_by, top_k=top_k)
                else:
                    print('NO need to train since already trained in record:', csv_path)
            
            modified_test_args = get_modified_args(test_args, general_params)
            
            # loop over testing settings under each general setting
            for test_params in all_test_possible_params:
                
                ########## check if should test ##########
                if mode == 'resume':
                    print('\n', '='*20, 'Checking if already did experiments', '='*20)
                    print('general_params:', general_params)
                    print('test_params:', test_params)
                    check_df = loaded_df.copy()
                    constrained_param = {**self.base_params, **general_params, **test_params}
                    
                    # add default params, delete useless params & dependent variables (result)
                    default_args = parse_args('test', parse_str='')
                    default_param = vars(default_args)
                    check_param = {**default_param, **constrained_param}
                    
                    # delete vars not for checking
                    del_vars = self.negligible_vars + self.dependent_vars
                    for var in del_vars:
                        if var in check_param:
                            del check_param[var]

                    # include default columns if param not in dataframe
                    for param in check_param:
                        if param not in check_df.columns:
                            print('param:', param)
                            if default_param[param] is None:
                                check_df[param] = np.nan
                            else:
                                check_df[param] = default_param[param]
#                             print(check_df)
#                             yooooo
                    check_df = get_matched_df(check_param, check_df) # should loaded_df be check_df ???
                    num_test_experiments = len(check_df)
                    if num_test_experiments>0: # already experiments
                        print('NO need to test since already tested %s times in record: %s'%(num_test_experiments, csv_path))
                        print(check_df)
                        continue
                    else:
                        print('Need to do testing since there\'s NO experiment in data.')
                elif mode == 'tmp_pkl':
                    pass
                
                final_test_args = get_modified_args(modified_test_args, test_params)
                
                write_record = {**general_params, **test_params}
                
                ########## write train_acc to record dict ##########
                if mode in ['resume', 'from_scratch']:
#                 if mode == 'resume': #in ['resume', 'draw_tasks']:
                    source_val = True
                    
                    # write source_val_acc
                    if source_val and 'cross' in base_params['dataset'] and base_params['dataset'] != 'cross':# in ['cross_base20cl', 'cross_char', 'cross_char_half', 'cross_char_quarter', 'cross_char2', 'cross_char2_base1lang']:
                        if should_train:
                            write_record['source_val_acc'] = train_result['source_val_acc']
                        else:
                            # to get source_val_acc, so no need test_params
                            check_df = loaded_df.copy()
                            check_df = get_matched_df({**self.base_params, **general_params}, check_df)
                            write_record['source_val_acc'] = check_df['source_val_acc'].iloc[0]
                    else:
                        write_record['source_val_acc'] = None
                    
                    # write train_acc
                    if should_train:
                        write_record['train_acc_mean'] = train_result['train_acc']
                    else:
                        # to get train_acc, so no need test_params
                        check_df = loaded_df.copy()
                        check_df = get_matched_df({**self.base_params, **general_params}, check_df)
                        write_record['train_acc_mean'] = check_df['train_acc_mean'].iloc[0]
                
                ########## save_features & test ##########
                if mode in ['from_scratch', 'resume', 'tmp_pkl']:
                
                    ########### judge if should do several exp for frac_ensemble ###########
                    tmp = final_test_args # to make variable name shorter
                    is_str_single_frac = isinstance(tmp.frac_ensemble, str) and ',' not in tmp.frac_ensemble
                    is_float_none_frac = isinstance(tmp.frac_ensemble, float) or tmp.frac_ensemble==1 or tmp.frac_ensemble is None
                    is_single_exp = is_float_none_frac or is_str_single_frac
                    if is_str_single_frac:
                        final_test_args.frac_ensemble = frac_ensemble_str2var(params.frac_ensemble)

                    if is_single_exp: # common frac_ensemble
                    
                        splits = ['val', 'novel'] # temporary no 'train'
                        for split in splits: # val, novel

                            ##### get args #####
                            split_final_test_args = copy_args(final_test_args)
                            split_final_test_args.split = split
                            print('\n', '='*20, 'Saving Features', '='*20)
                            print('general_params:', general_params)
                            print('test_params:', test_params)
                            print('data split:', split)

                            args_sanity_check(split_final_test_args, script='save_features')
                            args_sanity_check(split_final_test_args, script='test')
                            
                            ########## save features ##########
                            exp_save_features(copy_args(split_final_test_args))

                            print('\n', '='*20, 'Testing', '='*20)
                            print('general_params:', general_params)
                            print('test_params:', test_params)
                            print('data split:', split)
                            n_episodes = 10 if split_final_test_args.debug or mode=='draw_tasks' else 600

                            ########## testing and record to dict ##########
                            exp_record, task_datas = exp_test(
                                copy_args(split_final_test_args), n_episodes=n_episodes, should_del_features=True)#, show_data=show_data)
                            write_record['epoch'] = exp_record['epoch']
                            write_record[split+'_acc_mean'] = exp_record['acc_mean']
                            write_record[split+'_acc_std'] = exp_record['acc_std']

                            torch.cuda.empty_cache()
                            
                        ########## record to csv ##########
                        if mode in ['from_scratch', 'resume']:
                            print('Saving record to:', csv_path)
                            record_to_csv(final_test_args, write_record, csv_path=csv_path)

                            print('='*20, 'Current Experiments', '='*20)
                            choose_by = 'val_acc_mean'
                            top_k = None
                            self.sum_up_results(choose_by, top_k)

                        ########## record to pickle ##########
                        write_record['novel_task_datas'] = task_datas # currently ignore val_task_datas
                        self.results_pkl.append(write_record)
                        print('Saving self.results_pkl into:', pkl_path)
                        with open(pkl_path, 'wb') as handle:
                            pickle.dump(self.results_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                    else: ########## multiple frac_ensemble ##########
                        # make frac_ensemble list
                        frac_ls = final_test_args.frac_ensemble.split(',')
                        frac_ls = list(map(frac_ensemble_str2var, frac_ls))
                        print('frac_ls:', frac_ls)
                        
                        ########## check if each frac_ensemble already done in exps ##########
                        if mode == 'resume':
                            for frac in frac_ls.copy():
                                print('\n', '='*20, 'Checking if already did "frac_ensemble" experiments', '='*20)
                                print('general_params:', general_params)
                                print('test_params:', test_params)
                                check_df = loaded_df.copy()
                                constrained_param = {**self.base_params, **general_params, **test_params}
                                # only add this line different from before
                                constrained_param['frac_ensemble'] = frac

                                # add default params, delete useless params & dependent variables (result)
                                default_args = parse_args('test', parse_str='')
                                default_param = vars(default_args)
                                check_param = {**default_param, **constrained_param}

                                # delete vars not for checking
                                del_vars = self.negligible_vars + self.dependent_vars
                                for var in del_vars:
                                    if var in check_param:
                                        del check_param[var]

                                check_df = get_matched_df(check_param, loaded_df)
                                num_test_experiments = len(check_df)
                                if num_test_experiments>0: # already experiments
                                    print('NO need to test since already tested %s times in record: %s'%(num_test_experiments, csv_path))
                                    print(check_df)
                                    frac_ls.remove(frac)
                                else:
                                    print('Need to do testing for frac_ensemble:', frac,'since there\'s NO experiment in data.')
                        
                        ########## do exps for multi-frac_ensemble ##########
                        if len(frac_ls) > 0:
                            final_test_args.frac_ensemble = frac_ls
                            
                            ##### initialize record for different frac #####
                            frac_write_records = [copy.deepcopy(write_record) for _ in range(len(frac_ls))]
                            
                            ##### start exps #####
                            splits = ['val', 'novel']
                            for split in splits: # val, novel
                                
                                ##### get args #####
                                split_final_test_args = copy_args(final_test_args)
                                split_final_test_args.split = split
                                print('\n', '='*20, 'Saving Features', '='*20)
                                print('general_params:', general_params)
                                print('test_params:', test_params)
                                print('data split:', split)
                                
                                ##### sanity check #####
                                args_sanity_check(split_final_test_args, script='save_features')
                                args_sanity_check(split_final_test_args, script='test')
                                
                                ########## save features ##########
                                exp_save_features(copy_args(split_final_test_args))

                                print('\n', '='*20, 'Testing', '='*20)
                                print('general_params:', general_params)
                                print('test_params:', test_params)
                                print('data split:', split)
                                n_episodes = 10 if split_final_test_args.debug or mode=='draw_tasks' else 600
                                
                                ########## testing and record to dict ##########
                                
                                ##### return n_frac exps results #####
                                frac_exp_records, frac_task_datas = exp_test(
                                    copy_args(split_final_test_args), n_episodes=n_episodes, should_del_features=True)
                                
                                for frac_id, frac in enumerate(frac_ls):
                                    frac_write_record = frac_write_records[frac_id]
                                    exp_record = frac_exp_records[frac_id]
                                    frac_write_record['frac_ensemble'] = frac
                                    frac_write_record['epoch'] = exp_record['epoch']
                                    frac_write_record[split+'_acc_mean'] = exp_record['acc_mean']
                                    frac_write_record[split+'_acc_std'] = exp_record['acc_std']

                                torch.cuda.empty_cache()

                            ########## record n_frac exps to csv ##########
                            if mode in ['from_scratch', 'resume']:
                                print('Saving record to:', csv_path)
                                for frac_id, frac in enumerate(frac_ls):
                                    frac_write_record = frac_write_records[frac_id]
                                    tmp_test_args = copy_args(final_test_args)
                                    tmp_test_args.frac_ensemble = frac
                                    record_to_csv(tmp_test_args, frac_write_record, csv_path=csv_path)

                                print('='*20, 'Current Experiments', '='*20)
                                choose_by = 'val_acc_mean'
                                top_k = None
                                self.sum_up_results(choose_by, top_k)

                            ########## record n_frac exps to pickle ##########
                            for frac_id, frac in enumerate(frac_ls):
                                frac_write_record = frac_write_records[frac_id]
                                frac_write_record['novel_task_datas'] = frac_task_datas[frac_id] # currently ignore val_task_datas
                                self.results_pkl.append(frac_write_record)
                            print('Saving self.results_pkl into:', pkl_path)
                            with open(pkl_path, 'wb') as handle:
                                pickle.dump(self.results_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                
                        else:
                            print('='*20, 'all frac_ensemble exps had been done before.', '='*20)

                torch.cuda.empty_cache()
        
        ########## sum up results ##########
        # TODO: can also loop dataset
        if mode in ['from_scratch', 'resume', 'draw_tasks']:
            for choose_by in ['val_acc_mean', 'novel_acc_mean']:
                # read csv to compare results
                top_k = None
                self.sum_up_results(choose_by, top_k)
        
        if mode == 'draw_tasks':
            print('loading self.results_pkl from:', pkl_path)
            best_model_all_tasks = get_best_all_tasks(pkl_path = pkl_path)
            
            print('sorting tasks...')
            sorted_tasks = sorted(best_model_all_tasks, key = lambda i: float(i['acc'])) # in ascending order
            print('Draw Worst Tasks...')
            save_img_folder = os.path.join(self.record_folder, 'imgs')
            draw_tasks(
                sorted_tasks, n_tasks = 3, 
                save_img_folder = save_img_folder, exp_postfix = self.pkl_postfix
            ) # utilize self.results_pkl, save all_tasks of best res
#             self.draw_tasks(best_model_all_tasks, n_tasks = 3) # utilize self.results_pkl, save all_tasks of best res

    def sum_up_results(self, choose_by, top_k, show_same_params=True): # choose the best according to dataset & split
        
        def select_cols_if_exists(df, cols: list):
            for col in cols.copy(): # BUGFIX: some of col not removed
                if col not in list(df.columns):
                    logging.warning('sum_up_results()/"%s" not in dataframe, deleted from cols.'%(col))
                    cols.remove(col)
            return df[cols]
        
        def del_all_the_same_cols(df):
            if len(df.index) != 1:
#                 df = df[[col for col in df if not df[col].nunique()==1]]
                df = df[[col for col in df if not len(set(df[col]))==1]] # when only 2 unique and one is nan then this would work
            else:
                pass
            return df
        
        def replace_std_with_conf(df):
            n_episodes = 600
            df2 = df.copy()
            df2['novel_acc_95%CI'] = 1.96*df2['novel_acc_std']/np.sqrt(n_episodes)
            del df2['novel_acc_std']
            return df2
        
        csv_path = os.path.join(self.record_folder, self.fixed_params['test']['csv_name'])
        print('sum_up_results/Reading file:', csv_path)
        record_df = pd.read_csv(csv_path)
        
        default_test_args = parse_args('test', parse_str='')
        default_test_params = default_test_args.__dict__ # to avoid including other exps
        important_fixed_params = {**default_test_params, **self.base_params, **self.fixed_params['test']}

        # delete negligible_vars & changeable vars
        del_keys(important_fixed_params, self.negligible_vars)
        del_keys(important_fixed_params, self.possible_params['general'].keys())
        del_keys(important_fixed_params, self.possible_params['test'].keys())
        
        # for multi-frac_ensemble
        test_possible_params = copy.deepcopy(self.possible_params['test'])
        if 'frac_ensemble' in test_possible_params.keys():
            frac_param = test_possible_params['frac_ensemble']
            if len(frac_param)==1 and isinstance(frac_param[0],str):
                if ',' in frac_param[0]:
                    frac_ls = frac_param[0].split(',')
                    frac_ls = list(map(frac_ensemble_str2var, frac_ls))
                    test_possible_params['frac_ensemble'] = frac_ls
        
        all_possible_params = {**self.possible_params['general'], **test_possible_params}
        
        matched_df = get_matched_df(important_fixed_params, record_df, possible_params=all_possible_params)
        
        ##### debug #####
#         print('matched_df:\n%s'%(matched_df))

        if top_k==None:
            top_k = len(matched_df)
        
        if len(matched_df)!=0:
            sorted_df = matched_df.sort_values(by=choose_by, ascending=False)
            compare_cols = list(self.possible_params['general'].keys())+list(self.possible_params['test'].keys())
#             compare_cols = compare_cols + ['epoch', 'train_acc_mean', 'val_acc_mean', 'novel_acc_mean']
            compare_cols = compare_cols + ['epoch', 'train_acc_mean', 'source_val_acc', 'val_acc_mean', 'novel_acc_mean','novel_acc_std']
            print()
            print('Best Test Acc: %s, selected by %s'%(sorted_df['novel_acc_mean'].iloc[0], choose_by))
            
            show_df = select_cols_if_exists(sorted_df, compare_cols)
            show_df = del_all_the_same_cols(show_df)
            if 'novel_acc_std' in show_df.columns:
#                 print('computing 95% confidence interval...')
                show_df = replace_std_with_conf(show_df)
            
            print()
            print('='*20,'Top %s/%s results sorted by: %s'%(top_k, len(matched_df), choose_by), '='*20)
            print(show_df.head(top_k))
#             print(sorted_df[compare_cols].head(top_k))
        else:
            print('='*20, 'No experiment matching the conditions', '='*20)
    
    def exp_random_search():
        pass
    
    def draw_hyper_relation(hyper, control_params):
        ''' draw multiple figures to see hyper-performance relation
        Args:
            hyper (str)
            fix_params (List): draw figures for each combinations of fix_params
        Draw:
            x-axis: hyper values
            y-axis: performance
        '''
        pass

def frac_ensemble_str2var(frac_ensemble):
    if frac_ensemble.lower() == 'none':
        return None
    else:
        return float(frac_ensemble)

def draw_both_worst_tasks(pkl_path1, pkl_path2, n_tasks, save_img_folder, exp1_postfix, exp2_postfix):
    best_exp1_all_task = get_best_all_tasks(pkl_path1)
    best_exp2_all_task = get_best_all_tasks(pkl_path2)
    
    all_tasks = []
    for i in range(len(best_exp1_all_task)):
        exp1_task = best_exp1_all_task[i]
        exp2_task = best_exp2_all_task[i]
        
        task = copy.deepcopy(exp1_task)
        
        for img_key in task:
            if 'qu' in img_key:
                del task[img_key]['pred']
                del task[img_key]['pred_prob']
                pred1 = exp1_task[img_key]['pred']
                pred2 = exp2_task[img_key]['pred']
                task[img_key]['exp1_pred'] = pred1
                task[img_key]['exp2_pred'] = pred2
        
        del task['acc']
        acc1 = exp1_task['acc']
        acc2 = exp2_task['acc']
        task['both_avg_acc'] = (acc1 + acc2) / 2
        
        all_tasks.append(task)
    
    acc_increase_tasks = sorted(all_tasks, key=lambda i: i['both_avg_acc'])
    exp_postfix = 'avg_' + exp1_postfix + '_' + exp2_postfix
    
    draw_tasks(acc_increase_tasks, n_tasks = n_tasks, 
               save_img_folder = save_img_folder, exp_postfix = exp_postfix, mode = 'both_exp_avg')
    

def draw_most_differ_tasks(pkl_path1, pkl_path2, n_tasks, save_img_folder, exp1_postfix, exp2_postfix):
    best_exp1_all_task = get_best_all_tasks(pkl_path1)
    best_exp2_all_task = get_best_all_tasks(pkl_path2)
    
    all_tasks_diff = []
    for i in range(len(best_exp1_all_task)):
        exp1_task = best_exp1_all_task[i]
        exp2_task = best_exp2_all_task[i]
        
#         diff_task = exp1_task.copy()
        diff_task = copy.deepcopy(exp1_task)
#         print('diff_task.keys():', diff_task.keys()) # acc, img_keys
        del diff_task['acc']
        acc1 = exp1_task['acc']
        acc2 = exp2_task['acc']
        diff_task['1-2_acc'] = acc1 - acc2
        diff_task['2-1_acc'] = acc2 - acc1
        diff_task['exp1_acc'] = acc1
        diff_task['exp2_acc'] = acc2
        
        for img_key in diff_task:
#             print('img_key:', img_key)
#             print('diff_task[img_key]:', diff_task[img_key])
            if isinstance(diff_task[img_key], dict):
                if 'pred_prob' in diff_task[img_key]:
                    del diff_task[img_key]['pred']
                    del diff_task[img_key]['pred_prob']
#                     print('exp1_task[img_key].keys():', exp1_task[img_key].keys())
                    pred1 = exp1_task[img_key]['pred']
                    pred2 = exp2_task[img_key]['pred']
                    diff_task[img_key]['exp1_pred'] = pred1
                    diff_task[img_key]['exp2_pred'] = pred2
        all_tasks_diff.append(diff_task)
    
    diff12_increase_tasks = sorted(all_tasks_diff, key = lambda i: i['1-2_acc'])
    exp_postfix_2_gt_1 = exp1_postfix + '_lt_' + exp2_postfix
    print('Drawing tasks Exp2 better than Exp1 ...')
    draw_tasks(diff12_increase_tasks, n_tasks = n_tasks, 
#                save_img_folder = save_img_folder, exp_postfix = exp_postfix_2_gt_1, compare_diff = True)
               save_img_folder = save_img_folder, exp_postfix = exp_postfix_2_gt_1, mode = 'compare_diff')
    
def draw_tasks(all_tasks, n_tasks, save_img_folder = None, exp_postfix = None, mode = 'single_exp'):
    '''
    mode (str): 'single_exp'|'compare_diff'|'both_exp_avg'
    '''
# def draw_tasks(all_tasks, n_tasks, save_img_folder = None, exp_postfix = None, compare_diff = False):
    # draw top ? task imgs
#     print('len(all_tasks):'+str(len(all_tasks)))
#     print('all_tasks:', type(all_tasks), len(all_tasks)) # n_episodes
#     print('all_tasks[0]:', all_tasks[0].keys(), type(all_tasks[0]['acc'])) # 'acc'(float), 'c00_qu14'
    print('drawing with', n_tasks, 'tasks...')
    for i in range(n_tasks):
        print('the', i + 1, 'task')
        task = all_tasks[i]
        task_str = 'task_' + str(i + 1).zfill(2)
        save_filename = exp_postfix + '_' + task_str + '.png'
#         save_img_folder = os.path.join(self.record_folder, 'imgs')
        draw_single_task(
            task = task, save_filename = save_filename, save_img_folder = save_img_folder, 
#             compare_diff = compare_diff
            mode = mode
        )

# def draw_single_task(task, save_filename = None, save_img_folder = None, compare_diff = False):
def draw_single_task(task, save_filename = None, save_img_folder = None, mode = 'single_exp'):
    '''
    mode (str): 'single_exp'|'compare_diff'|'both_exp_avg'
    '''
    # TODO: automatically decide n_way
    n_way = 2
#     n_way = 5
    n_support = 5
    n_query = 15

    n_col = n_way
    n_row = n_support + n_query
    unit_size = 3
#     plt.figure()
#     img = plt.imread(task['c00_qu00']['path'])
#     plt.imshow(img)
#     plt.show()

    fig, axarr = plt.subplots(
        n_row, n_col, 
        figsize=(unit_size*n_col, unit_size*n_row) # WTF?????????? WHY???????
    )
#     fig.tight_layout()
#     plt.subplots_adjust(top=n_row/(n_row+1)) # to set margin for suptitle 'after' tight_layout()
    if mode == 'compare_diff':
        title_str = 'acc1=%.3f%%, acc2=%.3f%%'%(task['exp1_acc'], task['exp2_acc'])
        fig.suptitle(title_str, size = unit_size*n_col*3)
    elif mode == 'both_exp_avg':
        # TODO: 
        title_str = 'avg acc = %.3f%%'%(task['both_avg_acc'])
        fig.suptitle(title_str, size = unit_size*n_col*3)
    elif mode == 'single_exp':
        pass
#         plt.figure()
    
#     for n in range(n_row): # for each data per class
#         if n < n_support:
#             data_str = 'su' + str(n).zfill(2)
#         else:
#             data_str = 'qu' + str(n-n_support).zfill(2)
#         for cl in range(n_col): # for each class
#             cl_str = 'c' + str(cl).zfill(2)
    for cl in range(n_col): # for each class
        cl_str = 'c' + str(cl).zfill(2)
        
        if mode in ['compare_diff', 'both_exp_avg']:
            # make error coordinate on the top
            top_query_id = 0
            bottom_query_id = n_query - 1
        
        for n in range(n_row): # for each data per class
            if n < n_support:
                data_str = 'su' + str(n).zfill(2)
            else:
                data_str = 'qu' + str(n-n_support).zfill(2)
            
            
            key = cl_str + '_' + data_str
            path = task[key]['path']

            if 'su' in key:
                alpha = 1 # plot transparency ( 1 for solid)
                plt_row_id = n
            if 'qu' in key:
                alpha = 0.6 # plot transparency ( 1 for solid)
                
                if mode == 'compare_diff':
                    
                    plt_row_id = None
                    
                    pred1 = task[key]['exp1_pred']
                    pred2 = task[key]['exp2_pred']
                    
                    if cl != pred1 and cl == pred2: # exp2 correct, but exp1 error
                        plt_row_id = n_support + top_query_id
                        top_query_id += 1
                        
                        alpha = 1
#                         axarr[n, cl].title.set_color('r')
                        axarr[plt_row_id, cl].title.set_color('r')
                    elif cl != pred1 and cl != pred2: # exp1 and exp2 both error
                        plt_row_id = n_support + bottom_query_id
                        bottom_query_id -= 1
                        
#                         axarr[n, cl].title.set_color('b')
                        axarr[plt_row_id, cl].title.set_color('b')
                    elif cl == pred1 and cl == pred2: # exp1 and exp2 both correct
                        plt_row_id = n_support + bottom_query_id
                        bottom_query_id -= 1
                    else: # exp1 correct, but exp2 error (no use)
                        plt_row_id = n_support + bottom_query_id
                        bottom_query_id -= 1
                    
                    sub_title_str = 'pred1=%s\npred2=%s'%(pred1, pred2)
                    sub_title_size = unit_size * 4
#                     axarr[n, cl].title.set_text(sub_title_str)
#                     axarr[n, cl].title.set_size(sub_title_size)
                    axarr[plt_row_id, cl].title.set_text(sub_title_str)
                    axarr[plt_row_id, cl].title.set_size(sub_title_size)

                elif mode == 'single_exp':
                    pred = task[key]['pred']
                    pred_prob = task[key]['pred_prob']
                    is_correct = pred == cl
                    
                elif mode == 'both_exp_avg':
                    pred1 = task[key]['exp1_pred']
                    pred2 = task[key]['exp2_pred']
                    both_error = (pred1 != cl) and (pred2 != cl)
                    
                    if both_error:
                        plt_row_id = n_support + top_query_id
                        top_query_id += 1
                        alpha = 1
                        
                        axarr[plt_row_id, cl].title.set_color('r')
                    else:
                        plt_row_id = n_support + bottom_query_id
                        bottom_query_id -= 1
                    
                    # TODO:
                    sub_title_str = 'pred1 = %s\npred2 = %s'%(pred1, pred2)
                    sub_title_size = unit_size * 4
                    axarr[plt_row_id, cl].title.set_text(sub_title_str)
                    axarr[plt_row_id, cl].title.set_size(sub_title_size)

#             img = plt.imread(path)
            img = plt.imread(path, 0) # BUGFIX: ValueError: invalid PNG header
            if len(img.shape) == 2:
                axarr[plt_row_id, cl].imshow(
#                 axarr[n, cl].imshow(
                    img, cmap=cm.gray, 
                    aspect=1, # set aspect to avoid showing with actual size
                    alpha=alpha
                ) 
            else:
                axarr[plt_row_id, cl].imshow(
#                 axarr[n, cl].imshow(
                    img, 
                    aspect=1, # set aspect to avoid showing with actual size
                    alpha=alpha
                )
#             axarr[n, cl].set_axis_off()
    fig.tight_layout()
    plt.subplots_adjust(
        top = n_row/(n_row+1), # set margin for suptitle 'after' tight_layout()
        hspace = 0.7 # subplot height margin (for subtitle)
    )
    plt.show()
#         save_img_folder = os.path.join(self.record_folder, 'imgs')
    if not os.path.exists(save_img_folder):
        os.mkdir(save_img_folder)
#         filename = 'tmp.png'
    if save_filename is not None:
        save_path = os.path.join(save_img_folder, save_filename)
        print('Saving image to:', save_path)
        fig.savefig(save_path)#, bbox_inches='tight')

def get_best_all_tasks(pkl_path):
    print('loading self.results_pkl from:', pkl_path)
    with open(pkl_path, 'rb') as handle:
        results = pickle.load(handle)
    # get best exp task_datas
    sorted_res = sorted(results, key = lambda i: -float(i['val_acc_mean']))
    best_res = sorted_res[0]
    best_model_all_tasks = best_res['novel_task_datas']
    print('best_model_all_tasks[0]["c00_qu00"]:', best_model_all_tasks[0]["c00_qu00"])
    return best_model_all_tasks


def main(args):
    pass


if __name__=='__main__':
    args = None
    main(args)






