import numpy as np
from io_utils import parse_args
from train import exp_train_val
from save_features import exp_save_features
from test import exp_test, record_to_csv
from param_utils import get_all_params_comb, get_all_args_ls, get_modified_args, copy_args, get_matched_df
import pandas as pd
import os
from my_utils import del_keys
import logging

class ExpManager:
    def __init__(self, base_params, train_fixed_params, test_fixed_params, general_possible_params, test_possible_params):
        '''
        Args:
            base_params (dict): the core settings of the experiment
            train_fixed_params (dict): e.g. {'stop_epoch':700, 'gpu_id':0}
            test_fixed_params (dict): e.g. {'record_csv':'program_time.csv', 'gpu_id':0}
            general_possible_params (dict): dictionary of list of tunable parameters, different param would train different model. e.g. {'dropout_p':[0.25, 0.5]}
        '''
        self.base_params = base_params # general fixed params
        self.possible_params = {'general':general_possible_params, 'test':test_possible_params} # general means generalize to train/save_features/test
        self.fixed_params = {'train':train_fixed_params, 'test':test_fixed_params}
        self.negligible_vars = ['gpu_id', 'csv_name',] # can be ignored when comparing results but in ArgParser
        
        self.results = [] # params as well as results restore in the list of dictionaries
        self.record_folder = './record'
        self.csv_path = os.path.join(self.record_folder, test_fixed_params['csv_name'])
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
    
    def exp_grid(self, choose_by='val_acc_mean', resume=False):
        
        print('exp_grid() start.')
        print(self.base_params)
        default_args = {} # the raw default args of the code
        default_args['train'] = parse_args('train', parse_str='')
        default_args['test'] = parse_args('test', parse_str='')
        possible_params = self.possible_params
        
        train_args = get_modified_args(default_args['train'], {**self.base_params, **self.fixed_params['train']})
        test_args = get_modified_args(default_args['test'], {**self.base_params, **self.fixed_params['test']})
        all_general_params = get_all_params_comb(possible_params=possible_params['general'])
        all_test_params = get_all_params_comb(possible_params=possible_params['test'])
        
        csv_path = os.path.join(self.record_folder, self.fixed_params['test']['csv_name'])
        
        # TODO: refactor 成許多 function，有 argument: resume (boolean)
        is_csv_exists = os.path.exists(csv_path)
        if is_csv_exists:
            loaded_df = pd.read_csv(csv_path)
            is_csv_new = len(loaded_df)==0
        else:
            is_csv_new = True
#         if resume:
#             assert not is_csv_new, "csv file should exist and be filled with some content."
#         else: # new experiments
#             assert is_csv_new, "csv file shouldn't exist or should be empty."
        
        for params in all_general_params:
            
            if resume:
                print()
                print('='*20, 'Checking if already trained in:', csv_path, '='*20)
                print(params)
                # read csv
                loaded_df = pd.read_csv(csv_path)

                # check if ALL test_params has been experimented. if so, then do NOT even train model
                check_df = loaded_df.copy()
                check_param = {**self.base_params, **params}
                check_df = get_matched_df(check_param, check_df)
                num_experiments = len(check_df)
                if num_experiments!=0:
                    should_train = False
                else:
                    should_train = True
                
                if should_train:
                    # train model
                    print()
                    print('='*20, 'Training', '='*20)
                    print(params)
                    modified_train_args = get_modified_args(train_args, params)
                    train_result = exp_train_val(modified_train_args)
                else:
                    print('NO need to train since already trained in record:', csv_path)
            
            else: # not resume
                # train model
                print()
                print('='*20, 'Training', '='*20)
                print(params)
                modified_train_args = get_modified_args(train_args, params)
                train_result = exp_train_val(modified_train_args)
            
            modified_test_args = get_modified_args(test_args, params)
            
            # loop over testing settings under each general setting
            for test_params in all_test_params:
                if resume:
                    print()
                    print('='*20, 'Checking if already did experiments', '='*20)
                    print(params)
                    print(test_params)
                    check_df = loaded_df.copy()
                    check_param = {**self.base_params, **params, **test_params}
                    check_df = get_matched_df(check_param, loaded_df)
                    num_test_experiments = len(check_df)
                    if num_test_experiments>0: # already experiments
                        print('NO need to test since already tested %s times in record: %s'%(num_test_experiments, csv_path))
                        continue
                    
                final_test_args = get_modified_args(modified_test_args, test_params)
                
                splits = ['val', 'novel'] # temporary no 'train'
                write_record = {**params, **test_params}
                if resume:
                    if should_train:
                        write_record['train_acc_mean'] = train_result['train_acc']
                    else:
                        # to get train_acc, so no need test_params
                        check_df = loaded_df.copy()
                        check_df = get_matched_df({**self.base_params, **params}, check_df)
                        write_record['train_acc_mean'] = check_df['train_acc_mean'].iloc[0]
                else:
                    write_record['train_acc_mean'] = train_result['train_acc']
                
                for split in splits:
                    
                    split_final_test_args = copy_args(final_test_args)
                    split_final_test_args.split = split
                    print()
                    print('='*20, 'Saving Features', '='*20)
                    print(params)
                    print(test_params)
                    print('data split:', split)
                    exp_save_features(copy_args(split_final_test_args))
                    print()
                    print('='*20, 'Testing', '='*20)
                    print(params)
                    print(test_params)
                    print('data split:', split)
                    record = exp_test(copy_args(split_final_test_args), iter_num=600, should_del_features=True)
                    write_record['epoch'] = record['epoch']
                    write_record[split+'_acc_mean'] = record['acc_mean']
                
                self.results.append(write_record)
                print('Saving record to:', csv_path)
                record_to_csv(final_test_args, write_record, csv_path=csv_path)
                
        # TODO: can also loop dataset
        for choose_by in ['val_acc_mean', 'novel_acc_mean']:
            # TODO: 改成讀 csv 來判斷吧?
            record_df = pd.read_csv(csv_path)
            record_df = get_matched_df(self.base_params, record_df)
            
            sorted_df = record_df.sort_values(by=choose_by, ascending=False)
            compare_cols = list(self.possible_params['general'].keys())+list(self.possible_params['test'].keys())
            compare_cols = compare_cols + ['val_acc_mean', 'novel_acc_mean']
            top_k = 10
            print()
            print('Best Test Acc: %s, selected by %s'%(sorted_df['novel_acc_mean'].iloc[0], choose_by))
            print()
            print('='*20,'Top %s results sorted by: %s'%(top_k, choose_by), '='*20)
            print(sorted_df[compare_cols].head(top_k))
            
#             print('self.results:', self.results)
            
#             sorted_result = sorted(self.results, key = lambda i: i[choose_by], reverse=True)
#             best_result = sorted_result[0]
            
#             print('The best test acc is', best_result['novel_acc_mean'],'% on grid search chosen by:',choose_by)
#             print('Detail:\n', best_result)
        
    def sum_up_results(self, choose_by, top_k): # choose the best according to dataset & split
        csv_path = os.path.join(self.record_folder, self.fixed_params['test']['csv_name'])
        print('Reading file:', csv_path)
        record_df = pd.read_csv(csv_path)
        
        default_test_args = parse_args('test', parse_str='')
        default_test_params = default_test_args.__dict__ # to avoid including other exps
        important_fixed_params = {**default_test_params, **self.base_params, **self.fixed_params['test']}

        # delete negligible_vars & changeable vars
        del_keys(important_fixed_params, self.negligible_vars)
        del_keys(important_fixed_params, self.possible_params['general'].keys())
        del_keys(important_fixed_params, self.possible_params['test'].keys())
        
        all_possible_params = {**self.possible_params['general'], **self.possible_params['test']}
        
        matched_df = get_matched_df(important_fixed_params, record_df, possible_params=all_possible_params)
#         logging.debug('matched_df:\n%s'%(matched_df))

        if len(matched_df)!=0:
            sorted_df = matched_df.sort_values(by=choose_by, ascending=False)
            compare_cols = list(self.possible_params['general'].keys())+list(self.possible_params['test'].keys())
            compare_cols = compare_cols + ['val_acc_mean', 'novel_acc_mean']
            print()
            print('Best Test Acc: %s, selected by %s'%(sorted_df['novel_acc_mean'].iloc[0], choose_by))
            print()
            print('='*20,'Top %s/%s results sorted by: %s'%(top_k, len(matched_df), choose_by), '='*20)
            print(sorted_df[compare_cols].head(top_k))
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



def main(args):
    pass


if __name__=='__main__':
    args = None
    main(args)






