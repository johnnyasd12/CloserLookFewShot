import numpy as np
from io_utils import parse_args
from train import exp_train_val
from save_features import exp_save_features
from test import exp_test, record_to_csv
from param_utils import get_all_params_comb, get_all_args_ls, get_modified_args, copy_args, get_matched_df
import pandas as pd
import os

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
import matplotlib.pyplot as plt
# to draw gray images ???
import matplotlib.cm as cm
# to draw task
import copy

class ExpManager:
    def __init__(self, base_params, train_fixed_params, test_fixed_params, general_possible_params, test_possible_params, pkl_postfix):
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
        self.negligible_vars = ['gpu_id', 'csv_name',] # can be ignored when comparing results but in ArgParser
        
        self.results = [] # params as well as results restore in the list of dictionaries
        self.record_folder = './record'
        self.csv_path = os.path.join(self.record_folder, test_fixed_params['csv_name'])
        self.pkl_postfix = pkl_postfix
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
    
    def exp_grid(self, choose_by='val_acc_mean', mode='from_scratch'):
        '''
        Args:
            mode (str): 'from_scratch'|'resume'|'draw_tasks'
        '''
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
        
        # TODO: refactor 成許多 function，有 argument: mode
        is_csv_exists = os.path.exists(csv_path)
        if is_csv_exists:
            loaded_df = pd.read_csv(csv_path)
            is_csv_new = len(loaded_df)==0
        else:
            is_csv_new = True
#         if mode == 'resume':
#             assert not is_csv_new, "csv file should exist and be filled with some content."
#         else: # new experiments
#             assert is_csv_new, "csv file shouldn't exist or should be empty."
        
    
        pkl_postfix_str = '_' + self.pkl_postfix + '.pkl'
        pkl_path = csv_path.replace('.csv', pkl_postfix_str)
        if mode == 'resume':
            print('loading self.results from:', pkl_path)
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as handle:
                    self.results = pickle.load(handle)
            else:
                logging.warning('No previous result pickle file, only record self.results from scratch.')
            print('self.results at begin, len =', len(self.results))
        
        for params in all_general_params:
            
            if mode == 'resume':# or mode == 'draw_tasks':
                print()
                print('='*20, 'Checking if already trained in:', csv_path, '='*20)
                print('base_params:', self.base_params)
                print('general_params:', params)
                # read csv
                loaded_df = pd.read_csv(csv_path)

                # check if ALL test_params has been experimented. if so, then do NOT even train model
                check_df = loaded_df.copy()
                check_param = {**self.base_params, **params}
                check_df = get_matched_df(check_param, check_df)
                num_experiments = len(check_df)
                should_train = num_experiments == 0
                
            elif mode == 'from_scratch':
                should_train = True
            elif mode == 'draw_tasks':
                should_train = False
            
            if should_train:
                # train model
                print()
                print('='*20, 'Training', '='*20)
                print(params)
                modified_train_args = get_modified_args(train_args, params)
                train_result = exp_train_val(modified_train_args)
            else:
                if mode == 'draw_tasks':
                    print('NO need to train when draw_tasks.')
                else:
                    print('NO need to train since already trained in record:', csv_path)
            
            modified_test_args = get_modified_args(test_args, params)
            
            # loop over testing settings under each general setting
            for test_params in all_test_params:
                if mode == 'resume':
                    print('\n', '='*20, 'Checking if already did experiments', '='*20)
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
                
                write_record = {**params, **test_params}
                if mode == 'resume': #in ['resume', 'draw_tasks']:
                    if should_train:
                        write_record['train_acc_mean'] = train_result['train_acc']
                    else:
                        # to get train_acc, so no need test_params
                        check_df = loaded_df.copy()
                        check_df = get_matched_df({**self.base_params, **params}, check_df)
                        write_record['train_acc_mean'] = check_df['train_acc_mean'].iloc[0]
                elif mode == 'from_scratch':
                    write_record['train_acc_mean'] = train_result['train_acc']
                
                if mode in ['from_scratch', 'resume']:
                    splits = ['val', 'novel'] # temporary no 'train'
                    for split in splits: # val, novel

                        split_final_test_args = copy_args(final_test_args)
                        split_final_test_args.split = split
                        print('\n', '='*20, 'Saving Features', '='*20)
                        print('params:', params)
                        print('test_params:', test_params)
                        print('data split:', split)
                        exp_save_features(copy_args(split_final_test_args))
                        print('\n', '='*20, 'Testing', '='*20)
                        print('params:', params)
                        print('test_params:', test_params)
                        print('data split:', split)
                        n_episodes = 10 if split_final_test_args.debug or mode=='draw_tasks' else 600

                        exp_record, task_datas = exp_test(copy_args(split_final_test_args), n_episodes=n_episodes, should_del_features=True)#, show_data=show_data)
                        write_record['epoch'] = exp_record['epoch']
                        write_record[split+'_acc_mean'] = exp_record['acc_mean']
                    
                    print('Saving record to:', csv_path)
                    record_to_csv(final_test_args, write_record, csv_path=csv_path)
                    write_record['novel_task_datas'] = task_datas # currently ignore val_task_datas
                    self.results.append(write_record)
                    
                    print('Saving self.results into:', pkl_path)
                    with open(pkl_path, 'wb') as handle:
                        pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # TODO: can also loop dataset
        for choose_by in ['val_acc_mean', 'novel_acc_mean']:
            # read csv to compare results
            top_k = None
            self.sum_up_results(choose_by, top_k)
        
        # TODO: 5/12 save task_datas in file
        # directly save self.results in file 'csv_name.pkl'????????????
        # TODO: CANNOT just save to csv_path.pkl since different expmgr might have the same csv_path
        
#         if mode in ['from_scratch', 'resume']:
#             print('saving self.results into:', pkl_path)
#             with open(pkl_path, 'wb') as handle:
#                 pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if mode == 'draw_tasks':
            print('loading self.results from:', pkl_path)
            best_model_all_tasks = get_best_all_tasks(pkl_path = pkl_path)
            
            print('sorting tasks...')
            sorted_tasks = sorted(best_model_all_tasks, key = lambda i: float(i['acc'])) # in ascending order
            print('Draw Worst Tasks...')
            save_img_folder = os.path.join(self.record_folder, 'imgs')
            draw_tasks(
                sorted_tasks, n_tasks = 3, 
                save_img_folder = save_img_folder, exp_postfix = self.pkl_postfix
            ) # utilize self.results, save all_tasks of best res
#             self.draw_tasks(best_model_all_tasks, n_tasks = 3) # utilize self.results, save all_tasks of best res

    def sum_up_results(self, choose_by, top_k, show_same_params=True): # choose the best according to dataset & split
        
        def select_cols_if_exists(df, cols: list):
            for col in cols.copy(): # BUGFIX: some of col not removed
                if col not in list(df.columns):
                    logging.warning('sum_up_results()/"%s" not in dataframe, deleted from cols.'%(col))
                    cols.remove(col)
            return df[cols]
        
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
        
        all_possible_params = {**self.possible_params['general'], **self.possible_params['test']}
        
        matched_df = get_matched_df(important_fixed_params, record_df, possible_params=all_possible_params)
#         logging.debug('matched_df:\n%s'%(matched_df))

        if top_k==None:
            top_k = len(matched_df)
        if len(matched_df)!=0:
            sorted_df = matched_df.sort_values(by=choose_by, ascending=False)
            compare_cols = list(self.possible_params['general'].keys())+list(self.possible_params['test'].keys())
            compare_cols = compare_cols + ['train_acc_mean', 'val_acc_mean', 'novel_acc_mean']
            print()
            print('Best Test Acc: %s, selected by %s'%(sorted_df['novel_acc_mean'].iloc[0], choose_by))
            print()
            print('='*20,'Top %s/%s results sorted by: %s'%(top_k, len(matched_df), choose_by), '='*20)
            show_df = select_cols_if_exists(sorted_df, compare_cols)
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


def draw_most_differ_tasks(pkl_path1, pkl_path2, n_tasks, save_img_folder, exp1_postfix, exp2_postfix):
    best_exp_1_all_task = get_best_all_tasks(pkl_path1)
    best_exp_2_all_task = get_best_all_tasks(pkl_path2)
    
    all_tasks_diff = []
    for i in range(len(best_exp_1_all_task)):
        exp_1_task = best_exp_1_all_task[i]
        exp_2_task = best_exp_2_all_task[i]
        
#         diff_task = exp_1_task.copy()
        diff_task = copy.deepcopy(exp_1_task)
#         print('diff_task.keys():', diff_task.keys()) # acc, img_keys
        del diff_task['acc']
        acc1 = exp_1_task['acc']
        acc2 = exp_2_task['acc']
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
#                     print('exp_1_task[img_key].keys():', exp_1_task[img_key].keys())
                    pred1 = exp_1_task[img_key]['pred']
                    pred2 = exp_2_task[img_key]['pred']
                    diff_task[img_key]['exp1_pred'] = pred1
                    diff_task[img_key]['exp2_pred'] = pred2
        all_tasks_diff.append(diff_task)
    
    diff12_increase_tasks = sorted(all_tasks_diff, key = lambda i: i['1-2_acc'])
    exp_postfix_2_gt_1 = exp2_postfix + '_gt_' + exp1_postfix
    print('Drawing tasks Exp2 better than Exp1 ...')
    draw_tasks(diff12_increase_tasks, n_tasks = n_tasks, 
               save_img_folder = save_img_folder, exp_postfix = exp_postfix_2_gt_1, compare_diff = True)

#     diff21_increase_tasks = sorted(all_tasks_diff, key = lambda i: i['2-1_acc'])
#     for i in range(3):
#         print('diff12_increase_tasks[i]["1-2_acc"]:', diff12_increase_tasks[i]['1-2_acc'])
#     for i in range(3):
#         print('diff21_increase_tasks[i]["1-2_acc"]:', diff21_increase_tasks[i]['1-2_acc'])
#     exp_postfix_2_lt_1 = exp2_postfix + '_lt_' + exp1_postfix
#     print('Drawing tasks Exp1 better than Exp2 ...')
#     draw_tasks(diff21_increase_tasks, n_tasks = n_tasks, 
#                save_img_folder = save_img_folder, exp_postfix = exp_postfix_2_lt_1, compare_diff = True)
    

def draw_tasks(all_tasks, n_tasks, save_img_folder = None, exp_postfix = None, compare_diff = False):
    # draw top ? task imgs
#     print('len(all_tasks):'+str(len(all_tasks)))
#     print('all_tasks:', type(all_tasks), len(all_tasks)) # n_episodes
#     print('all_tasks[0]:', all_tasks[0].keys(), type(all_tasks[0]['acc'])) # 'acc'(float), 'c00_qu14'
    print('drawing with', n_tasks, 'tasks...')
    for i in range(n_tasks):
        print('the', i + 1, 'task')
        task = all_tasks[i]
        save_filename = 'task_' + str(i + 1) + '_' + exp_postfix + '.png'
#         save_img_folder = os.path.join(self.record_folder, 'imgs')
        draw_single_task(
            task = task, save_filename = save_filename, save_img_folder = save_img_folder, 
            compare_diff = compare_diff
        )


def draw_single_task(task, save_filename = None, save_img_folder = None, compare_diff = False):
    n_way = 5
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
    fig.tight_layout()
    plt.subplots_adjust(top=n_row/(n_row+1)) # to set margin for suptitle 'after' tight_layout()
    if compare_diff:
        title_str = 'acc1=%s%%, acc2=%s%%'%(task['exp1_acc'], task['exp2_acc'])
        fig.suptitle(title_str, size = unit_size*7)
    else:
        pass
#         plt.figure()
    for n in range(n_row): # for each data per class
        if n < n_support:
            data_str = 'su' + str(n).zfill(2)
        else:
            data_str = 'qu' + str(n-n_support).zfill(2)
        for cl in range(n_col): # for each class
            cl_str = 'c' + str(cl).zfill(2)
            key = cl_str + '_' + data_str
            path = task[key]['path']
#                 print(task[key].keys())
#                 print('key:', key)
#                 print('path:', path)
            if 'qu' in key:
                if compare_diff:
                    pred1 = task[key]['exp1_pred']
                    pred2 = task[key]['exp2_pred']
                    sub_title_str = 'pred1=%s, pred2=%s'%(pred1, pred2)
                    sub_title_size = unit_size * 4
                    axarr[n, cl].title.set_text(sub_title_str)
                    axarr[n, cl].title.set_size(sub_title_size)
                    if cl != pred1 and cl == pred2: # exp2 correct, but exp1 error
                        axarr[n, cl].title.set_color('r')
                    elif cl != pred1 and cl != pred2: # exp1 and exp2 both error
                        axarr[n, cl].title.set_color('b')
                else:
                    pred = task[key]['pred']
                    pred_prob = task[key]['pred_prob']
                    is_correct = pred == cl

            img = plt.imread(path)
            if len(img.shape) == 2:
                axarr[n, cl].imshow(img, cmap=cm.gray, aspect=1) # set aspect to avoid showing with actual size
            else:
                axarr[n, cl].imshow(img, aspect=1) # set aspect to avoid showing with actual size
            
#             if compare_diff:
#                 if 'qu' in key:
#                     sub_title_str = 'pred=%s'%(pred)
#                     axarr[n, cl].title.set_text(sub_title_str)
#                     if cl != pred:
#                         axarr[n, cl].title.set_color('r')
#             else:
#                 pass

#     if compare_diff:
#         title_str = 'acc1=%s%%, acc2=%s%%'%(task['exp1_acc'], task['exp2_acc'])
#         fig.suptitle(title_str)
#     else:
#         pass
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
    print('loading self.results from:', pkl_path)
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






