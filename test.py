import os
# for better error message when encounter RuntimeError: CUDA error: device-side assert triggered
if False:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet, ProtoNetAE, ProtoNetAE2
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
# from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file, decoder_dict, get_checkpoint_dir
from io_utils import *
from my_utils import *
from my_utils import set_random_seed
import pandas as pd
from tqdm import tqdm

from model_utils import get_model
import datetime

import gc

def exp_test(params, n_episodes, should_del_features=False):#, show_data=False):
    start_time = datetime.datetime.now()
    print('exp_test() started at',start_time)
    
    set_random_seed(0) # successfully reproduce "normal" testing. 
    
    if params.gpu_id:
        set_gpu_id(params.gpu_id)
    
#     acc_all = []

    model = get_model(params, 'test')
    
    ########## get settings ##########
    n_shot = params.test_n_shot if params.test_n_shot is not None else params.n_shot
    few_shot_params = dict(n_way = params.test_n_way , n_support = n_shot)
    if params.gpu_id:
        model = model.cuda()
    else:
        model = to_device(model)
    checkpoint_dir = get_checkpoint_dir(params)
    print('loading from:',checkpoint_dir)
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
    else:
        modelfile   = get_best_file(checkpoint_dir)
    
    ########## load model ##########
    if modelfile is not None:
        if params.gpu_id is None:
            tmp = torch.load(modelfile)
        else: # TODO: figure out WTF is going on here
            print('params.gpu_id =', params.gpu_id)
            map_location = 'cuda:0'
#             gpu_str = 'cuda:' + '0'#str(params.gpu_id)
#             map_location = {'cuda:1':gpu_str, 'cuda:0':gpu_str} # see here: https://hackmd.io/koKAo6kURn2YBqjoXXDhaw#RuntimeError-CUDA-error-invalid-device-ordinal
            tmp = torch.load(modelfile, map_location=map_location)
#                 tmp = torch.load(modelfile)
        if not params.method in ['baseline', 'baseline++'] : 
            # if 'baseline' or 'baseline++' then NO NEED to load model !!!
            model.load_state_dict(tmp['state'])
            print('Model successfully loaded.')
        else:
            print('No need to load model for baseline/baseline++ when testing.')
        load_epoch = int(tmp['epoch'])
    
    ########## testing ##########
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        image_size = get_img_size(params)
        load_file = get_loadfile_path(params, params.split)

        datamgr         = SetDataManager(image_size, n_episode = n_episodes, n_query = 15 , **few_shot_params)
        
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)
        
        ########## last record and post-process ##########
        torch.cuda.empty_cache()
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # TODO afterward: compute this
        acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(n_episodes))
        # writing settings into csv
        acc_mean_str = '%4.2f' % (acc_mean)
        acc_std_str = '%4.2f' %(acc_std)
        # record beyond params
        extra_record = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str, 'epoch':load_epoch}
        if should_del_features:
            del_features(params)
        end_time = datetime.datetime.now()
        print('exp_test() start at', start_time, ', end at', end_time, '.\n')
        print('exp_test() totally took:', end_time-start_time)
        return extra_record, task_datas

    else: # not MAML
        acc_all = []
#         # draw_task: initialize task acc(actually can replace acc_all), img_path, img_is_correct, etc.
#         task_datas = [None]*n_episodes # list of dict
        # directly use extracted features
        all_feature_files = get_all_feature_files(params)
        
        if params.n_test_candidates is None: # common setting (no candidate)
            # draw_task: initialize task acc(actually can replace acc_all), img_path, img_is_correct, etc.
            task_datas = [None]*n_episodes # list of dict
            
            feature_file = all_feature_files[0]
            cl_feature, cl_filepath = feat_loader.init_loader(feature_file, return_path=True)
            cl_feature_single = [cl_feature]
            
            for i in tqdm(range(n_episodes)):
                # TODO afterward: fix data list? can only fix class list?
                task_data = feature_evaluation(
                    cl_feature_single, model, params=params, n_query=15, **few_shot_params, 
                    cl_filepath=cl_filepath,
                )
                acc = task_data['acc']
                acc_all.append(acc)
                task_datas[i] = task_data
            
            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('loaded from %d epoch model.' %(load_epoch))
            print('%d episodes, Test Acc = %4.2f%% +- %4.2f%%' %(n_episodes, acc_mean, 1.96* acc_std/np.sqrt(n_episodes)))
            
            ########## last record and post-process ##########
            torch.cuda.empty_cache()
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            # TODO afterward: compute this
            acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(n_episodes))
            # writing settings into csv
            acc_mean_str = '%4.2f' % (acc_mean)
            acc_std_str = '%4.2f' %(acc_std)
            # record beyond params
            extra_record = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str, 'epoch':load_epoch}
            if should_del_features:
                del_features(params)
            end_time = datetime.datetime.now()
            print('exp_test() start at', start_time, ', end at', end_time, '.\n')
            print('exp_test() totally took:', end_time-start_time)
            return extra_record, task_datas
        else: # n_test_candidates settings
                
            candidate_cl_feature = [] # features of each class of each candidates
            print('Loading features of %s candidates into dictionaries...' %(params.n_test_candidates))
            for n in tqdm(range(params.n_test_candidates)):
                nth_feature_file = all_feature_files[n]
                cl_feature, cl_filepath = feat_loader.init_loader(nth_feature_file, return_path=True)
                candidate_cl_feature.append(cl_feature)

            print('Evaluating...')
            
            # TODO: frac_acc_all
            is_single_exp = not isinstance(params.frac_ensemble, list)
            if is_single_exp:
                # draw_task: initialize task acc(actually can replace acc_all), img_path, img_is_correct, etc.
                task_datas = [None]*n_episodes # list of dict
                ########## test and record acc ##########
                for i in tqdm(range(n_episodes)):
                    # TODO afterward: fix data list? can only fix class list?

                    task_data = feature_evaluation(
                        candidate_cl_feature, model, params=params, n_query=15, **few_shot_params, 
                        cl_filepath=cl_filepath,
                    )
                    acc = task_data['acc']
                    acc_all.append(acc)
                    task_datas[i] = task_data
                    
                    collected = gc.collect()
#                     print("Garbage collector: collected %d objects." % (collected))

                acc_all  = np.asarray(acc_all)
                acc_mean = np.mean(acc_all)
                acc_std  = np.std(acc_all)
                print('loaded from %d epoch model.' %(load_epoch))
                print('%d episodes, Test Acc = %4.2f%% +- %4.2f%%' %(n_episodes, acc_mean, 1.96* acc_std/np.sqrt(n_episodes)))
                collected = gc.collect()
                print("garbage collector: collected %d objects." % (collected))
                
                ########## last record and post-process ##########
                torch.cuda.empty_cache()
                timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                # TODO afterward: compute this
                acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(n_episodes))
                # writing settings into csv
                acc_mean_str = '%4.2f' % (acc_mean)
                acc_std_str = '%4.2f' %(acc_std)
                # record beyond params
                extra_record = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str, 'epoch':load_epoch}
                if should_del_features:
                    del_features(params)
                end_time = datetime.datetime.now()
                print('exp_test() start at', start_time, ', end at', end_time, '.\n')
                print('exp_test() totally took:', end_time-start_time)
                return extra_record, task_datas
            else: ########## multi-frac_ensemble exps ##########
                
                ########## (haven't modified) test and record acc ##########
                n_fracs = len(params.frac_ensemble)
                
                ##### initialize frac_data #####
                frac_acc_alls = [[0]*n_episodes for _ in range(n_fracs)]
                frac_acc_means = [None]*n_fracs
                frac_acc_stds = [None]*n_fracs
                # draw_task: initialize task acc(actually can replace acc_all), img_path, img_is_correct, etc.
                ep_task_data_each_frac = [[None]*n_episodes for _ in range(n_fracs)] # list of list of dict

                for ep_id in tqdm(range(n_episodes)):
                    # TODO afterward: fix data list? can only fix class list?
                    
                    # TODO my_utils.py: feature_eval return frac_task_data
                    frac_task_data = feature_evaluation(
                        candidate_cl_feature, model, params=params, n_query=15, **few_shot_params, 
                        cl_filepath=cl_filepath,
                    )
                    for frac_id in range(n_fracs):
                        task_data = frac_task_data[frac_id]
                        # TODO: i think here's the problem???
                        acc = task_data['acc']
                        frac_acc_alls[frac_id][ep_id] = acc
                        ep_task_data_each_frac[frac_id][ep_id] = task_data
                        
                        collected = gc.collect()
#                         print("Garbage collector: collected %d objects." % (collected))
                    collected = gc.collect()
#                     print("Garbage collector: collected %d objects." % (collected))
                ### debug
#                 print('frac_acc_alls:', frac_acc_alls)
#                 yee
                for frac_id in range(n_fracs):
                    frac_acc_alls[frac_id]  = np.asarray(frac_acc_alls[frac_id])
                    acc_all = frac_acc_alls[frac_id]
                    acc_mean = np.mean(acc_all)
                    acc_std = np.std(acc_all)
                    frac_acc_means[frac_id] = acc_mean
                    frac_acc_stds[frac_id]  = acc_std
                    print('loaded from %d epoch model, frac_ensemble:'%(load_epoch), params.frac_ensemble[frac_id])
                    print('%d episodes, Test Acc = %4.2f%% +- %4.2f%%' %(n_episodes, acc_mean, 1.96* acc_std/np.sqrt(n_episodes)))
                
                ########## (haven't modified) last record and post-process ##########
                torch.cuda.empty_cache()
                timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                # TODO afterward: compute this
#                 acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(n_episodes))
                frac_extra_records = []
                for frac_id in range(n_fracs):
                    # writing settings into csv
                    acc_mean = frac_acc_means[frac_id]
                    acc_std = frac_acc_stds[frac_id]
                    acc_mean_str = '%4.2f' % (acc_mean)
                    acc_std_str = '%4.2f' %(acc_std)
                    # record beyond params
                    extra_record = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str, 'epoch':load_epoch}
                    frac_extra_records.append(extra_record)
                
                if should_del_features:
                    del_features(params)
                end_time = datetime.datetime.now()
                print('exp_test() start at', start_time, ', end at', end_time, '.\n')
                print('exp_test() totally took:', end_time-start_time)
                
                return frac_extra_records, ep_task_data_each_frac

def frac_ensemble_str2var(frac_ensemble):
    if frac_ensemble.lower() == 'none':
        return None
    else:
        return float(frac_ensemble)

def get_all_feature_files(params):
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        pass
    else:
        checkpoint_dir = get_checkpoint_dir(params)
        feature_file = get_save_feature_filepath(params, checkpoint_dir, params.split)
        keyword = None
        if 'candidate' in feature_file:
            keyword = 'candidate'
        elif 'complement' in feature_file:
            keyword = 'complement'
        
        if keyword == None:
            all_feature_files = [feature_file] # only 1 model saves features
        else:
            all_feature_files = []
            for n in tqdm(range(params.n_test_candidates)):
                nth_feature_file = feature_file.replace(keyword, keyword+str(n+1))
                all_feature_files.append(nth_feature_file)
        
        return all_feature_files

def del_features(params):
    all_feature_files = get_all_feature_files(params)
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        pass
    else:
        checkpoint_dir = get_checkpoint_dir(params)
        feature_file = get_save_feature_filepath(params, checkpoint_dir, params.split)
        print('Deleting %d feature file(s): %s'%(len(all_feature_files), feature_file))
        for filename in all_feature_files:
            os.remove(filename)
        print('Finished deleting.')
        
#         files_tqdm = tqdm(all_feature_files)
#         for filename in files_tqdm:
#             files_tqdm.set_description('Deleting feature file:', filename)
#             os.remove(filename)


def record_to_csv(params, extra_record, csv_path):
    if os.path.isfile(csv_path):
        print('reading:', csv_path)
        df = pd.read_csv(csv_path)
        new_df = params2df(params, extra_record)
        out_df = pd.concat([df, new_df], axis=0, join='outer', sort=False)
    else:
        print('making file:', csv_path)
        out_df = params2df(params, extra_record)
        
    print('saving to:', csv_path)
    with open(csv_path, 'w') as f:
        out_df.to_csv(f, header=True, index=False)

def record_task_datas(params, task_datas):
    pass

if __name__ == '__main__':
    
    print('test.py start')
    
    params = parse_args('test')
    
    # TODO: modify params.split to change between base/val/novel
    # TODO: test_possible_params
    # get test result
    extra_record = exp_test(params=params, n_episodes=600)
    
    # TODO: params assign csv_name
    record_to_csv(params, extra_record, csv_path='./record/results.csv')
#     record_to_csv(params, extra_record, csv_path='./record/results_backup_'+extra_record['time']+'.csv')
    if params.csv_name is not None:
        record_to_csv(params, extra_record, csv_path='./record/'+params.csv_name)
