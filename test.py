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

from model_utils import get_few_shot_params, get_model

def exp_test(params, iter_num):
    print('exp_test() start.')
    
    set_random_seed(0) # successfully reproduce "normal" testing. 
    
    if params.gpu_id:
        set_gpu_id(params.gpu_id)
    
    acc_all = []

    model = get_model(params, 'test')
    
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    
    if params.gpu_id:
        model = model.cuda()
    else:
        model = to_device(model)

    checkpoint_dir = get_checkpoint_dir(params)
    
    #modelfile   = get_resume_file(checkpoint_dir)
    # load model
    print('loading from:',checkpoint_dir)
    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            if params.gpu_id is None:
                tmp = torch.load(modelfile)
            else: # TODO: figure out WTF is going on here
                map_location = 'cuda:0'
#                 gpu_str = 'cuda:' + '0'#str(params.gpu_id)
#                 map_location = {'cuda:1':gpu_str, 'cuda:0':gpu_str} # see here: https://hackmd.io/koKAo6kURn2YBqjoXXDhaw#RuntimeError-CUDA-error-invalid-device-ordinal
                tmp = torch.load(modelfile, map_location=map_location)
#                 tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
            load_epoch = int(tmp['epoch'])
    else: # if 'baseline' or 'baseline++'
        load_epoch = -1 # TODO: get load_epoch, first save 'epoch' in train.py

    # train/val/novel
    split = params.split
    
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        image_size = get_img_size(params)
        load_file = get_loadfile_path(params, split)

        datamgr         = SetDataManager(image_size, n_episode = iter_num, n_query = 15 , **few_shot_params)
        
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

    else: # not MAML
        # directly use extracted features
        feature_file = get_save_feature_filepath(params, checkpoint_dir, split)
        
        if 'candidate' in feature_file:
            feature_files = []
            candidate_cl_feature = [] # features of each class of each candidates
            print('Loading features of %s candidates into dictionaries...' %(params.n_test_candidates))
            for n in tqdm(range(params.n_test_candidates)):
                nth_feature_file = feature_file.replace('candidate','candidate'+str(n+1))
                feature_files.append(nth_feature_file)
                cl_feature = feat_loader.init_loader(nth_feature_file)
                candidate_cl_feature.append(cl_feature)
            
            
            ############### Sanity Check ###############
            # check len(cl_feature) of each candidate
#             print('Sanity Check...')
#             prev_cl_feature = candidate_cl_feature[0]
#             for nth_cl_feature in candidate_cl_feature:
#                 for cl in nth_cl_feature.keys():
#                     n_data_prev = len(prev_cl_feature[cl])
#                     n_data = len(nth_cl_feature[cl])
#                     if n_data_prev != n_data:
#                         print('n_data_prev & n_data NOT the same !!!!')
#                         print('cl:', cl, ', n_data:', n_data, ', n_data_prev:', n_data_prev)
            ############### Sanity Check ###############
            
            
            print('Evaluating...')
            # TODO: aggregate this and lower part of for loop, only cl_feature are different
            for i in tqdm(range(iter_num)):
                # TODO: fix data list? can only fix class list?
                acc = feature_evaluation(candidate_cl_feature, model, params=params, n_query=15, **few_shot_params)
                # TODO: draw something here ???
                acc_all.append(acc)
                
        else: # common setting (no candidate)
            cl_feature = feat_loader.init_loader(feature_file)
            cl_feature_single = [cl_feature]
            for i in tqdm(range(iter_num)):
                # TODO: fix data list? can only fix class list?
                acc = feature_evaluation(cl_feature_single, model, params=params, n_query=15, **few_shot_params)
                # TODO: draw something here ???
                acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('loaded from %d epoch model.' %(load_epoch))
        print('%d episodes, Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
    torch.cuda.empty_cache()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    
    # writing settings into csv
    acc_mean_str = '%4.2f' % (acc_mean)
    acc_std_str = '%4.2f' %(acc_std)
    # beyond params
    extra_record = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str, 'epoch':load_epoch}
    
    return extra_record

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

if __name__ == '__main__':
    
    print('test.py start')
    
    params = parse_args('test')
    
    # TODO: modify params.split to change between base/val/novel
    # TODO: test_possible_params
    # get test result
    extra_record = exp_test(params=params, iter_num=600)
    
    # TODO: params assign csv_name
    record_to_csv(params, extra_record, csv_path='./record/results.csv')
#     record_to_csv(params, extra_record, csv_path='./record/results_backup_'+extra_record['time']+'.csv')
    if params.csv_name is not None:
        record_to_csv(params, extra_record, csv_path='./record/'+params.csv_name)

# def record_txt(params, iter_num, acc_str):    
#     # writing settings into txt
#     if params.save_iter != -1:
#         split_str = params.split + "_" +str(params.save_iter)
#     else:
#         split_str = params.split
#     with open('./record/results.txt' , 'a') as f:
#         # this part should modify for every argument change
#         aug_str = '-aug' if params.train_aug else ''
#         aug_str += '-adapted' if params.adaptation else ''
#         aug_str += ('-Decoder' + params.recons_decoder+str(params.recons_lambda)) if params.recons_decoder else ''
#         aug_str += ('-'+params.aug_target+'('+params.aug_type+')') if params.aug_type else ''
        
#         if params.method in ['baseline', 'baseline++'] :
#             exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
#         else:
#             exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
#         acc_descr = '%d Test Acc = ' %(iter_num)
#         acc_descr += acc_str
#         f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_descr)  )
        
    
    