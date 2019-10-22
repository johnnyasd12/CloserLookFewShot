import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
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
import pandas as pd

def get_img_settings(params, split):
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224
    
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split +'.json'
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
        else:
            loadfile  = configs.data_dir['emnist'] + split +'.json' 
    else: 
        loadfile    = configs.data_dir[params.dataset] + split + '.json'
    return image_size, loadfile

if __name__ == '__main__':
    print('test.py start')
    params = parse_args('test')

    if params.gpu_id:
        set_gpu_id(params.gpu_id)
    get_model_func = True
    
    acc_all = []

    iter_num = 600

    if get_model_func:
        model = get_model(params)
    else:
        if params.recons_decoder == None:
            print('params.recons_decoder == None')
            recons_decoder = None
        else:
            recons_decoder = decoder_dict[params.recons_decoder]
            print('recons_decoder:\n',recons_decoder)
    
#     few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) # BUGFIX: decoder ?
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    
    # ======== set meta-learning method and backbone ========
    if get_model_func:
        pass
    else:
        if params.dataset in ['omniglot', 'cross_char']:
            assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
            params.model = 'Conv4S'

        if params.method == 'baseline':
            model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
        elif params.method == 'baseline++':
            model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
        elif params.method == 'protonet':
            if recons_decoder is None:
                model = ProtoNet( model_dict[params.model], **few_shot_params )
            elif 'Hidden' in params.recons_decoder:
                model = ProtoNetAE2(model_dict[params.model], **few_shot_params, recons_func = recons_decoder, lambda_d=params.recons_lambda)
            else:
                model = ProtoNetAE(model_dict[params.model], **few_shot_params, recons_func = recons_decoder, lambda_d=params.recons_lambda)
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S': 
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        else:
            raise ValueError('Unknown method')

    if params.gpu_id:
        device = torch.device('cuda:'+str(params.gpu_id))
    else:
        device = None
#     model = model.cuda()
    if device is None:
        model = to_device(model)
    else:
        model = model.cuda()

    # set save directory
#     checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    
#     if params.recons_decoder:
#         checkpoint_dir += '_%sDecoder%s' %(params.recons_decoder, params.recons_lambda)
#     if params.train_aug:
#         checkpoint_dir += '_aug'
#     if not params.method in ['baseline', 'baseline++'] :
#         checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    checkpoint_dir = get_checkpoint_dir(params)
    
    #modelfile   = get_resume_file(checkpoint_dir)
    # load model file ???
    print('loading from:',checkpoint_dir)
    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            if params.gpu_id is None:
                tmp = torch.load(modelfile)
            else:
                tmp = torch.load(modelfile, map_location='cuda:0')#+str(params.gpu_id))
            model.load_state_dict(tmp['state'])

    # train/val/novel
    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    
    
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        image_size, load_file = get_img_settings(params, split)
#         if 'Conv' in params.model:
#             if params.dataset in ['omniglot', 'cross_char']:
#                 image_size = 28
#             else:
#                 image_size = 84 
#         else:
#             image_size = 224

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)
        
#         if params.dataset == 'cross':
#             if split == 'base':
#                 loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
#             else:
#                 loadfile   = configs.data_dir['CUB'] + split +'.json'
#         elif params.dataset == 'cross_char':
#             if split == 'base':
#                 loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
#             else:
#                 loadfile  = configs.data_dir['emnist'] + split +'.json' 
#         else: 
#             loadfile    = configs.data_dir[params.dataset] + split + '.json'

        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

    else: # not MAML
        # directly use extracted features
        test_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes # so the variable name "test_file" is proper?
        cl_data_file = feat_loader.init_loader(test_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            # TODO: draw something here ???
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
    
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    
    # writing settings into csv
    acc_mean_str = '%4.2f' % (acc_mean)
    acc_std_str = '%4.2f' %(acc_std)
    extra_dict = {'time':timestamp, 'acc_mean':acc_mean_str, 'acc_std':acc_std_str}
    
    csv_path = './record/results.csv'
    csv_backup_path = './record/results_backup_'+timestamp+'.csv'
    print('reading:', csv_path)
    df = pd.read_csv(csv_path)
    new_df = params2df(params, extra_dict)
    out_df = pd.concat([df, new_df], axis=0, join='outer', sort=False)
    print('saving to:', csv_backup_path)
    with open(csv_backup_path, 'w') as f:
        out_df.to_csv(f, header=True, index=False)
    print('saving to:', csv_path)
    with open(csv_path, 'w') as f:
        out_df.to_csv(f, header=True, index=False)
    
    # writing settings into txt
    with open('./record/results.txt' , 'a') as f:
        # this part should modify for every argument change
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        aug_str += ('-Decoder' + params.recons_decoder+str(params.recons_lambda)) if params.recons_decoder else ''
        aug_str += ('-'+params.aug_target+'('+params.aug_type+')') if params.aug_type else ''
        
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
        acc_descr = '%d Test Acc = ' %(iter_num)
        acc_descr += acc_str
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_descr)  )
        
    # use SetDataManager to transform original image???
    # I wrote this ???
    '''
    print('='*10 + 'start ploting' + '='*10)
    image_size, loadfile = get_img_settings(params, split)
    datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)
    novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
    model.eval()
    acc_mean = model.test_loop( novel_loader)
    '''
    
    torch.cuda.empty_cache()