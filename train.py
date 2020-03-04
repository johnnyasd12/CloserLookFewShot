import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone

from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet, ProtoNetAE, ProtoNetAE2
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
# from io_utils import model_dict, parse_args, get_resume_file, decoder_dict, get_checkpoint_dir
from io_utils import *
from my_utils import *
from model_utils import get_few_shot_params, get_model, restore_vaegan

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, record):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    result = {}
    max_acc = 0
    best_epoch = 0
    
    need_train_acc = True
    
    if params.patience is not None:
        stop_delta = 0.
        early_stopping = EarlyStopping(patience=params.patience, verbose=False, delta=stop_delta, mode='max')
    else:
        early_stopping = None

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        if need_train_acc:
            train_acc, train_loss = model.train_loop(epoch, base_loader,  optimizer, compute_acc=need_train_acc)
        else:
            train_loss = model.train_loop(epoch, base_loader,  optimizer, compute_acc=need_train_acc)
        model.eval()

        acc = model.test_loop( val_loader)
        record['train_loss'].append(train_loss)
        record['val_acc'].append(acc)
        
        if need_train_acc:
            record['train_acc'].append(train_acc)
            
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
#             print("best model! save...")
            max_acc = acc
            best_epoch = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            print("best model! save at:", outfile)
            torch.save({'record':record, 'epoch':epoch, 'state':model.state_dict()}, outfile)

        if params.save_freq != None:
            if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'record':record, 'epoch':epoch, 'state':model.state_dict()}, outfile)
        
        if early_stopping is not None:
            early_stopping(acc, model)
            if early_stopping.early_stop:
                print('EarlyStop: not improved more than %f after %d epoch. ' % (stop_delta, params.patience))
                break
        
    print('The best accuracy is',(str(max_acc)+'%'), 'at epoch', best_epoch)
    # TODO: print train_acc
    
    result['best_epoch'] = best_epoch
    
    result['train_loss_his'] = record['train_loss'].copy()
    result['train_acc_his'] = record['train_acc'].copy()
    result['val_acc_his'] = record['val_acc'].copy()
    
    result['train_loss'] = record['train_loss'][best_epoch]# avg train_acc of best epoch
    result['train_acc'] = record['train_acc'][best_epoch] # avg train_acc of best epoch
    result['val_acc'] = max_acc
    
    return model, result

def get_train_val_filename(params):
    # this part CANNOT share with save_features.py & test.py
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json'
    return base_file, val_file

def set_default_stop_epoch(params):
    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default

def get_train_val_loader(params):
    # to prevent circular import
    from data.datamgr import SimpleDataManager, SetDataManager, AugSetDataManager, VAESetDataManager
    
    image_size = get_img_size(params)
    base_file, val_file = get_train_val_filename(params)
    
    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        
    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

#         train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
#         test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        train_few_shot_params    = get_few_shot_params(params, 'train')
        test_few_shot_params     = get_few_shot_params(params, 'test')
        if params.vaegan_exp is not None:
            # TODO
            is_training = False
            vaegan = restore_vaegan(params.dataset, params.vaegan_exp, params.vaegan_step, is_training=is_training)
            
            # DDDDDDEEEEEEEEBBBBUUUUGG DEBUG
            if configs.debug:
                batch_x, batch_y = vaegan.data(32) # batch_size actually useless in omniglot & miniImagenet
                fig_x = vaegan.data.data2fig(batch_x[:16], nr=4, nc=4, 
                                             save_path='./debug/rec_samples/x_batch.png')
                rec_batch = vaegan.rec_samples(batch_x, lambda_zlogvar=params.zvar_lambda) # -1~1
                fig_rec = vaegan.data.data2fig(rec_batch[:16], nr=4, nc=4, 
                                              save_path='./debug/rec_samples/rec_batch.png')
                rec_single = vaegan.rec_samples(batch_x[:1], lambda_zlogvar=params.zvar_lambda) # -1~1
                fig_rec = vaegan.data.data2fig(rec_single[:1], nr=1, nc=1, 
                                              save_path='./debug/rec_samples/rec_single.png')
                
            base_datamgr            = VAESetDataManager(
                                        image_size, n_query=n_query, 
                                        vaegan_exp = params.vaegan_exp, 
                                        vaegan_step = params.vaegan_step, 
                                        vaegan_is_train = params.vaegan_is_train, 
                                        lambda_zlogvar=params.zvar_lambda, 
                                        fake_prob = params.fake_prob, 
                                        **train_few_shot_params)
            # train_val or val???
            val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
            
            
        elif params.aug_target is None:
            assert params.aug_type is None
            
            base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        else:
            aug_type = params.aug_type
            assert aug_type is not None
            base_datamgr            = AugSetDataManager(image_size, n_query = n_query, 
                                                        aug_type=aug_type, aug_target=params.aug_target, 
                                                        **train_few_shot_params)
            val_datamgr             = AugSetDataManager(image_size, n_query = n_query, 
                                                        aug_type=aug_type, aug_target='test-sample', 
                                                        **test_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, n_channel, w, h] tensor        
        
    else:
        raise ValueError('Unknown method')
    return base_loader, val_loader

def exp_train_val(params):
    start_time = get_time_now()
    print('exp_train_val() started at',start_time)
    np.random.seed(10)
    record = {
        'train_loss':[], 
        'val_acc':[], 
        'train_acc':[], 
    }

    if params.gpu_id:
        set_gpu_id(params.gpu_id)
    
    model = get_model(params, 'train')

    optimization = 'Adam'

    set_default_stop_epoch(params)

    base_loader, val_loader = get_train_val_loader(params)

    if params.gpu_id:
        model = model.cuda()
    else:
        model = to_device(model)

    params.checkpoint_dir = get_checkpoint_dir(params)
    
    if not os.path.isdir(params.checkpoint_dir):
        print('making directory:',params.checkpoint_dir)
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            if 'record' in list(tmp.keys()):
                record = tmp['record']
            model.load_state_dict(tmp['state'])
        else:
            raise ValueError('resume_file is None!!!')
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        # TODO: checkpoint_dir for resume haven't synchronize
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model, result = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params, record)
    
    torch.cuda.empty_cache()
    print('exp_train_val() start at', start_time, ', end at', get_time_now())
    
    return result

if __name__=='__main__':
    params = parse_args('train')
    result = exp_train_val(params)
