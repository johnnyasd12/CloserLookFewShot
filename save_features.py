import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager, AugSimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
# from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file, get_checkpoint_dir
from io_utils import *

from my_utils import *
from my_utils import set_random_seed
from model_utils import get_backbone_func, batchnorm_use_target_stats
from tqdm import tqdm
import datetime

def exp_save_features(params):
    start_time = datetime.datetime.now()
    print('exp_save_features() started at',start_time)
    
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'
    if params.gpu_id:
        set_gpu_id(params.gpu_id)
    
    image_size = get_img_size(params)
    
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    split = params.split

    loadfile = get_loadfile_path(params, split)

    checkpoint_dir = get_checkpoint_dir(params)
    print('save_features.py checkpoint_dir:', checkpoint_dir)
    
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
#    elif params.method in ['baseline', 'baseline++'] :
#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)

    outfile = get_save_feature_filepath(params, checkpoint_dir, split) # string type
    print('outfile: %s'%(outfile))

    if params.aug_type is None:
        datamgr         = SimpleDataManager(image_size, batch_size = 64)
    else:
        datamgr         = AugSimpleDataManager(image_size, batch_size = 64, 
                                               aug_type=params.aug_type, aug_target='test-sample') # aug_target= 'all' or 'test-sample', NO 'test-batch'
    data_loader      = datamgr.get_data_loader(loadfile, aug = False, shuffle=False, return_path=True)

    backbone_func = get_backbone_func(params)
    backbone_net = backbone_func()
    
    if params.gpu_id:
        backbone_net = backbone_net.cuda()
        tmp = torch.load(modelfile, map_location='cuda:0')#+str(params.gpu_id))
    else:
        backbone_net = to_device(backbone_net)
        tmp = torch.load(modelfile)
    
    state = tmp['state']
    load_epoch = tmp['epoch'] if 'epoch' in tmp else -1
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    print('Loading %s epoch state_dict into backbone...'%(load_epoch))
    backbone_net.load_state_dict(state)
    print('Finished Loading.')
    backbone_net.eval()
    if params.target_bn:
        print('switching batch_norm layers to train mode...')
        backbone_net.apply(batchnorm_use_target_stats)

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
#     print('saving features to:', outfile)
    save_features(backbone_net, data_loader, outfile, params)
    
    end_time = datetime.datetime.now()
    print('exp_save_features() start at', start_time, ', end at', end_time, '.\n')
    print('exp_save_features() totally took:', end_time-start_time)

def save_features(feature_net, data_loader, outfile, params):
    
    set_random_seed(0)
    
    n_candidates = 1 if params.n_test_candidates == None else params.n_test_candidates
    outfile_candidate = 'candidate' in outfile or 'complement' in outfile
    print('Whether candidate in outfile:', outfile_candidate)
    if (outfile_candidate)^(params.n_test_candidates != None):
        raise ValueError('outfile & params.n_test_candidates mismatch.')
    
    for n in range(n_candidates):

        if 'candidate' in outfile: # then dropout
            assert params.sample_strategy == 'none'
            outfile_n = outfile.replace('candidate', 'candidate'+str(n+1))
            print(outfile_n, ': procedure start...')
            feature_net.sample_random_subnet()
        elif 'complement' in outfile: # then use complementary-sample strategy
            assert params.sample_strategy == 'complement'
            assert params.dropout_p <= 0.5
            outfile_n = outfile.replace('complement', 'complement'+str(n+1))
            print(outfile_n, ': procedure start...')
            n_combinations = int(1//params.dropout_p) # e.g. 1//0.33 = 3
            complement_id = n//n_combinations # e.g. 13//3 = 4
            complement_remainder = n%n_combinations # e.g. 13%3 = 1
            if complement_remainder == 0:
                dropout_layers = feature_net.active_dropout_ls
                assert len(dropout_layers)==1
                mask_combs = []
                for d_layer in dropout_layers:
                    mask_comb = d_layer.get_mask_comb() # len(mask_combs) = n_combinations
                    mask_combs.append(mask_comb)
            for d_layer, mask_comb in list(zip(dropout_layers, mask_combs)):
                mask = mask_comb[complement_remainder]
#                 d_layer.set_eval_mask(mask)
                d_layer.eval_mask = mask
        else:
            outfile_n = outfile
        
        f = h5py.File(outfile_n, 'w')
        max_count = len(data_loader.dataset)
#         max_count_ori = len(data_loader)*data_loader.batch_size # SHOULD be calculated by dataset not dataloader
        all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
        all_feats=None
        all_paths=None
        count=0
        
        for i, datas in enumerate(tqdm(data_loader)):
            paths, x, y = datas
            if params.gpu_id:
                x = x.cuda()
            else:
                x = to_device(x)

            x_var = Variable(x)
            feats = feature_net(x_var)
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
                all_paths = f.create_dataset('all_paths', [max_count,], dtype='S10')
            all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count+feats.size(0)] = y.cpu().numpy()
            all_paths[count:count+feats.size(0)] = np.string_(paths) # BUGFIX: No conversion path for dtype: dtype('<U81')
            count = count + feats.size(0)

        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count
        
        print('Finish saving features to:', outfile_n)
        f.close()
        
if __name__ == '__main__':
    params = parse_args('save_features')
    exp_save_features(params)
    

