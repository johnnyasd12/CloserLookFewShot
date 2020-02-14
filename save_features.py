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
from model_utils import get_backbone_func, batchnorm_use_target_stats
from tqdm import tqdm

def save_features(feature_net, data_loader, outfile, params):
    
    n_candidates = 1 if params.n_test_candidates == None else params.n_test_candidates
    outfile_candidate = 'candidate' in outfile
    print('candidate in outfile', outfile_candidate)
    if (outfile_candidate)^(params.n_test_candidates != None):
        raise ValueError('outfile & params.n_test_candidates mismatch.')
        
    for n in range(n_candidates):

        if 'candidate' in outfile: # then dropout
            outfile = outfile.replace('candidate', 'candidate'+str(n))
            # TODO: dropout
            feature_net.sample_random_subnet()
        
        f = h5py.File(outfile, 'w')
        max_count = len(data_loader)*data_loader.batch_size
        all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
        all_feats=None
        count=0

        for i, (x,y) in enumerate(tqdm(data_loader)):

            if params.gpu_id:
                x = x.cuda()
            else:
                x = to_device(x)

            x_var = Variable(x)
            feats = feature_net(x_var)
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
            all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count+feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count

        f.close()


if __name__ == '__main__':
    params = parse_args('save_features')
    
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

    if params.aug_type is None:
        datamgr         = SimpleDataManager(image_size, batch_size = 64)
    else:
        datamgr         = AugSimpleDataManager(image_size, batch_size = 64, 
                                               aug_type=params.aug_type, aug_target='test-sample') # aug_target= 'all' or 'test-sample', NO 'test-batch'
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)

    backbone_func = get_backbone_func(params)
    backbone_net = backbone_func()
    
    if params.gpu_id:
        backbone_net = backbone_net.cuda()
        tmp = torch.load(modelfile, map_location='cuda:0')#+str(params.gpu_id))
    else:
        backbone_net = to_device(backbone_net)
        tmp = torch.load(modelfile)
    
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    backbone_net.load_state_dict(state)
    backbone_net.eval()
    if params.target_bn:
        print('switching batch_norm layers to train mode...')
        backbone_net.apply(batchnorm_use_target_stats)

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    print('saving features to:', outfile)
    save_features(backbone_net, data_loader, outfile, params)
