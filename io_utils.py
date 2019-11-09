import numpy as np
import os
import glob
import argparse
import backbone

import configs
import pandas as pd

from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet, ProtoNetAE, ProtoNetAE2
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

# embedding model architecture
model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 

# reconstruction decoder
decoder_dict = dict(
    Conv = backbone.DeConvNet(),
    ConvS = backbone.DeConvNetS(), 
    FC = backbone.DeFCNet(), 
    HiddenConv = backbone.DeConvNet2(), 
    HiddenConvS = backbone.DeConvNetS2(), 
    Res18 = backbone.DeResNet18(), 
    Res10 = backbone.DeResNet10(), 
    HiddenRes10 = backbone.DeResNet10_2(), 
)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--gpu_id'      , default=None, type=str, help='which gpu to use')
    
    # extra argument
    # assign image resize
    parser.add_argument('--image_size', default=None, type=int, help='the rescaled image size')
    # auxiliary reconstruction task
    parser.add_argument('--recons_decoder'   , default=None, choices=['FC','Conv','HiddenConv','Res18','Res10','HiddenRes10','ConvS','HiddenConvS'], help='reconstruction decoder')
    # coefficient of reconstruction loss
    parser.add_argument('--recons_lambda'   , default=0, type=float, help='lambda of reconstruction loss') # TODO: default=None? 0? will bug?
    parser.add_argument('--aug_type', default=None, choices=['rotate', 'bright', 'contrast', 'mix'], help='task augmentation mode') # TODO: rename to aug_mode
    parser.add_argument('--aug_target', default=None, choices=['batch', 'sample'], help='data augmentation by task or by sample')
        
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        
#         parser.add_argument('--test_aug_target', default=None, choices=['all', 'test-sample'], help='val data augmentation by sample or all')
        parser.add_argument('--patience'    , default=None, type=int, help='early stopping patience')
        
        
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
        
#         parser.add_argument('--test_aug_target', default=None, choices=['batch', 'sample'], help='test data augmentation by sample or batch')
        
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        
#         parser.add_argument('--test_aug_target', default=None, choices=['batch', 'sample'], help='test data augmentation by sample or batch')
        
    elif script == 'draw_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--reduce_mode', choices=['pca', 'pca-tsne', 'tsne'])
        parser.add_argument('--d_classes', default=5, type=int, help='number of classes should be draw')
        parser.add_argument('--d_samples', default=20, type=int, help='number of samples per class should be draw')
    
    else:
        raise ValueError('Unknown script')
    
    params = parser.parse_args()
    
    # sanity check
    if (params.aug_type==None)^(params.aug_target==None):
        raise ValueError('aug_type & aug_target not match.')
    if (params.recons_decoder==None)^(params.recons_lambda==0):
        raise ValueError('recons_decoder & recons_lambda not match. ')
    if script == 'save_features':
        if params.method in ['maml' , 'maml_approx']:
            raise ValueError('MAML does not support save_features')

    return params

def get_checkpoint_dir(params):
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.recons_decoder: # extra decoder
        checkpoint_dir += '_%sDecoder%s' %(params.recons_decoder,params.recons_lambda)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if params.aug_type is not None:
        checkpoint_dir += '_%s-%s' %(params.aug_type, params.aug_target)
    return checkpoint_dir

def get_few_shot_params(params, mode=None):
    '''
    :param mode: 'train', 'test'
    '''
    few_shot_params = {
        'train': dict(n_way = params.train_n_way, n_support = params.n_shot), 
        'test': dict(n_way = params.test_n_way, n_support = params.n_shot) 
    }
    if mode is None:
        return few_shot_params
    else:
        return few_shot_params[mode]

def get_model(params):
    train_few_shot_params    = get_few_shot_params(params, 'train')
    test_few_shot_params     = get_few_shot_params(params, 'test')
    
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'
        if params.recons_decoder is not None:
            if 'ConvS' not in params.recons_decoder:
                raise ValueError('omniglot / cross_char should use ConvS/HiddenConvS decoder.')
    
    if params.method in ['baseline', 'baseline++'] :
        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'
    
    if params.recons_decoder == None:
        print('params.recons_decoder == None')
        recons_decoder = None
    else:
        recons_decoder = decoder_dict[params.recons_decoder]
        print('recons_decoder:\n',recons_decoder)

    
    # not sure
    if params.method == 'baseline':
        model           = BaselineTrain( model_dict[params.model], params.num_classes)
    elif params.method == 'baseline++':
        model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

    if params.method == 'protonet':
        if recons_decoder is None:
            model = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif 'Hidden' in params.recons_decoder:
            if params.recons_decoder == 'HiddenConv': # 'HiddenConv', 'HiddenConvS'
                model = ProtoNetAE2(model_dict[params.model], **train_few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 2)
            elif params.recons_decoder == 'HiddenConvS': # 'HiddenConv', 'HiddenConvS'
                model = ProtoNetAE2(model_dict[params.model], **train_few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 2, is_color=False)
            elif params.recons_decoder == 'HiddenRes10':
                model = ProtoNetAE2(model_dict[params.model], **train_few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 6)
        else:
            model = ProtoNetAE(model_dict[params.model], **train_few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda) # WTFFFFFFFF lambda_d just 1
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
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

        model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    
    return model

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    print('get assigned file:', assign_file)
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        print('NO .tar file, get_resume_file failed. ')
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    print('get resume file with max epoch:', resume_file)
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        print('best file:', best_file)
        return best_file
    else:
        print('NOT found best file:',best_file,' , go get resume file')
        return get_resume_file(checkpoint_dir)

def params2df(params, extra_dict):
    params_dict = params.__dict__.copy()
    new_dict = {**params_dict, **extra_dict} if extra_dict is not None else params_dict
    for key,value in new_dict.items(): # make value to be list
        new_dict[key] = [value]
    new_df = pd.DataFrame.from_dict(new_dict)
    return new_df

if __name__ == '__main__':
    filename = 'test_exp.csv'
    df = pd.read_csv(filename)
    print('read_csv:\n', df.tail())
    
    params = parse_args('test')
    extra_dict = {'test_acc_mean':70, 'test_acc_std':0.68, 'time':'191013_193906'}
    new_df = params2df(params, extra_dict)
    out_df = pd.concat([df, new_df], axis=0, join='outer', sort=False)
    print('out_df\n', out_df.tail())
    with open(filename, 'w') as f:
        out_df.to_csv(f, header=True, index=False)
    
    