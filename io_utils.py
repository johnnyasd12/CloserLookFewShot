import numpy as np
import os
import glob
import argparse
import backbone

import configs
import pandas as pd

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
    parser.add_argument('--dataset'     , choices=['CUB','miniImagenet','cross','omniglot','cross_char'], required=True)
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
    # GMM_VAE_GAN augmentation
    parser.add_argument('--vaegan_exp', default=None, type=str, help='the GMM_VAE_GAN experiment name')
    parser.add_argument('--vaegan_step', default=None, type=int, help='the GMM_VAE_GAN restore step')
    parser.add_argument('--zvar_lambda', default=None, type=float, help='the GMM_VAE_GAN zlogvar_lambda')
    parser.add_argument('--fake_prob', default=None, type=float, help='the probability to replace real image with GMM_VAE_GAN generated image. ')
    parser.add_argument('--vaegan_is_train', action='store_true', help='whether the vaegan is_training==True.')
    

    parser.add_argument('--debug', action='store_true', help='whether is debugging.')
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
        parser.add_argument('--target_bn', action='store_true', help='use target domain statistics to do batch normalization.')
        
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        
#         parser.add_argument('--test_aug_target', default=None, choices=['batch', 'sample'], help='test data augmentation by sample or batch')
        parser.add_argument('--target_bn', action='store_true', help='use target domain statistics to do batch normalization.')
        
    elif script == 'draw_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--reduce_mode', choices=['pca', 'pca-tsne', 'tsne'])
        parser.add_argument('--d_classes', default=5, type=int, help='number of classes should be draw')
        parser.add_argument('--d_samples', default=20, type=int, help='number of samples per class should be draw')
    elif script == 'make_llvae_dataset':
        parser.add_argument('--dataset', choices=['omniglot', 'CUB', 'miniImagenet', 'emnist'], help='dataset you want to reconstruct by Lr-LiVAE.')
        parser.add_argument('--mode', choices=['all', 'trian', 'val', 'test', 'noLatin'], help='data split.')
        parser.add_argument('--batch_size', default=32, type=int, help='batch size when generating reconstructed samples.')
        parser.add_argument('--is_training', action='restore_true', help='whether the gmm_vae_gan set as training mode.')
        parser.add_argument('--gen_mode', default='rec', choices=['rec', 'gen'])
#         parser.add_argument('--vae_exp_name', )
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
    if params.vaegan_exp:
        is_train_str = '_is-train' if params.vaegan_is_train else ''
        checkpoint_dir += '/%s-%s/lamb-var%s_fake-prob%s' %(params.vaegan_exp, params.vaegan_step, 
                                        params.zvar_lambda, params.fake_prob)
        checkpoint_dir += is_train_str
    # TODO: target_bn_stats??? NONONONO should not here, becuz will affect get model file
    
    return checkpoint_dir

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


def get_img_size(params):
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28 if params.image_size is None else params.image_size
        else:
            image_size = 84 if params.image_size is None else params.image_size
    else:
        image_size = 224 # if params.image_size is None else params.image_size
    return image_size

def get_loadfile_path(params, split):
#     if 'Conv' in params.model:
#         if params.dataset in ['omniglot', 'cross_char']:
#             image_size = 28
#         else:
#             image_size = 84 
#     else:
#         image_size = 224
    
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
    return loadfile

def get_save_feature_filepath(params, checkpoint_dir, split):
    # TODO: split is actually params.split, waiting for refactoring
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    
    # TODO: target_bn_stats
    target_bn_str = '_target-bn' if params.target_bn else ''
    
    outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str + target_bn_str + ".hdf5")
    
#     if params.save_iter != -1:
#         outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
#     else:
#         outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")
    
    return outfile


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
    
    