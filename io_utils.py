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
            ResNet101 = backbone.ResNet101, 
    
#             ResNet18Widen1 = backbone.ResNet18Widen1, 
            Conv4SFat2     = backbone.Conv4SFat2, 
            Conv4SThin2    = backbone.Conv4SThin2, 
            ResNet18Fat2   = backbone.ResNet18Fat2, 
            ResNet18Thin2  = backbone.ResNet18Thin2, 
#             Conv4Drop = backbone.Conv4Drop, 
#             Conv4SDrop = backbone.Conv4SDrop
) 

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

def parse_args(script, parse_str=None):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    
    # expmgr.py, train.py, save_features.py, test.py
    parser.add_argument('--dataset'     , default=None, choices=['CUB','miniImagenet','cross','omniglot','cross_char'])#, required=True)
    parser.add_argument('--model'       , default=None,      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default=None,   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') # this param also for validation. #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--gpu_id'      , default=None, type=str, help='which gpu to use')
    
    # extra argument
    parser.add_argument('--debug', action='store_true', help='whether is debugging. If True, then don\'t record.')
    
    ##### Custom Settings: train.py / save_features.py / test.py #####
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

    # domain CustomDropout
    parser.add_argument('--dropout_p', default=0, type=float, help='the domain CustomDropout probability. (1-dropout_p = keep_prob)')
    parser.add_argument('--dropout_block_id', default=3, type=int, help='the domain CustomDropout block id (all-drop if -1). Useless if dropout_p is 0.')
    parser.add_argument('--more_to_drop', default=None, type=str, choices=[None, 'double']) # None, 'double', 'by_rate'
    # minimize Gram Matrix
    parser.add_argument('--min_gram', default=None, type=str, choices=[None, 'l2', 'l1'], help='whether minimize the norm of Gram Matrix')
    parser.add_argument('--gram_bid', default=None, type=str, choices=[None, 'before_dropout', 'after_dropout', 0,1,2,3], help='which block to compute feature map Gram matrix. "dropout" to follow dropout_bid.')
    parser.add_argument('--lambda_gram', default=None, type=float, help='the coefficient of Gram Matrix loss.')
    
    if script == 'expmgr':
        pass
    elif script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=None, type=int, help='Save frequency (epoch)')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        
#         parser.add_argument('--test_aug_target', default=None, choices=['all', 'test-sample'], help='val data augmentation by sample or all')
        parser.add_argument('--patience'    , default=None, type=int, help='early stopping patience')
        
        
    elif script=='save_features' or script=='test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        
#         parser.add_argument('--test_aug_target', default=None, choices=['batch', 'sample'], help='test data augmentation by sample or batch')
        parser.add_argument('--target_bn', action='store_true', help='use target domain statistics to do batch normalization.')
        # CustomDropout
        parser.add_argument('--n_test_candidates', default=None, type=int, help='the number of dropout subnet candidates.')
        parser.add_argument('--sample_strategy', default='none', type=str, choices=['none', 'complement'])
        
        # test-only drop neurons
        parser.add_argument('--test_dropout_p', default=None, type=float, help='the test-time sampling(?) dropout rate, if None then default is dropout_p.')
        parser.add_argument('--test_dropout_bid', default=None, type=int, help='the test-time sampling(?) dropout block id (all-drop if -1), if None then dropout_p should be also None.')
        parser.add_argument('--finetune_dropout_p', default=None, type=float, help='the dropout rate when finetuning output layer, only affect when method is baseline/baseline++.')
        
        ############ test.py ########## but i think save_features.py is okay
#         if script == 'test': # can also parse in save_features.py?? i think no effect is ok
        parser.add_argument('--csv_name'       , default=None, type=str, help='extra record csv file name.')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        # CustomDropout parameter
        parser.add_argument('--frac_ensemble', default=None, type=float, help='the final fraction of dropout subnets ensemble. (default only 1 subnet, no ensemble)')
        parser.add_argument('--candidate_metric', default='acc', choices=['acc', 'loss'], type=str, help='To choose the ensemble subnets, according to  which metric of sub-validation set. (if None then "acc")')
        parser.add_argument('--ensemble_strategy', default='vote', choices=['vote', 'avg_prob'], type=str, help='How to get the prediction of networks ensemble, only available when argument "frac_ensemble" is assigned.') # originally default None, but causes BUGGGGGG so modified to 'vote'. 
        
        
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
    
    if parse_str == None:
        params = parser.parse_args()
    else:
        params = parser.parse_args(parse_str)
    
    # sanity check
#     if params.dropout_block_id is not None and params.dropout_block_id != 'all':
#         params.dropout_block_id = int(params.dropout_block_id)
        
    if script=='save_features' or script=='test':
        
#         if params.test_dropout_bid is not None and params.test_dropout_bid != 'all':
#             params.test_dropout_bid = int(params.test_dropout_bid)

        if params.finetune_dropout_p is not None:
            if params.method not in ['baseline', 'baseline++']:
                raise ValueError('finetune_dropout_p and method not match.')
        
        if (params.test_dropout_p is None) ^ (params.test_dropout_bid is None):
            raise ValueError('test_dropout_p and test_dropout_bid not match.')
        if params.test_dropout_p is not None and params.n_test_candidates is None:
            raise ValueError('test_dropout_p and n_test_candidates not match.')
            
        if params.n_test_candidates is not None: # both should be True or False
            
            if False:
                if params.dropout_p == 0:
                    raise ValueError('dropout_p and n_test_candidates not match.')
                if params.test_dropout_p is None:
                    raise ValueError('test_dropout_p and n_test_candidates not match.')
            # should be like this, but why above code can pass when doing experiments before???
            if params.dropout_p == 0 and params.test_dropout_p is None:
                raise ValueError('dropout_p/test_dropout_p and n_test_candidates not match.')
            
            if params.method in ['baseline', 'baseline++']:
                if params.n_test_candidates > 10:
                    raise ValueError('too many test candidates for baseline.')
                if params.frac_ensemble != 1:
                    raise ValueError('frac_ensemble for baseline methods should be 1.')
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
    if params.debug:
        checkpoint_dir = '%s/debug-checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.recons_decoder: # extra decoder
        checkpoint_dir += '_%sDecoder%s' %(params.recons_decoder,params.recons_lambda)
    if params.train_aug:
        checkpoint_dir += '_aug'
    # meta-learning methods
    if not params.method  in ['baseline', 'baseline++']: 
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    # special augmentation experiments
    if params.aug_type is not None:
        checkpoint_dir += '_%s-%s' %(params.aug_type, params.aug_target)
    # vaegan experiments
    if params.vaegan_exp:
        is_train_str = '_is-train' if params.vaegan_is_train else ''
        checkpoint_dir += '/%s-%s/lamb-var%s_fake-prob%s' %(params.vaegan_exp, params.vaegan_step, 
                                        params.zvar_lambda, params.fake_prob)
        checkpoint_dir += is_train_str
    # target_bn_stats??? NONONONO should not here, becuz will affect get model file
    
    # custom dropout experiments & more_to_drop settings
    if params.dropout_p != 0:
        checkpoint_dir += '_dropout%s' % (params.dropout_p)
        checkpoint_dir += '_block%s' % (params.dropout_block_id)
        if params.more_to_drop == 'double':
            checkpoint_dir += 'double-dim'
        if params.min_gram != None:
#             checkpoint_dir += '_min-gram-%s-lambda%s' % (params.min_gram, params.lambda_gram) # min-gram-l2, min-gram-l1
            gram_bid = params.gram_bid
            if isinstance(gram_bid, str):
                gram_bid = gram_bid.replace('_', '-') # before_dropout -> before-dropout
            checkpoint_dir += '_min-gram-%s-lambda%s%s' % (params.min_gram, params.lambda_gram, gram_bid) # min-gram-l2, min-gram-l1
    else: # dropout_p == 0
        if params.more_to_drop == 'double':
            checkpoint_dir += '_block%sdouble-dim'%(params.dropout_block_id)
    
    return checkpoint_dir

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    print('get assigned file:', assign_file)
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        print('NO .tar file, get_resume_file() failed. ')
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    print('get resume model file with max epoch:', resume_file)
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        print('best model file:', best_file)
        return best_file
    else:
        print('NOT found best model file:',best_file,' , go get resume model file')
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
    
    # CustomDropout
    # checkpoint_dir already has dropout_p information
    # should save_feature several times on different candidates
    if params.n_test_candidates == None:
        dropout_candidates_str = ''
    elif params.sample_strategy == 'none':
        dropout_candidates_str = '_candidate'
    elif params.sample_strategy == 'complement':
        dropout_candidates_str = '_complement'
#     dropout_candidates_str = '' if params.n_test_candidates == None else '_candidate' 
    # should add candidate number in save_features.py
    
    extra_str = target_bn_str + dropout_candidates_str
    outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str + extra_str + ".hdf5")
    
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
    
    