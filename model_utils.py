import backbone
from methods.baselinetrain import BaselineTrain, BaselineTrainMinGram
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet, ProtoNetAE, ProtoNetAE2, ProtoNetMinGram
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

from io_utils import model_dict, decoder_dict
import os
import configs

import sys
llvae_dir = configs.llvae_dir
sys.path.append(llvae_dir)
from datas import Omniglot
from nets import *
import LrLiVAE
LrLiVAE.DEBUG = configs.debug
from LrLiVAE import GMM_AE_GAN

import torch

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

def get_backbone_func(params):
    print('get_backbone_func() start...')
    
    # copy from get_model()
#     if params.dataset in ['omniglot', 'cross_char']:
# #         assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
#         assert 'Conv4' in params.model and not params.train_aug ,'omniglot/cross_char only support Conv4 without augmentation'
#         params.model = params.model.replace('Conv4', 'Conv4S') # because Conv4Drop should also be Conv4SDrop
#         if params.recons_decoder is not None:
#             if 'ConvS' not in params.recons_decoder:
#                 raise ValueError('omniglot / cross_char should use ConvS/HiddenConvS decoder.')
    
    # decide dropout settings
    dropout_p = params.dropout_p
    dropout_bid = params.dropout_block_id
    if hasattr(params, 'test_dropout_p'): # save_features or test
        if params.test_dropout_p is not None:
            dropout_p = params.test_dropout_p
            dropout_bid = params.test_dropout_bid
    
    if params.method in ['relationnet', 'relationnet_softmax']:
        b_func = {
            'Conv4':backbone.Conv4NP,
            'Conv6':backbone.Conv6NP,
            'Conv4S':backbone.Conv4SNP,
            'Conv4SThin2':backbone.Conv4SNPThin2,
            'Conv4SThin4':backbone.Conv4SNPThin4,
        }
#         if params.model == 'Conv4': 
#             backbone_func = lambda: backbone.Conv4NP(
#             dropout_p=dropout_p, dropout_block_id=dropout_bid
#             , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)
#         elif params.model == 'Conv6': 
#             backbone_func = lambda: backbone.Conv6NP(
#             dropout_p=dropout_p, dropout_block_id=dropout_bid
#             , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)
#         elif params.model == 'Conv4S': 
#             backbone_func = lambda: backbone.Conv4SNP(
#             dropout_p=dropout_p, dropout_block_id=dropout_bid
#             , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)
        if 'Conv' in params.model:
            backbone_func = lambda: b_func[params.model](
            dropout_p=dropout_p, dropout_block_id=dropout_bid
            , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)
        else: # ResNet
            backbone_func = lambda: model_dict[params.model](
            dropout_p=dropout_p, dropout_block_id=dropout_bid
            , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)

    else: # not RelationNet
        
        backbone_func = lambda: model_dict[params.model](
            dropout_p=dropout_p, dropout_block_id=dropout_bid
            , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)
#         backbone_func = lambda: model_dict[params.model](
#             dropout_p=params.dropout_p, dropout_block_id=params.dropout_block_id
#             , more_to_drop=params.more_to_drop, gram_bid = params.gram_bid)

    print('get_backbone_func() finished.')
    return backbone_func

def get_model(params, mode):
    '''
    Args:
        params: argparse params
        mode: (str), 'train', 'test'
    '''
    print('get_model() start...')
    few_shot_params_d = get_few_shot_params(params, None)
    few_shot_params = few_shot_params_d[mode]
    
    if params.dataset in ['omniglot', 'cross_char', 'cross_char_half']:
#         assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        assert 'Conv4' in params.model and not params.train_aug ,'omniglot/cross_char only support Conv4 without augmentation'
        params.model = params.model.replace('Conv4', 'Conv4S') # because Conv4Drop should also be Conv4SDrop
        if params.recons_decoder is not None:
            if 'ConvS' not in params.recons_decoder:
                raise ValueError('omniglot / cross_char should use ConvS/HiddenConvS decoder.')
    
    if params.method in ['baseline', 'baseline++'] and mode=='train':
        if params.dataset == 'omniglot': # 4112/688/1692
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char': # 1597/31/31
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char_half': # 1597/31/31
            assert params.num_classes >= 758, 'class number need to be larger than max label id in base class'
        if params.dataset == 'miniImagenet': # 64/16/20
            assert params.num_classes >= 64, 'class number need to be larger than max label id in base class'
        if params.dataset == 'CUB': # 100/50/50
            assert params.num_classes >= 100, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross': # 64+16+20/50/50
            assert params.num_classes >= 100, 'class number need to be larger than max label id in base class'
    
    if params.recons_decoder == None:
        print('params.recons_decoder == None')
        recons_decoder = None
    else:
        recons_decoder = decoder_dict[params.recons_decoder]
        print('recons_decoder:\n',recons_decoder)

    backbone_func = get_backbone_func(params)
    
    if 'baseline' in params.method:
        loss_types = {
            'baseline':'softmax', 
            'baseline++':'dist', 
        }
        loss_type = loss_types[params.method]
        
        if recons_decoder is None and params.min_gram is None: # default baseline/baseline++
            if mode == 'train':
                model = BaselineTrain(
                    model_func = backbone_func, loss_type = loss_type, 
                    num_class = params.num_classes, **few_shot_params)
            elif mode == 'test':
                model = BaselineFinetune(
                    model_func = backbone_func, loss_type = loss_type, 
                    **few_shot_params, finetune_dropout_p = params.finetune_dropout_p)
        else: # other settings for baseline
            if params.min_gram is not None:
                min_gram_params = {
                    'min_gram':params.min_gram, 
                    'lambda_gram':params.lambda_gram, 
                }
                if mode == 'train':
                    model = BaselineTrainMinGram(
                        model_func = backbone_func, loss_type = loss_type, 
                        num_class = params.num_classes, **few_shot_params, **min_gram_params)
                elif mode == 'test':
                    model = BaselineFinetune(
                        model_func = backbone_func, loss_type = loss_type, 
                        **few_shot_params, finetune_dropout_p = params.finetune_dropout_p)
#                     model = BaselineFinetuneMinGram(backbone_func, loss_type = loss_type, **few_shot_params, **min_gram_params)
            
    
    elif params.method == 'protonet':
        # default ProtoNet
        if recons_decoder is None and params.min_gram is None:
            model = ProtoNet( backbone_func, **few_shot_params )
        else: # other settings
            if params.min_gram is not None:
                min_gram_params = {
                    'min_gram':params.min_gram, 
                    'lambda_gram':params.lambda_gram, 
                }
                model = ProtoNetMinGram(backbone_func, **few_shot_params, **min_gram_params)

            if params.recons_decoder is not None:
                if 'Hidden' in params.recons_decoder:
                    if params.recons_decoder == 'HiddenConv': # 'HiddenConv', 'HiddenConvS'
                        model = ProtoNetAE2(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 2)
                    elif params.recons_decoder == 'HiddenConvS': # 'HiddenConv', 'HiddenConvS'
                        model = ProtoNetAE2(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 2, is_color=False)
                    elif params.recons_decoder == 'HiddenRes10':
                        model = ProtoNetAE2(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 6)
                    elif params.recons_decoder == 'HiddenRes18':
                        model = ProtoNetAE2(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, extract_layer = 8)
                else:
                    if 'ConvS' in params.recons_decoder:
                        model = ProtoNetAE(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, is_color=False)
                    else:
                        model = ProtoNetAE(backbone_func, **few_shot_params, recons_func=recons_decoder, lambda_d=params.recons_lambda, is_color=True)
    elif params.method == 'matchingnet':
        model           = MatchingNet( backbone_func, **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
#         if params.model == 'Conv4': 
#             feature_model = backbone.Conv4NP
#         elif params.model == 'Conv6': 
#             feature_model = backbone.Conv6NP
#         elif params.model == 'Conv4S': 
#             feature_model = backbone.Conv4SNP
#         else:
#             feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

        model           = RelationNet( backbone_func, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model           = MAML(  backbone_func, approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char', 'cross_char_half']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
        raise ValueError('Unexpected params.method: %s'%(params.method))
    
    print('get_model() finished.')
    return model


def batchnorm_use_target_stats(m):
    ''' only call this after common testing
    '''
#     print('switching batch_norm layers to train mode...')
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#         print(m.training)
        m.train()
#         print(m.training)


def restore_vaegan(dataset, vae_exp_name, vae_restore_step, is_training=False):
    experiment_name = vae_exp_name #'omn_noLatin_1114_0956'
    restore_step = vae_restore_step
    llvae_dir = configs.llvae_dir
    log_dir = os.path.join(llvae_dir, 'logs', experiment_name)
    model_dir = os.path.join(llvae_dir, 'models',experiment_name)
    print('model_dir:',model_dir)
    print('log_dir:',log_dir)
    
    print('initializing subnets of GMM_AE_GAN...')
    if dataset == 'omniglot' or dataset == 'cross_char':
        split = 'noLatin' if dataset=='cross_char' else 'train'
        datapath = './filelists/omniglot/hdf5'
        data = Omniglot(datapath=datapath, 
                        size=28, batch_size=32, 
                       is_tanh=True, flag='conv', split=split)

        generator = GeneratorMnist(size = data.size)
#         identity = IdentityMnist(data.y_dim, data.z_dim, size = data.size) # z_dim should be data.zc_dim ??
        identity = IdentityMnist(data.y_dim, data.zc_dim, size = data.size) # z_dim should be data.zc_dim ??
        attribute = AttributeMnist(data.z_dim, size = data.size)
        discriminator = DiscriminatorMnistSN(size=data.size)
#         discriminator = DiscriminatorMnistSNComb(size=data.size) # which to use?
        latent_discriminator = LatentDiscriminator(y_dim = data.y_dim)
    elif dataset == 'mnist':
        data = mnist(is_tanh=True)
        generator = GeneratorMnist(size = data.size)
#         identity = IdentityMnist(data.y_dim, data.z_dim, size = data.size) # z_dim should be data.zc_dim ??
        identity = IdentityMnist(data.y_dim, data.zc_dim, size = data.size) # z_dim should be data.zc_dim ??
        attribute = AttributeMnist(data.z_dim, size = data.size)
        discriminator = DiscriminatorMnistSN(size=data.size)
#         discriminator = DiscriminatorMnistSNComb(size=data.size) # which to use?
        latent_discriminator = LatentDiscriminator(y_dim = data.y_dim)
    
    else:
        raise ValueError('GMM_AE_GAN doesn\'t support dataset \'%s\' currently.' % (dataset))
    
    # load model
    print('initializing GMM_AE_GAN')
    vaegan = GMM_AE_GAN(
        generator, identity, attribute, discriminator, latent_discriminator, 
        data, is_training, log_dir=log_dir,
        model_dir=model_dir
    )
    print('done.')
    
    print('restoring GMM_VAE model...')
    vaegan.restore(restore_step)
    print('done.')
    return vaegan


def show_bn_detail(bn):
    print(bn)
    print('runnning mean:',bn.running_mean,'\nrunning var:',bn.running_var)
#     print('beta:',bn.beta, '\ngamma:',bn.gamma)
    print('beta:',bn.bias, '\ngamma:',bn.weight)

