# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, HDF5Dataset, AugSetDataset, AugSimpleDataset, VAESetDataset
from abc import abstractmethod

import torchvision.transforms.functional as TF
import random

# for LrLiVAE
import configs
import sys
llvae_dir = configs.llvae_dir
sys.path.append(llvae_dir)
from LrLiVAE import GMM_AE_GAN
from nets import *
from datas import *

from my_utils import describe

from matplotlib import pyplot as plt

import configs

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale': # Scale is actually Resize
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
    def get_simple_transform(self, aug = False): # to make hdf5 file
        if aug:
            transform_list = ['RandomSizedCrop']
        else:
            transform_list = ['Scale','CenterCrop']
        transform_list += ['ToTensor']
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
    def get_crop_transform(self, aug = False): # same as get_simple_transform, but no ToTensor operation
        if aug:
            transform_list = ['RandomSizedCrop']
        else:
            transform_list = ['Scale','CenterCrop']
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
    def get_hdf5_transform(self, aug = False, inputs='pil'): 
        # simple_transform(-toTensor) + this = composed_transform
        # crop_transform + this = composed_transform
        if inputs == 'pil':
            transform_list = []
        elif inputs == 'tensor':
            transform_list = ['ToPILImage']
        else:
            raise 'wrong inputs mode.'
        
        if aug:
            transform_list += ['ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list += ['ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
    def get_vae_transform(self, vaegan, lambda_zlogvar, fake_prob):
        def b4_vae(img):
            '''
            :param img: numpy array, shape=(w,h,1), range=(0~1)
            :return: ndarray, shape=(1,w,h,1), range=(-1~1) if is_tanh else (0~1)
            '''
            if vaegan.data.is_tanh: # if should input -1~1 to gmmvaegan
                img = img*2 - 1
            img = np.transpose(img, axes=(1,2,0)) # 3,h,w -> h,w,3
            img = img[np.newaxis,:,:,0:1] # -> 1,h,w,1
            return img
        
        def after_vae(img):
            img = np.repeat(img, repeats=3, axis=3)[0] # 1,h,w,1 -> h,w,3
            img = np.transpose(img, axes=(2,0,1)) # h,w,3 -> 3,h,w
            if vaegan.data.is_tanh: # if gmmvaegan output is -1~1 then rescale to 0~1
                img = img/2 + 0.5
            return img
            
        def wrapper(img):
            rand_num = np.random.random()
            if rand_num <= fake_prob:

                img = img.cpu().numpy()
    #             print('get_vae_transform/img type,shape:', type(img), img.shape)
    #             print('get_vae_transform/img:', img.min(), '~', img.max())
                img = b4_vae(img) # 1,h,w,1
    #             print('get_vae_transform/ after b4vae img type,shape:', type(img), img.shape)
    #             print('get_vae_transform/ after b4vae img:', img.min(), '~', img.max())

                rec_img = vaegan.rec_samples(img, lambda_zlogvar=lambda_zlogvar) # 1,h,w,1
    #             print('get_vae_transform/ after recon img type,shape:', type(img), img.shape)
    #             print('get_vae_transform/ after recon img:', img.min(), '~', img.max())

                # TODO: DDDDDEEEEBBBBUUUUUGGGGG
                if configs.debug:
                    idx = str(np.random.randint(3))
                    dst_dir = './debug/rec_samples'
                    draw_img = np.repeat(img, repeats=3, axis=3) # 1,h,w,3
                    draw_rec_img = np.repeat(rec_img,repeats=3, axis=3) # 1,h,w,3
                    print('\n')
    #                 describe(rec_img, 'datamgr/rec_img')
    #                 print('draw_img.shape:', draw_img.shape)
    #                 print('draw_rec_img.shape:', draw_rec_img.shape)
    #                 print('vaegan.is_training:', vaegan.is_training)
    #                 print('vaegan.data.is_tanh:', vaegan.data.is_tanh)
                    fig = vaegan.data.data2fig(draw_img[:,:,:,:1], nr=1, nc=1, 
                                               save_path=dst_dir+'/%sfig.png'%(idx)) # this is fine
                    plt.close(fig)
                    fig_rec = vaegan.data.data2fig(draw_rec_img[:,:,:,:1], nr=1, nc=1, 
                                                   save_path=dst_dir+'/%sfig_rec.png'%(idx))
                    plt.close(fig_rec)
    #                 vaegan.data.sample2fig2jpg(draw_img[0], dst_dir, '%ssample.jpg'%(idx))
    #                 vaegan.data.sample2fig2jpg(draw_rec_img[0], dst_dir, '%ssample_rec.jpg'%(idx))
    #                 vaegan.check_weights(name='GeneratorMnist/Conv2d_transpose_2/weights:0')
    #                 if idx == '0':
    #                     raise 'just stop mannnnn'
                final_img = after_vae(rec_img)
    #             print('get_vae_transform/ after post_vae img type,shape:', type(img), img.shape)
    #             print('get_vae_transform/ after post_vae img:', img.min(), '~', img.max())
                final_img = torch.from_numpy(final_img).float()
            else:
                final_img = img
            return final_img
        return wrapper
    
    def get_aug_transform(self, aug_type, aug_target):
        '''
        :param aug_type: str, 'rotate', 'bright', 'contrast', 'mix'
        :param aug_target: str, 'sample', 'batch', 'test-sample', 'test-batch'
        '''
        def get_random_rotate_angle(abs_angle_range):
            '''
            :param abs_angle_range: tuple
            '''
            # only one side
#             angle = random.randint(*abs_angle_range)*(-1)#**random.randint(0,1)
            angle = random.randint(*abs_angle_range)#**random.randint(0,1)
            return angle
        
        def get_random_factor(abs_perturb_range):
            '''
            :param abs_perturb_range: tuple
            '''
            # only one side
            lower, upper = abs_perturb_range
            assert lower <= upper
            perturb = (lower + (upper-lower)*random.random())#*(-1)#**random.randint(0,1)
            factor = 1 + perturb
            return factor
        

        aug_params = {}
        aug_params['rotate'] = {
#             'train_range':(0, 20), # should +/-
#             'test_range':(15, 25), # should +/-
            'train_range':(-15, -10), # should +/-
            'test_range':(10, 15), # should +/-
            'angle':None
        }
        aug_params['bright'] = {
#             'train_range':(0, 0.3), # should 1 +/- range
#             'test_range':(0.25, 0.5), # should 1 +/- range
            'train_range':(-0.6, -0.2), # should 1 +/- range
            'test_range':(0.3, 0.5), # should 1 +/- range
            'factor':None
        }
        aug_params['contrast'] = {
#             'train_range':(0, 0.3), # should 1 +/- range
#             'test_range':(0.25, 0.5), # should 1 +/- range
            'train_range':(0, 0.3), # should 1 +/- range
            'test_range':(0.25, 0.5), # should 1 +/- range
            'factor':None
        }

        # Deciding transform parameters
        if aug_target == 'batch': # random select for every batch
            aug_params['rotate']['angle'] = get_random_rotate_angle(aug_params['rotate']['train_range'])
            aug_params['bright']['factor'] = get_random_factor(aug_params['bright']['train_range'])
            aug_params['contrast']['factor'] = get_random_factor(aug_params['contrast']['train_range'])

        elif aug_target == 'test-batch': # random select for every batch
            aug_params['rotate']['angle'] = get_random_rotate_angle(aug_params['rotate']['test_range'])
            aug_params['bright']['factor'] = get_random_factor(aug_params['bright']['test_range'])
            aug_params['contrast']['factor'] = get_random_factor(aug_params['contrast']['test_range'])
        
        def wrapper(img):

            nonlocal aug_params

            if aug_target == 'batch' or aug_target == 'test-batch':
                pass # nothing to do

            # Deciding transform parameters
            elif aug_target == 'sample':
                aug_params['rotate']['angle'] = get_random_rotate_angle(aug_params['rotate']['train_range'])
                aug_params['bright']['factor'] = get_random_factor(aug_params['bright']['train_range'])
                aug_params['contrast']['factor'] = get_random_factor(aug_params['contrast']['train_range'])

            elif aug_target == 'test-sample':
                aug_params['rotate']['angle'] = get_random_rotate_angle(aug_params['rotate']['test_range'])
                aug_params['bright']['factor'] = get_random_factor(aug_params['bright']['test_range'])
                aug_params['contrast']['factor'] = get_random_factor(aug_params['contrast']['test_range'])

            else:
                raise ValueError('Invalid aug_target: %s' %(aug_target))

            # Process the image
            if aug_type == 'rotate':
                img = TF.rotate(img, aug_params['rotate']['angle'])
#                 print('rotate angle:', aug_params['rotate']['angle'])

            elif aug_type == 'bright':
                img = TF.adjust_brightness(img, aug_params['bright']['factor'])
#                 print('bright factor:', aug_params['bright']['factor'])

            elif aug_type == 'contrast':
                img = TF.adjust_contrast(img, aug_params['contrast']['factor'])
#                 print('contrast factor:', aug_params['contrast']['factor'])

            elif aug_type == 'mix':
                img = TF.adjust_contrast(img, aug_params['contrast']['factor'])
                img = TF.adjust_brightness(img, aug_params['bright']['factor'])
                img = TF.rotate(img, aug_params['rotate']['angle'])

            else:
                raise ValueError('Invalid aug_type: %s' %(aug_type))
            
            return img
        
        return wrapper

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class ResizeDataManager(DataManager): # for making hdf5 file
    def __init__(self, image_size, batch_size=1):
        super(ResizeDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        
    def get_data_loader(self, data_file, aug):
        transform = self.trans_loader.get_simple_transform(aug=aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = False) # pin_memory for fast load to GPU, but i don't need it
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class HDF5DataManager(DataManager):
    def __init__(self, image_size, batch_size, recons_func = None):
        super(HDF5DataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        
    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_hdf5_transform(aug)
        dataset = HDF5Dataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True) # TODO: tune pin_memory
#         data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True) # BUGFIX: set num_workers=0 to fix ConnectionResetError: [Errno 104] Connection reset by peer?
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class OrderedDataManager(DataManager):
    def __init__(self, image_size, batch_size, aug):
        super(OrderedDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.aug = aug
        
    def get_data_loader(self, data_file):
#         transform = transforms.ToTensor()
        transform = self.trans_loader.get_simple_transform(aug=self.aug)
        dataset = SimpleDataset(data_file, transform=transform, return_path=True)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True) # not sure if should be True when input to a TensorFlow model
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, recons_func = None):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class AugSimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, aug_type, aug_target, recons_func = None):
        # only support all the same augmentation for now
        super(AugSimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        
        self.aug_type = aug_type
        assert aug_target == 'all' or aug_target == 'test-sample'
        self.aug_target = aug_target

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
#         transform = self.trans_loader.get_composed_transform(aug)
        pre_transform = self.trans_loader.get_crop_transform(aug)
        post_transform = self.trans_loader.get_hdf5_transform(aug)
        aug_transform = self.trans_loader.get_aug_transform(self.aug_type, self.aug_target)
#         dataset = SimpleDataset(data_file, transform)
        dataset = AugSimpleDataset(data_file, pre_transform=pre_transform, post_transform=post_transform, aug_target=self.aug_target)
        dataset.set_aug_transform(aug_transform) # TODO: AugSimple set_aug_transform
        self.dataset = dataset
        collate_fn = self.get_collate() # to get different transform for every batch
        num_workers = 12 # TODO: or???
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = num_workers, pin_memory = True, collate_fn=collate_fn)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader
    
    def get_collate(self):
        def wrapper(batch):
            transform = self.trans_loader.get_aug_transform(self.aug_type, aug_target=self.aug_target)
            self.dataset.set_aug_transform(transform)
            # if auto_collation then default_collate, else default_convert
            return torch.utils.data._utils.collate.default_collate(batch)
#             return torch.utils.data._utils.collate.default_convert(batch)
        return wrapper

class AugSetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, aug_type, aug_target, n_episode =100, recons_func = None):
        ''' to get a data_loader
        :param aug_type: str, 'rotate', ...
        '''
        super(AugSetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)
        
        self.dataset = None
        self.aug_type = aug_type
        self.aug_target = aug_target

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
#         transform = self.trans_loader.get_composed_transform(aug) # TODO: maybe change here?
        pre_transform = self.trans_loader.get_crop_transform(aug)
        aug_transform = self.trans_loader.get_aug_transform(self.aug_type, aug_target=self.aug_target)
        post_transform = self.trans_loader.get_hdf5_transform(aug)
        dataset = AugSetDataset( data_file , self.batch_size, pre_transform=pre_transform, post_transform=post_transform, aug_target=self.aug_target)
        dataset.set_aug_transform(aug_transform)
        self.dataset = dataset
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode ) # sample classes randomly
        collate_fn = self.get_collate() # to get different transform for every batch
        num_workers = 0 if self.aug_target=='batch' else 12 # BUGFIX: there's a bug when multiprocessing, but we can still multi-process when aug_target != 'batch' because only batch need align set_aug_transform for every batch?
        data_loader_params = dict(batch_sampler = sampler,  num_workers = num_workers, pin_memory = True, collate_fn=collate_fn)
        
        
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
    
    def get_collate(self):
        def wrapper(batch):
            transform = self.trans_loader.get_aug_transform(self.aug_type, aug_target=self.aug_target)
            self.dataset.set_aug_transform(transform)
            # if auto_collation then default_collate, else default_convert
            return torch.utils.data._utils.collate.default_collate(batch)
#             return torch.utils.data._utils.collate.default_convert(batch)
        return wrapper

class SetDataManager(DataManager):
    ''' to get a data_loader
    '''
    def __init__(self, image_size, n_way, n_support, n_query, n_episode =100, recons_func = None): # n_episode spell wrong.....
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode ) # sample classes randomly
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class VAESetDataManager(SetDataManager):
    def __init__(self, image_size, n_way, n_support, n_query, vaegan_exp, vaegan_step, lambda_zlogvar, fake_prob, n_episode =100, recons_func = None):
        super(VAESetDataManager, self).__init__(image_size, n_way, n_support, n_query, n_episode, recons_func)
        self.vaegan_params = {
            'exp_name':vaegan_exp, 
            'step':vaegan_step, 
            'lambda_var':lambda_zlogvar, 
            'fake_prob':fake_prob, 
        }
#         self.vaegan_exp = vaegan_exp
#         self.vaegan_step = vaegan_step
#         self.lambda_zlogvar = lambda_zlogvar
#         self.fake_prob = fake_prob
        
    def get_data_loader(self, data_file, aug):
#         pre_transform = self.trans_loader.get_simple_transform(aug)
#         aug_transform = self.trans_loader.get_vae_transform(self.vaegan, self.lambda_zlogvar, self.fake_prob)
#         post_transform = self.trans_loader.get_hdf5_transform(aug, inputs='tensor')
        transform = self.trans_loader.get_composed_transform(aug)
        fake_img_transform = self.trans_loader.get_hdf5_transform(aug)
        dataset = VAESetDataset(
            data_file, self.batch_size, transform, fake_img_transform, self.vaegan_params)
#         dataset = VAESetDataset(data_file , self.batch_size, pre_transform=pre_transform, post_transform=post_transform, aug_transform=aug_transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode ) # sample classes randomly
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True) # to debug
#         data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = False) # to debug
        # TODO: cancel debug mode
#         data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        
        return data_loader


# class VAESetDataManager(SetDataManager):
#     def __init__(self, image_size, n_way, n_support, n_query, vaegan, lambda_zlogvar, fake_prob, n_episode =100, recons_func = None):
#         super(VAESetDataManager, self).__init__(image_size, n_way, n_support, n_query, n_episode, recons_func)
#         self.vaegan = vaegan
#         self.lambda_zlogvar = lambda_zlogvar
#         self.fake_prob = fake_prob
        
#     def get_data_loader(self, data_file, aug):
#         pre_transform = self.trans_loader.get_simple_transform(aug)
#         aug_transform = self.trans_loader.get_vae_transform(self.vaegan, self.lambda_zlogvar, self.fake_prob)
#         post_transform = self.trans_loader.get_hdf5_transform(aug, inputs='tensor')
        
#         dataset = VAESetDataset(data_file , self.batch_size, pre_transform=pre_transform, post_transform=post_transform, aug_transform=aug_transform)
#         sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode ) # sample classes randomly
# #         data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True) # to debug
#         data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = False, 
#                                  collate_fn=self.get_collate(None)) # to debug
#         # TODO: cancel debug mode
# #         data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)
#         data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        
#         return data_loader
    
#     def get_collate(self, batch_transform):
#         def mycollate(batch):
#             collated = torch.utils.data.dataloader.default_collate(batch)
#             if configs.debug:
#                 print('batch[0]:', type(batch[0]), len(batch[0]))
#                 print('collated:', type(collated), len(collated))
#             if batch_transform is not None:
#                 collated = batch_transform(collated)
#             return collated
#         return mycollate


