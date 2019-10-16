# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, HDF5Dataset, AugSetDataset, AugSimpleDataset
from abc import abstractmethod

import torchvision.transforms.functional as TF
import random

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
        elif transform_type=='Scale':
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
    
    def get_hdf5_transform(self, aug = False): # simple_transform + this = composed_transform
        if aug:
            transform_list = ['ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
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
            angle = random.randint(*abs_angle_range)*(-1)#**random.randint(0,1) # 0 +- angle_range
            return angle
        
        def get_random_factor(abs_perturb_range):
            '''
            :param abs_perturb_range: tuple
            '''
            # only one side
            lower, upper = abs_perturb_range
            perturb = (lower + (upper-lower)*random.random())*(-1)#**random.randint(0,1) # 1+-(0~b) 
            factor = 1 + perturb
            return factor
        

        params = {}
        params['rotate'] = {
            'train_range':(0, 20), # should +/-
            'test_range':(15, 25), # should +/-
            'angle':None
        }
        params['bright'] = {
            'train_range':(0, 0.3), # should 1 +/- range
            'test_range':(0.25, 0.5), # should 1 +/- range
            'factor':None
        }
        params['contrast'] = {
            'train_range':(0, 0.3), # should 1 +/- range
            'test_range':(0.25, 0.5), # should 1 +/- range
            'factor':None
        }

        # Deciding transform parameters
        if aug_target == 'batch': # random select for every batch
            params['rotate']['angle'] = get_random_rotate_angle(params['rotate']['train_range'])
            params['bright']['factor'] = get_random_factor(params['bright']['train_range'])
            params['contrast']['factor'] = get_random_factor(params['contrast']['train_range'])

        elif aug_target == 'test-batch': # random select for every batch
            params['rotate']['angle'] = get_random_rotate_angle(params['rotate']['test_range'])
            params['bright']['factor'] = get_random_factor(params['bright']['test_range'])
            params['contrast']['factor'] = get_random_factor(params['contrast']['test_range'])
        
        def wrapper(img):

            nonlocal params

            if aug_target == 'batch' or aug_target == 'test-batch':
                pass # nothing to do

            # Deciding transform parameters
            elif aug_target == 'sample':
                params['rotate']['angle'] = get_random_rotate_angle(params['rotate']['train_range'])
                params['bright']['factor'] = get_random_factor(params['bright']['train_range'])
                params['contrast']['factor'] = get_random_factor(params['contrast']['train_range'])

            elif aug_target == 'test-sample':
                params['rotate']['angle'] = get_random_rotate_angle(params['rotate']['test_range'])
                params['bright']['factor'] = get_random_factor(params['bright']['test_range'])
                params['contrast']['factor'] = get_random_factor(params['contrast']['test_range'])

            else:
                raise ValueError('Invalid aug_target: %s' %(aug_target))

            # Process the image
            if aug_type == 'rotate':
                img = TF.rotate(img, params['rotate']['angle'])
                print('rotate angle:', params['rotate']['angle'])

            elif aug_type == 'bright':
                img = TF.adjust_brightness(img, params['bright']['factor'])
                print('bright factor:', params['bright']['factor'])

            elif aug_type == 'contrast':
                img = TF.adjust_contrast(img, params['contrast']['factor'])
                print('contrast factor:', params['contrast']['factor'])

            elif aug_type == 'mix':
                img = TF.adjust_contrast(img, params['contrast']['factor'])
                img = TF.adjust_brightness(img, params['bright']['factor'])
                img = TF.rotate(img, params['rotate']['angle'])

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
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       # TODO: tune pin_memory
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
        post_transform = self.trans_loader.get_hdf5_transform(aug)
        aug_transform = self.trans_loader.get_aug_transform(self.aug_type, aug_target=self.aug_target)
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

#     def collate_fn_task(self, batch): # TODO: deprecate this
#         '''
#         type(batch) = list, len(batch) = n_way
#         type(batch[0] = list, len(batch[0]) = 2)
#         type(batch[0][0] = Torch.Tensor, shape = [batch_size, ...]) (img)
#         type(batch[0][1] = Torch.Tensor, shape = [batch_size]) (target)
#         '''
#         a = batch[0][0]
#         b = batch[0][1]
#         print(a.shape, b.shape)
#         # initialize transform
        
#         # randomly set transform parameter
        
#         # compose several transforms
        
#         # apply transform on batch data
#         for (cls_batch_x, cls_batch_y) in batch:
            
#             # to PIL image
            
#             # transform
            
#             # to torch tensor
#             pass

class SetDataManager(DataManager):
    ''' to get a data_loader
    '''
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100, recons_func = None): # n_episode spell wrong.....
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide ) # sample classes randomly
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


