# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x

import h5py
from my_utils import *


class VirtualSetDataset:
    def __init__(self, data, batch_size):#, transform): # one item = ONE specific Class, but batch_sampler call this several times at once
        
        self.X = data[0]
        self.y = data[1]

        self.cl_list = np.unique(self.y).tolist()

        self.sub_meta = {} # ALL images, content = class_1: [data_path_1, ..., data_path_n], class_k: [data_path_1, ..., ]
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for feat,label in zip(self.X, self.y):
            self.sub_meta[label].append(feat)

        self.sub_dataloaders = [] # there're k dataloaders, k is # classes
        sub_data_loader_params = dict(batch_size = batch_size, # how many data to grab at current category???
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = VirtualSubDataset(self.sub_meta[cl], cl)#, transform = transform )
            self.sub_dataloaders.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloaders[i]))

    def __len__(self):
        return len(self.cl_list)


class VirtualSubDataset: # one iteration is one image of one class
    def __init__(self, sub_meta, cl, transform=lambda x:x, target_transform=lambda x:x):
        self.sub_meta = sub_meta # feature array of certian class
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
#         image_path = os.path.join( self.sub_meta[i])
#         img = Image.open(image_path).convert('RGB')
#         img = self.transform(img)
        x = self.sub_meta[i]
        x = self.transform(x)
        target = self.target_transform(self.cl)
        return x, target

    def __len__(self):
        return len(self.sub_meta)

class SetDataset:
    def __init__(self, data_file, batch_size, transform): # one item = ONE specific Class, but batch_sampler call this several times at once
        with open(data_file, 'r') as f: # data_file is json
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {} # ALL images, content = class_1: [data_path_1, ..., data_path_n], class_k: [data_path_1, ..., ]
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloaders = [] # there're k dataloaders, k is # classes
        sub_data_loader_params = dict(batch_size = batch_size, # how many data to grab at current category???
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloaders.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloaders[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset: # one iteration is one image of one class
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta # list of image names(dirs)
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class HDF5Dataset:
    def __init__(self, data_file, transform, target_transform=identity):
        self.file_path = data_file
        self.imgs = None
        self.labels = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["labels"])
        
#         self.file = h5py.File(data_file, 'r')
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, i):
        if self.imgs is None:
            self.imgs = h5py.File(self.file_path, 'r')["images"]
            self.labels = h5py.File(self.file_path, 'r')["labels"]
        img = self.imgs[i]
        target = self.labels[i]
#         img = self.file['images'][i,:,:,:]
#         target = self.file['labels'][i]
        target = self.target_transform(target)
        return img, target
        
    def __len__(self):
#         return self.file['labels'].shape[0]
        return self.dataset_len
            

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity, return_path=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path

    def __getitem__(self,i):
        image_path = self.meta['image_names'][i]
#         image_path = os.path.join(???, image_path)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        if self.return_path:
            return image_path, img, target
        else:
            return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class EpisodicBatchSampler(object): # to sample n_way classes index for every iteration
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


# class VAESetDataset:
#     def __init__(self, data_file, batch_size, pre_transform, post_transform, aug_transform): # only ONE Class for one item, but batch_sampler call this several times at once
#         with open(data_file, 'r') as f: # data_file is json
#             self.meta = json.load(f)

#         self.cl_list = np.unique(self.meta['image_labels']).tolist()
#         self.debug_flag = 0 # another debug flag

#         self.sub_meta = {} # ALL images, content = class_1: [data_path_1, ..., data_path_n], class_k: [data_path_1, ..., ]
#         for cl in self.cl_list:
#             self.sub_meta[cl] = []

#         for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
#             self.sub_meta[y].append(x)

# #         global global_datasets # for multi processes
# #         global_datasets = []
#         self.sub_dataloaders = [] # there're k dataloaders, k is # classes
#         self.sub_datasets = [] # k datasets, k is # classes
#         sub_data_loader_params = dict(batch_size = batch_size, # how many data to grab at current category???
#                                   shuffle = True,
#                                   num_workers = 0, #use main thread only or may receive multiple batches
#                                   pin_memory = False)        
#         for cl in self.cl_list:
#             # can utilize AugSubDataset, but don't use set_aug_transform() & aug_target
#             sub_dataset = AugSubDataset(self.sub_meta[cl], cl, pre_transform=pre_transform, post_transform=post_transform, aug_transform=aug_transform)
#             self.sub_datasets.append(sub_dataset)
#             self.sub_dataloaders.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

#     def __getitem__(self,i):
#         return next(iter(self.sub_dataloaders[i]))

#     def __len__(self):
#         return len(self.cl_list)
        

class VAESetDataset:
    def __init__(self, data_file, batch_size, transform, fake_img_transform, vaegan_params): # only ONE Class for one item, but batch_sampler call this several times at once
#         self.vaegan_exp = vaegan_exp
#         self.vaegan_step = vaegan_step
#         self.lamdba_zlogvar = lambda_zlogvar
#         self.fake_prob = fake_prob
        self.vaegan_params = vaegan_params
        
        with open(data_file, 'r') as f: # data_file is json
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.debug_flag = 0 # another debug flag

        self.sub_meta = {} # ALL images, content = class_1: [data_path_1, ..., data_path_n], class_k: [data_path_1, ..., ]
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

#         global global_datasets # for multi processes
#         global_datasets = []
        self.sub_dataloaders = [] # there're k dataloaders, k is # classes
        self.sub_datasets = [] # k datasets, k is # classes
        sub_data_loader_params = dict(batch_size = batch_size, # how many data to grab at current category???
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = VAESubDataset(self.sub_meta[cl], cl, transform=transform, fake_img_transform=fake_img_transform, vaegan_params=vaegan_params)
            self.sub_datasets.append(sub_dataset)
            self.sub_dataloaders.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloaders[i]))

    def __len__(self):
        return len(self.cl_list)

class AugSetDataset:
    def __init__(self, data_file, batch_size, pre_transform, post_transform, aug_target): # only ONE Class for one item, but batch_sampler call this several times at once
        with open(data_file, 'r') as f: # data_file is json
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.debug_flag = 0 # another debug flag

        self.sub_meta = {} # ALL images, content = class_1: [data_path_1, ..., data_path_n], class_k: [data_path_1, ..., ]
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

#         global global_datasets # for multi processes
#         global_datasets = []
        self.sub_dataloaders = [] # there're k dataloaders, k is # classes
        self.sub_datasets = [] # k datasets, k is # classes
        sub_data_loader_params = dict(batch_size = batch_size, # how many data to grab at current category???
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = AugSubDataset(self.sub_meta[cl], cl, pre_transform=pre_transform, post_transform=post_transform, aug_target=aug_target)
            self.sub_datasets.append(sub_dataset)
            self.sub_dataloaders.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloaders[i]))

    def __len__(self):
        return len(self.cl_list)
    
    def set_aug_transform(self, aug_transform):
        for sub_dataset in self.sub_datasets:
            sub_dataset.set_aug_transform(aug_transform)

class AugSubDataset: # one item is one image of one class
    def __init__(self, sub_meta, cl, pre_transform, post_transform=transforms.ToTensor(), aug_transform=identity, target_transform=identity, aug_target='batch'):
        self.sub_meta = sub_meta # list of image names(dirs)
        self.cl = cl 
        
        self.pre_transform = pre_transform
        self.aug_transform = aug_transform
        self.post_transform = post_transform
        
        self.target_transform = target_transform
        
        self.debug_flag = 0
        self.aug_target = aug_target
        

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = self.pre_transform(img)
        img = self.aug_transform(img)
        img = self.post_transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)
    
    def set_aug_transform(self, aug_transform):
        '''
        :param aug_transform: the transform function
        '''
#         print('set_aug_transform called')
        self.aug_transform = aug_transform
        self.debug_flag += 1
#         print('self.debug_flag =', self.debug_flag)

class AugSimpleDataset:
    def __init__(self, data_file, pre_transform, post_transform, aug_transform=identity, target_transform=identity, aug_target='all'):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        self.pre_transform = pre_transform
        self.aug_transform = aug_transform
        self.post_transform = post_transform
        
        self.target_transform = target_transform
        
        self.debug_flag = 0
        self.aug_target = aug_target

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
#         timer = Timer('open to RGB')
        img = Image.open(image_path).convert('RGB')
        img = self.pre_transform(img)
        img = self.aug_transform(img)
#         if self.debug_flag <= 2 and self.aug_target=='test-sample':
#             folder = 'debug-'+self.aug_target
#             if not os.path.exists(folder):
#                 os.makedirs(folder)
#             filename = str(self.debug_flag) + '-' + str(i) + '.jpg'
#             print('saving', filename)
#             file_path = os.path.join(folder, filename)
#             img.save(file_path)
        img = self.post_transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
#         timer()
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])
    
    def set_aug_transform(self, aug_transform):
        '''
        :param aug_transform: the transform function
        '''
#         print('set_aug_transform called')
        self.aug_transform = aug_transform
        self.debug_flag += 1
#         print('self.debug_flag =', self.debug_flag)

class VAESubDataset: # one iteration is one image of one class
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), fake_img_transform=None, vaegan_params=None, target_transform=identity):
        '''
        Args:
            vae_params (dict): keys = ['exp_name', 'step', 'lambda_var', 'fake_prob']
        '''
        self.sub_meta = sub_meta # list of image names(dirs)
        self.cl = cl 
        self.transform = transform
        self.fake_img_transform = fake_img_transform
        self.target_transform = target_transform
        self.vaegan_params = vaegan_params
#         # to avoid circular import
#         from make_llvae_dataset import get_gen_path

    def __getitem__(self,i):
        # to avoid circular import
        from make_llvae_dataset import get_gen_path
        vaegan_exp = self.vaegan_params['exp_name']
        vaegan_step = self.vaegan_params['step']
        fake_prob = self.vaegan_params['fake_prob']
        lambda_zlogvar = self.vaegan_params['lambda_var']
        vaegan_is_train = self.vaegan_params['is_training']
        
        rand_num = np.random.random()
        use_vaegan_img = rand_num <= fake_prob
        
        image_path = os.path.join( self.sub_meta[i])
        if use_vaegan_img:
            image_path = get_gen_path(
                ori_file=image_path, 
                vaegan_exp=vaegan_exp, 
                vaegan_step=vaegan_step, 
                zvar_lambda=lambda_zlogvar, 
                is_training=vaegan_is_train
            )
        img = Image.open(image_path).convert('RGB')
        # TODO: different transform
        if use_vaegan_img:
            img = self.fake_img_transform(img)
        else:
            img = self.transform(img)
        
#         if False:
#             from my_utils import describe
#             print('use_vaegan_img:', use_vaegan_img)
#             describe(np.array(img), 'VAESubDataset/__getitem__()/img')
        
        
        target = self.target_transform(self.cl)
        return img, target
    
    def __len__(self):
        return len(self.sub_meta)

