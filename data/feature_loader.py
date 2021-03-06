import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None, store_path = False):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
            if store_path:
                self.all_paths = []
        else:
            self.f = file_handle # opened HDF5 file
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
            if store_path:
                self.all_paths = self.f['all_paths'][...]
            
    def __getitem__(self, i):
        if store_path:
            return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i]), self.all_paths[i]
        else:
            return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename, return_path=False):
    '''
    :return: dictionary, key = class_idx, content = list of all_data_features?
    '''
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f, store_path = return_path)

    #all_labels = [ l for l  in fileset.all_labels if l != 0]
    all_feats = fileset.all_feats_dset # ndarray, shape=(n_data, n_dims)
    all_labels = fileset.all_labels # ndarray, shape=(n_data,)
    if return_path:
        all_paths = fileset.all_paths # shape=(n_data,)
    
    class_list = np.unique(np.array(all_labels)).tolist() 
    inds = range(len(all_labels))

    cl_features = {}
    cl_filepaths = {}
    for cl in class_list:
        cl_features[cl] = []
        if return_path:
            cl_filepaths[cl] = []
    
    for ind in inds:
        cl = all_labels[ind]
        feats = all_feats[ind]
        path = all_paths[ind]
        cl_features[cl].append(feats)
        if return_path:
            cl_filepaths[cl].append(path)
    
    if return_path:
        return cl_features, cl_filepaths
    else:
        return cl_features
