import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle # opened HDF5 file
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
            
            ############### Sanity Check ###############
#             print('SimpleHDF5Dataset/self.all_labels:', type(self.all_labels), self.all_labels.shape, self.all_labels)
            ############### Sanity Check ###############
            
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename):
    '''
    :return: dictionary, key = class_idx, content = list of all_data_features?
    '''
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #all_labels = [ l for l  in fileset.all_labels if l != 0]
    all_feats = fileset.all_feats_dset # ndarray, shape=(n_data, n_dims)
    all_labels = fileset.all_labels # ndarray, shape=(n_data,)
    
    
    ############### Sanity Check ###############
#     print('init_loader/all_labels:', type(all_labels), all_labels.shape, all_labels)
    ############### Sanity Check ###############
    
    # don't need this code now
#     while np.sum(all_feats[-1]) == 0:
#         print('init_loader/all_feats[-1]:', all_feats[-1])
#         all_feats  = np.delete(all_feats,-1,axis = 0)
#         all_labels = np.delete(all_labels,-1,axis = 0)
    
    ############### Sanity Check ###############
#     print('init_loader/all_labels after delete:', type(all_labels), all_labels.shape, all_labels)
    ############### Sanity Check ###############
    
    class_list = np.unique(np.array(all_labels)).tolist() 
    inds = range(len(all_labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    
    for ind in inds:
        cl = all_labels[ind]
        feats = all_feats[ind]
        cl_data_file[cl].append(feats)
    
    ############### Sanity Check has Problem here (problem trace to save_features) ###############
#     sum_n_data = 0
    
#     n_data_frequency = {}
#     for cl in class_list: # find the most frequent number
#         n_data = len(cl_data_file[cl])
#         if n_data in n_data_frequency:
#             n_data_frequency[n_data] += 1
#         else:
#             n_data_frequency[n_data] = 1
#     most_freq_n_data = max(n_data_frequency.keys(), key=(lambda k: n_data_frequency[k]))
    
#     for cl in class_list:
#         n_data = len(cl_data_file[cl])
#         sum_n_data += n_data
# #         if n_data != most_freq_n_data:
# #             print('init_loader/ class', cl, ', n_data:', n_data, 'is NOT', most_freq_n_data, '!!!!!!!')
#     print('sum_n_data:', sum_n_data)
    ############### Sanity Check ###############
    
    
    return cl_data_file
