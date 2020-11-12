import os
import numpy as np



def info_interval_2_folder_name(info_interval):
    info_int_s1 = str(info_interval[0]).zfill(2)
    info_int_s2 = str(info_interval[1]).zfill(2)
    datafolder = 'filelists/virtual_info%s%s/'%(info_int_s1, info_int_s2)
    return datafolder

def get_random_indices(arr, n):
    # return shape: (n, arr.ndim)
    dim_indices = []
    for dim_size in arr.shape:
        indices = np.random.choice(dim_size, size = n, replace = False) # shape:(n,)
        dim_indices.append(indices)
    dim_indices = np.stack(dim_indices) # shape: (arr.ndim, n)
    dim_indices = dim_indices.T # shape: (n, arr.ndim)
    return dim_indices


# def load_dataset(path):
#     assert '.npz' in path, 'load path should be .npz file'
#     data = np.load(path)
#     return data['X'], data['y']

class DatasetGenerator:
    def __init__(self, n_dims, n_all_classes, n_classes, n_samples_per_class):#, datafolder='./'):
        self.n_dims = n_dims
        self.n_all_classes = n_all_classes
        self.n_classes = n_classes # dictionary base50cl/base400cl/val/novel
        self.n_samples_per_class = n_samples_per_class
        self.n_samples = n_all_classes * n_samples_per_class
#         self.datafolder = datafolder
        assert n_all_classes == n_classes['base'] + n_classes['val'] + n_classes['novel']
        
#     def gen_linear_dataset(self, datafolder, all_latent_centers, transform_matrix, shift_factor):
#         ''' generate dataset decided by n_latent factors follows Gaussian distributions
#         '''
#         cl_feat_noise_rate = 0.5
# #         latent_centers = np.random.normal(loc = 0, scale = 1, size = (self.n_all_classes, n_latent))
# #         transform_matrix = np.random.normal(loc = 0, scale = 1, size = (self.n_dims, n_latent))
        
    def gen_Gaussian_dataset(self, datafolder, distrib_center, distrib_std, cls_x_std, informative_interval):
        ''' generate base100cl, base200cl, base(400cl), val, novel datasets
        '''
        n_informative = informative_interval[1] - informative_interval[0] + 1
#         cl_n_informative = n_informative # // 2 # this is replaced by info_noisy_frac
#         cl_info_noise_frac = 0.2
#         info_noise_std = 10
        cl_info_noise_frac = 0.0 
        # 0.0 naive5cl=1%up(/97%), domain=3%up, Ldomain=11.8%up / 0.2 naive5cl=4%up, domain=1.7%up 
        # 0.5 naive5cl=4%up, domain=1%up / 0.8 is trash
        
        distrib_center_info_feat = distrib_center[:n_informative] # hack to easy implement becuz distrib_center all the same currently
        distrib_std_info_feat = distrib_std[:n_informative] # hack to easy implement becuz distrib_center all the same currently
#         distrib_center_info_feat = distrib_center[:cl_n_informative] # hack to easy implement becuz distrib_center all the same currently
#         distrib_std_info_feat = distrib_std[:cl_n_informative] # hack to easy implement becuz distrib_center all the same currently
        
        informative_x_centers = np.random.normal(
            loc = distrib_center_info_feat, scale = distrib_std_info_feat, 
            size = (self.n_all_classes, n_informative)
#             size = (self.n_all_classes, cl_n_informative)
        )

        X_info = []
        for cl in range(self.n_all_classes):
            # each class should have different feature intervals. (but all inside informative_interval)
            info_x_center = informative_x_centers[cl]
            cl_X_info = np.random.normal(
                loc = info_x_center, scale = cls_x_std, 
                size = (self.n_samples_per_class, n_informative)
#                 size = (self.n_samples_per_class, cl_n_informative)
#                 size = (n_informative, self.n_samples_per_class) # this would get error @@
            )
            # TODO: add random noise on X_info (column-wise? element-wise?)
            # column-wise add random noise (different for each class)
            n_noise_cols = int(n_informative * cl_info_noise_frac)
            noise_indices = np.random.choice(cl_X_info.shape[1], n_noise_cols, replace = False)
#             cl_X_info[:, noise_indices] += np.random.normal(
#                 0, info_noise_std, size = (cl_X_info.shape[0], n_noise_cols))
            cl_X_info[:, noise_indices] = np.random.normal(
                0, distrib_std[0], size = (cl_X_info.shape[0], n_noise_cols))
            
            X_info.append(cl_X_info)
        X_info = np.concatenate(X_info, axis=0)
        X_noninfo_center = distrib_center[:-n_informative] # hack (actually should be dimensions except informative_interval)
        X_noninfo_std = distrib_std[:-n_informative] # hack (actually should be dimensions except informative_interval)
        X_noninfo = np.random.normal(
            loc = X_noninfo_center, scale = X_noninfo_std, 
            size = (self.n_samples, self.n_dims-n_informative)
        )
        X1 = X_noninfo[:, :informative_interval[0]] # hack
        X2 = X_noninfo[:, informative_interval[0]:] # hack
        X_all = np.concatenate(
            (X1, X_info, X2)
            , axis=1)
        y_all = np.repeat(np.arange(self.n_all_classes),self.n_samples_per_class,axis=0) # [0 0 1 1 2 2 ...]
        test_dataset_ls = ['val', 'novel']
        Xs = {}
        ys = {}
#         for split in test_dataset_ls:
        for split in self.n_classes.keys():
            dataset_n_classes = self.n_classes[split]
            dataset_n_samples = dataset_n_classes * self.n_samples_per_class
            if 'base' in split:
                X = X_all[:dataset_n_samples]
                y = y_all[:dataset_n_samples]
            elif split == 'novel':
                X = X_all[-dataset_n_samples:]
                y = y_all[-dataset_n_samples:]
            elif split == 'val':
                novel_n_samples = self.n_classes['novel'] * self.n_samples_per_class
                X = X_all[-novel_n_samples-dataset_n_samples:-novel_n_samples]
                y = y_all[-novel_n_samples-dataset_n_samples:-novel_n_samples]
            else:
                raise ValueError('Unknown split: %s'%(split))

            Xs[split] = X
            ys[split] = y
            filename = split + '.npz'
            out_path = os.path.join(datafolder, filename)
            print('Saving file: %s'%(out_path))
            if not os.path.exists(datafolder):
                print('Folder not exist: "%s"'%(datafolder))
                print('Making directory...')
                os.makedirs(datafolder)
            np.savez(out_path, X=X, y=y)
        
        return Xs, ys


def get_non_info_prob(n_informative, n_cl_informative, n_base_cl):
    once_non_informative_prob = (n_informative - n_cl_informative) / n_informative
    base_non_informative_prob = once_non_informative_prob ** n_base_cl
    return base_non_informative_prob

if __name__ == '__main__':
    
    n_dims = 100
    n_all_classes = 600 # base max 400 / source_val 100 / source_novel 100
    n_samples_per_class = 100 # i think max 80support + 15query
    n_classes = {
        'base':400, 'val':100, 'novel':100, 
        'base5cl':5, 'base25cl':25, 'base50cl':50, 'base100cl':100, 'base200cl':200,}
    distrib_center = np.zeros(n_dims)
    distrib_std = 20 * np.ones(n_dims)
    cls_x_std = 4
    
    # dataset_informative_intervals = [(0, 29), (10, 39), (20, 49), (30, 59)]
    # dataset_informative_intervals = [(0, 49), (25, 74), (50, 99)]
    info_interval = (50, 99)
#     datafolder = './virtual'
    datafolder = info_interval_2_folder_name(info_interval)
    
    
    dataset_generator = DatasetGenerator(
        n_dims = n_dims, n_all_classes = n_all_classes, n_classes = n_classes, 
        n_samples_per_class = n_samples_per_class)#, datafolder = datafolder)

    # for info_interval in dataset_informative_intervals:

    print('datafolder:', datafolder)
    # if should_gen_Gaussian:
    dataset_generator.gen_Gaussian_dataset(
        datafolder = datafolder, 
        distrib_center=distrib_center, distrib_std=distrib_std, 
        cls_x_std=cls_x_std, informative_interval=info_interval)



