# should be executed after save_features.py

import os
import data.feature_loader as feat_loader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import numpy as np
# random.seed(42)
# np.random.seed(42)
from io_utils import *
from my_utils import *
from tqdm import tqdm
from scipy import stats

# import seaborn as sns
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

if __name__ == '__main__':
    print('draw_features.py start')
    params = parse_args('draw_features')
    checkpoint_dir = get_checkpoint_dir(params)
    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    
#     model = get_model(params)
#     few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    
#     if params.gpu_id:
#         set_gpu_id(params.gpu_id)
#         device = torch.device('cuda:'+str(params.gpu_id))
#         model = model.cuda()
#     else:
#         device = None
#         model = to_device(model)

    # draw whole data distribution or task sample distribution? 
    feat_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5")
    cl_data_file = feat_loader.init_loader(feat_file) # dictionary, keys: class_num, content: list of features
    ''' features
    for Conv4, dim = 1600 = 64*5*5
    '''
    feat_list = []
    label_list = []
    num_classes = len(cl_data_file.keys()) # useless
    cls_transform = {} # transform class_id to 0~draw_n_classes-1 to draw color
    draw_n_classes = params.d_classes
    draw_samples_per_class = params.d_samples
    draw_class_indices = np.random.permutation(list(cl_data_file.keys()))[:draw_n_classes]
    for i, cls_idx in enumerate(draw_class_indices):
        cls_transform[cls_idx] = i
        cls_feats = cl_data_file[cls_idx]
        cls_n_samples = len(cls_feats)
        
        sampled_indices = np.random.permutation(cls_n_samples)[:draw_samples_per_class]
        concat_feats = np.stack(cls_feats, axis=0)
        sampled_feats = concat_feats[sampled_indices]
        sampled_labels = np.array([cls_idx]*draw_samples_per_class)
        
        feat_list.append(sampled_feats)
        label_list.append(sampled_labels)
    
    features = np.concatenate(feat_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    
    print('features:', features.shape, ', labels:', labels.shape)
    print('features:\n', stats.describe(features, axis=None))
    
    # TODO: shuffle the feature and label?
    
    # TODO: standardize the features?
    
    # TODO: PCA to about 50 dimensions
    if 'pca' in params.reduce_mode:
        pca_dim = 50 if params.reduce_mode == 'pca-tsne' else 2
        pca = PCA(n_components=pca_dim)
        print(pca)
        print('Running PCA...')
        pca_feat = pca.fit_transform(features)
        print('PCA finished.')
    
    if 'tsne' in params.reduce_mode:
        perplexity = draw_samples_per_class
        perplexity = np.clip(perplexity, 5, 50) # it's sugguested that perplexity between 5~50
        tsne = TSNE(
            n_components = 2, 
            verbose = 1, 
            perplexity = perplexity, 
            early_exaggeration = 12., 
            learning_rate = 200, 
            init = 'random', 
        )
        print(tsne)
        print('Running t-SNE...')
        if params.reduce_mode == 'pca-tsne':
            final_feat = tsne.fit_transform(pca_feat)
        elif params.reduce_mode == 'tsne':
            final_feat = tsne.fit_transform(features)
        else:
            raise ValueError('Unknown reduce_mode: %s' % (params.reduce_mode))
        print('t-SNE finished.')
    else:
        final_feat = pca_feat
    
    print('features after dimension reduction:\n', stats.describe(final_feat, axis=None))
    # TODO: draw final_feat
    plt.figure(figsize=(8, 8))
    tqdm_feat = tqdm(range(final_feat.shape[0]))
    tqdm_feat.set_description('Drawing')
    colors = plt.cm.rainbow(np.linspace(0, 1, draw_n_classes))
    for i in tqdm_feat:
        cls_id = labels[i]
        transformed_cls_id = cls_transform[cls_id]
        color = colors[transformed_cls_id]
        plt.text(final_feat[i, 0], final_feat[i, 1], str(labels[i]), color=color,#plt.cm.Set1(labels[i]), 
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlim((final_feat[:,0].min(), final_feat[:,0].max()))
    plt.ylim((final_feat[:,1].min(), final_feat[:,1].max()))
    plt.xticks([])
    plt.yticks([])
    png_file = 'draw_features.png'
    eps_file = 'draw_feature.eps'
    plt.savefig(png_file)
    plt.savefig(eps_file, format='eps')
#     plt.show()
#     plt.close()
    
    
    
    
    
    