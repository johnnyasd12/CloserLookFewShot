from data.datamgr import OrderedDataManager, TransformLoader
# from io_utils import parse_args
import argparse

from model_utils import restore_vaegan
from tqdm import tqdm
import numpy as np
import os

import skimage.io
from my_utils import describe

def b4vae(imgs, is_tanh, is_color):
    '''
    Args:
        imgs (ndarray): 0~1, shape=[b,3,h,w]
    '''
    if is_tanh:
        imgs = imgs*2 - 1
    imgs = np.transpose(imgs, (0,2,3,1)) # [b,h,w,3]
    if not is_color:
        imgs = imgs[:,:,:,:1] # [b,h,w,1]
    
    return imgs
    
def after_vae(imgs, is_tanh, is_color, out_order='NCHW'):
    '''
    Args:
        imgs (ndarray): -1~1, shape=[b,h,w,1]
        out_order (str): 'NCHW' or 'NHWC'
    '''
    if is_tanh:
        imgs = imgs/2 + 0.5 # 0~1
    if not is_color:
        imgs = np.repeat(imgs, repeats=3, axis=3) # [b,h,w,3]
    if out_order == 'NCHW':
        imgs = np.transpose(imgs, (0,3,1,2)) # [b,3,h,w]
    
    return imgs

def get_gen_path(ori_file, vaegan_exp, vaegan_step, zvar_lambda, is_training):
    is_train_str = '_is-train' if is_training else ''
    parent_dir = os.path.join(
        './filelists-'+vaegan_exp, 
        'step'+str(vaegan_step)+'_'+'lambda'+str(zvar_lambda)+is_train_str
    )
    sub_path = ori_file.split('filelists/')[1]
    gen_path = os.path.join(parent_dir, sub_path)
    return gen_path

def save_img(path, img, verbose=0):
    skimage.io.imsave(path, img)
    if verbose > 0:
        print('finish save image to:', path)

def n_gen_imgs_exists(batch_imgs_path):
    n_imgs = len(batch_imgs_path)
    count = 0
    for file_path in batch_imgs_path:
        if os.path.isfile(file_path):
            count += 1
    return count

if __name__ == '__main__':

#     args = parse_args('make_llvae_dataset')
    parser = argparse.ArgumentParser(description= 'make_llvae_dataset.')
#     parser.add_argument('--dataset', choices=['omniglot', 'CUB', 'miniImagenet', 'emnist'], help='dataset you want to reconstruct by Lr-LiVAE.')
    parser.add_argument('--dataset'     , choices=['CUB', 'miniImagenet', 'cross', 'omniglot', 'cross_char'])
    # TODO: this code is 冗餘
#     parser.add_argument('--mode', choices=['all', 'trian', 'val', 'test', 'noLatin'], help='data split.')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size when generating reconstructed samples, it seems the larger the better')
    parser.add_argument('--aug', action='store_true', help='whether do data augmentation before input to LrLiVAE')
    parser.add_argument('--vaegan_exp', type=str, help='the GMM_VAE_GAN experiment name')
    parser.add_argument('--vaegan_step', type=int, help='the GMM_VAE_GAN restore step')
    parser.add_argument('--zvar_lambda', type=float, help='the GMM_VAE_GAN zlogvar_lambda')
    parser.add_argument('--is_training', action='store_true', help='whether the gmm_vae_gan set as training mode.')
    parser.add_argument('--gpu', type=str, help='gpu id')
    parser.add_argument('--gen_mode', default='rec', choices=['rec', 'gen'])
    
    args = parser.parse_args()
    # ======= prepare dataset ======
#     data_path = os.path.join('filelists',args.dataset)
#     file = {'all':'all.json', 'train':'base.json', 'val':'val.json', 'test':'novel.json', 'noLatin':'noLatin.json'}
#     file_name = file[args.mode]
#     file_path = os.path.join(data_path, file_name)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    from train import get_train_val_filename
    base_file, _  = get_train_val_filename(args)
    print('base_file:', base_file)
    
    image_sizes = {'omniglot':28, 'cross_char':28}
    image_size = image_sizes[args.dataset]
    datamgr = OrderedDataManager(image_size=image_size, batch_size=args.batch_size, aug=args.aug)
    data_loader = datamgr.get_data_loader(data_file=base_file)
    
    ######## settings #######
    is_color_settings = {'omniglot':False, 'cross_char':False}
    is_color = is_color_settings[args.dataset]
    
    # ======= restore GMM_VAE_GAN model =======
    gvaegan = restore_vaegan(
        dataset=args.dataset, 
        vae_exp_name=args.vaegan_exp, 
        vae_restore_step=args.vaegan_step, 
        is_training=args.is_training, 
    )
    
#     # TODO: 
#     out_folder = os.path.join('debug','rec_dataset')
    
    tqdm_data_loader = tqdm(data_loader)
    n_data = 0
    sum_gen_files_exists = 0
    for i, data in enumerate(tqdm_data_loader):
        batch_img_path, batch_x, batch_y = data[0], data[1].numpy(), data[2].numpy()
        n_samples = batch_y.shape[0]
        n_gen_files_exists = n_gen_imgs_exists(batch_img_path)
        sum_gen_files_exists += n_gen_files_exists
        n_data += n_samples
        any_file_not_generated = n_gen_files_exists != n_samples
        
        if any_file_not_generated:
            print(True)
            batch_x_rec = b4vae(batch_x, gvaegan.data.is_tanh, is_color)
            batch_x_rec = gvaegan.rec_samples(batch_x_rec, args.zvar_lambda)
            batch_x_rec = after_vae(batch_x_rec, gvaegan.data.is_tanh, is_color, out_order='NHWC')

            for j in range(n_samples):
    #                 sample_ori = b4vae(batch_x[j:j+1], is_tanh=gvaegan.data.is_tanh, is_color=is_color)
                sample_ori_path = batch_img_path[j]
    #                 sample_rec = batch_x_rec[j:j+1]
                sample_rec = batch_x_rec[j]
    #                 describe(sample_rec, 'sample_rec')

    #                 file_name = str(j)+'.jpg'
    #                 out_sample_path = os.path.join(out_folder, file_name)
    #                 fig = gvaegan.data.data2fig(sample, save_path=out_sample_path)

    #                 rec_file_name = str(j)+'rec'+str(args.zvar_lambda)+'.jpg'
    #                 out_sample_rec_path = os.path.join(out_folder, rec_file_name)

                out_sample_rec_path = get_gen_path(
                    sample_ori_path, 
                    vaegan_exp=args.vaegan_exp, 
                    vaegan_step=args.vaegan_step, 
                    zvar_lambda=args.zvar_lambda, 
                    is_training=args.is_training
                )
                out_sample_rec_dir = os.path.dirname(out_sample_rec_path)
                if not os.path.exists(out_sample_rec_dir):
                    os.makedirs(out_sample_rec_dir)
                save_img(out_sample_rec_path, sample_rec)
    #                 print('out_sample_rec_path:', out_sample_rec_path)
    #                 fig_rec = gvaegan.data.data2fig(sample_rec, nr=1, nc=1, save_path=out_sample_rec_path)
    
    print('n_data:', n_data)
    print('sum_gen_files_exists:', sum_gen_files_exists)
    if sum_gen_files_exists == n_data:
        print('Warning: ALL %d files already exist None of them are changed by the program.' % (n_data))
    elif sum_gen_files_exists > 0:
        print('Warning: There are %d files already exist and some of them are overloaded by the program.' % (sum_gen_files_exists))
    else: # none exist file
        print('The program successfully generated %d images.' % (n_data))







