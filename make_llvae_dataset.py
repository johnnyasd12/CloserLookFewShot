from data.datamgr import OrderedDataManager, TransformLoader
from io_utils import parse_args

from model_utils import restore_vaegan
import os

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
    
def after_vae(imgs, is_tanh, is_color):
    '''
    Args:
        imgs (ndarray): -1~1, shape=[b,h,w,1]
    '''
    if is_tanh:
        imgs = imgs/2 + 0.5 # 0~1
    imgs = np.transpose(imgs, (0,3,1,2)) # [b,1,h,w]
    if not is_color:
        imgs = np.repeat(imgs, repeats=3, axis=1) # [b,3,h,w]
    
    return imgs

if __name__ == '__main__':

    args = parse_args('make_llvae_dataset')
    
    # ======= prepare dataset ======
    data_path = os.path.join('filelists',args.dataset)
    file = {'all':'all.json', 'train':'base.json', 'val':'val.json', 'test':'novel.json', 'noLatin':'noLatin.json'}
    file_name = file[args.mode]
    file_path = os.path.join(data_path, file_name)
    
    image_sizes = {'omniglot':28,}
    image_size = image_sizes[args.dataset]
    datamgr = OrderedDataManager(image_size=image_size, batch_size=args.batch_size, aug=args.aug)
    data_loader = datamgr.get_data_loader(data_file=file_path)
    
    ######## settings #######
    is_color_settings = {'omniglot':False}
    is_color = is_color_settings[args.dataset]
    
    # ======= restore GMM_VAE_GAN model =======
    gvaegan = restore_vaegan(
        dataset=args.dataset, 
        vae_exp_name=args.vaegan_exp, 
        vae_restore_step=args.vaegan_step, 
        is_training=args.is_training, 
    )
    
    ####### set output folder ######
    # TODO: 
    out_folder = os.path.join('debug','rec_dataset')
    
    tqdm_data_loader = tqdm(data_loader)
    for i, data in enumerate(tqdm_data_loader):
        batch_img_path, batch_x, batch_y = data[0], data[1].numpy(), data[2].numpy()
        
        batch_x_rec = b4vae(batch_x, gvaegan.data.is_tanh, is_color)
        batch_x_rec = gvaegan.rec_samples(batch_x_rec, args.zvar_lambda)
        batch_x_rec = after_vae(batch_x_rec, gvaegan.data.is_tanh, is_color)
        







