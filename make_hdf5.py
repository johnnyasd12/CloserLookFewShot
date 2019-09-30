import argparse
from data.dataset import SimpleDataset
from data.datamgr import SimpleDataManager, ResizeDataManager
import os
from tqdm import tqdm
import numpy as np
import h5py

parser = argparse.ArgumentParser(description= 'yeeeee, im description of make_hdf5 parser!')
parser.add_argument('--dataset', help='CUB/miniImagenet/omniglot/emnist', required=True)
parser.add_argument('--mode', help='all/train/val/test/', required=True)
parser.add_argument('--aug', action='store_true', help='use the augmented data or not')
parser.add_argument('--channel_order', help='NCHW for PyTorch / NHWC for ???', required=True)
parser.add_argument('--img_size', help='image size', required=True, type=int)
parser.add_argument('--batch_size', help='The batch size when processing the data', default=50, type=int)
parser.add_argument('--debug', action='store_true', help='debug mode on')

args = parser.parse_args()
# assert args.dataset is not None
# assert args.mode is not None
# assert args.argument is not None

# prepare to load data
data_path = os.path.join('filelists',args.dataset)
file = {'all':'all.json', 'train':'base.json', 'val':'val.json', 'test':'novel.json'}
file_path = os.path.join(data_path,file[args.mode])
datamgr = ResizeDataManager(args.img_size, batch_size=50)
data_loader = datamgr.get_data_loader(data_file=file_path, aug=args.aug)
imgs_list = []
labels_list = []

# load data
t_loader = tqdm(data_loader)
for i, data in enumerate(t_loader):
    batch_x, batch_y = data[0].numpy(), data[1].numpy()
    if args.channel_order == 'NHWC':
        batch_x = np.transpose(batch_x, axis=(0,2,3,1))
    imgs_list.append(batch_x)
    labels_list.append(batch_y)
    if i==2 and args.debug:
        break

imgs = np.concatenate(imgs_list, axis=0)
labels = np.concatenate(labels_list, axis=0)
print('Final: imgs shape:',imgs.shape,', labels shape:',labels.shape)

# write hdf5 file
filename = args.mode + '-' + args.channel_order + '-' + str(args.img_size)
if args.aug:
    filename += '-aug'
filename += '.h5'
out_path = os.path.join(data_path, 'hdf5')
if not os.path.exists(out_path):
    print('make directory:', out_path)
    os.makedirs(out_path)
out_file = os.path.join(out_path, filename)
print('making file:', out_file)

with h5py.File(out_file,'w-') as f:
    f['images'] = imgs
    f['labels'] = labels
    
print('Finish writing file:', out_file)


