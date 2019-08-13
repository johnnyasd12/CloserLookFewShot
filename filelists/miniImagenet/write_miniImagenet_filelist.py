import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd() 
data_path = join(cwd, 'preprocessed') # join(cwd,'ILSVRC2015/Data/CLS-LOC/train')
savedir = './'
dataset_list = ['base', 'val', 'novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

cl = -1
folderlist = [] # to store label??

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} } # label1:[fname1,fname2,...], label2:[fname...], ...
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

for dataset in dataset_list:
    with open(datasetmap[dataset] + ".csv", "r") as lines: # read train.csv, val.csv, test.csv
        for i, line in enumerate(lines): # record file paths ??
            if i == 0:
                continue
            fid, _ , label = re.split(',|\.', line) # fid here: filename before .jpg
            label = label.replace('\n','')
#             print('fid',fid)
#             print('label',label)
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = [] # new label
                fnames = listdir( join(data_path, label) ) # preprocessed files names.jpg in this class
                for i,fname in enumerate(fnames):
#                 fname_number = [ int(re.split('_|\.', fname)[1]) for fname in fnames] # BUGFIX
                fname_number = [ int(re.split('_|\.', fname)[0]) for fname in fnames] # preprocessed files names before.jpg
                sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0] # this class files names.jpg
                 
#             fid = int(fid[-5:])-1 # last 5 number of fid
#             print('fid after:',fid,', len of sorted_fnames:',len(sorted_fnames))
#             name = sorted_fnames[fid]
            name = fid[-8:] + '.jpg'
            fname = join( data_path,label, name ) # file path, BUGFIX: sorted_fnames[fid]
#             print('fname after:',fname)
#             print()
            filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist() 

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
