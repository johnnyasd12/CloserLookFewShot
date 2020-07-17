import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd() 
data_path = join(cwd,'CUB_200_2011/images')
savedir = './'
# dataset_list = ['base','val','novel']

n_base_class = 25

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


file_list = []
label_list = []
count_n_base_class = 0
        
for i, classfile_list in enumerate(classfile_list_all):
    if (i%2 == 0):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        count_n_base_class += 1
        if count_n_base_class == n_base_class:
            break

dataset = 'base_%dcl'%(n_base_class)
fo = open(savedir + dataset + ".json", "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item  for item in folder_list])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item  for item in file_list])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item  for item in label_list])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("%s -OK" %dataset)
