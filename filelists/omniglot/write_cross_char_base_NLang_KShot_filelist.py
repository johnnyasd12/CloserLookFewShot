import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd() 
data_path = join(cwd,'images')
savedir = './'

n_lang = 3 # noLatin = 49??
k_shot = None


#if not os.path.exists(savedir):
#    os.makedirs(savedir)

cl = -1
folderlist = []

language_folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
language_folder_list.sort()

if n_lang is not None:
    language_folder_list = language_folder_list[:n_lang] # only take first n languages

filelists = {}


for language_folder in language_folder_list:
    if language_folder == 'Latin':
        continue
    language_folder_path = join(data_path, language_folder)
    character_folder_list = [cf for cf in listdir(language_folder_path) if isdir(join(language_folder_path, cf))]
    character_folder_list.sort()
    print('language_folder:', language_folder)
    print('len(character_folder_list):', len(character_folder_list))
    for character_folder in character_folder_list:
        character_folder_path = join(language_folder_path, character_folder)
        label = join(language_folder,character_folder)
        folderlist.append(label)
        filelists[label] =  [ join(character_folder_path,img) for img in listdir(character_folder_path) if (isfile(join(character_folder_path,img)) and img[-3:] == 'png')]
        if k_shot is not None:
            filelists[label] = filelists[label][:k_shot]

filelists_flat = []
labellists_flat = []
for key, filelist in filelists.items():
    cl += 1
    random.shuffle(filelist)
    filelists_flat += filelist
    labellists_flat += np.repeat(cl, len(filelist)).tolist() 

print('cl:', cl, 'n_classes:', cl+1)

fname = 'noLatin'
if n_lang is not None:
    fname += '_%dlang'%(n_lang)
if k_shot is not None:
    fname += '_%dshot'%(k_shot)

fname += '.json'

fo = open(join(savedir, fname), "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item  for item in folderlist])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item  for item in filelists_flat])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item  for item in labellists_flat])
fo.seek(0, os.SEEK_END) 
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write(']}')

fo.close()

print("%s -OK"%(fname))
