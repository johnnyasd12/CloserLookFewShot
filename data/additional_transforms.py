# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)



class ImageJitter(object):
    def __init__(self, transformdict):
        # [(IMageEnhacne.Brightness, 0.4), (ImageEnhance.Contrast, 0.4), (ImageEnhance.Sharpness, 0.4)]
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms)) # from 0 to 1

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1 # (-0.4 ~ 0.4) + 1 = 0.6 ~ 1.4
            out = transformer(out).enhance(r).convert('RGB')

        return out




