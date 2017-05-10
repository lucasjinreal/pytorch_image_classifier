# -*- coding: utf-8 -*-
# file: data_loader.py
# author: JinTian
# time: 10/05/2017 8:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from global_config import *
from PIL import Image


class DataLoader(object):
    def __init__(self, data_dir, image_size, batch_size=4):
        """
        this class is the normalize data loader of PyTorch.
        The target image size and transforms can edit here.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
            'val': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
        }

        self._init_data_sets()

    def _init_data_sets(self):
        self.data_sets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}

        self.data_loaders = {x: torch.utils.data.DataLoader(self.data_sets[x], batch_size=self.batch_size,
                                                            shuffle=True, num_workers=4)
                             for x in ['train', 'val']}
        self.data_sizes = {x: len(self.data_sets[x]) for x in ['train', 'val']}
        self.data_classes = self.data_sets['train'].classes

    def load_data(self, data_set='train'):
        return self.data_loaders[data_set]

    def show_image(self, tensor, title=None):
        inp = tensor.numpy().transpose((1, 2, 0))
        # put it back as it solved before in transforms
        inp = self.normalize_std * inp + self.normalize_mean
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()

    def make_predict_inputs(self, image_file):
        """
        this will make a image to PyTorch inputs, as the same with training images.
        this will return a tensor, default not using CUDA.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_tensor = self.data_transforms['val'](image).float()
        image_tensor.unsqueeze_(0)
        return Variable(image_tensor)



