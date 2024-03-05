from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np
from lib.klib.baseio import *
from tools.Image_Tools import gen_circle_3d, gen_circle_gaussian_3d
from tools.dataset import Compose
import pandas as pd

import copy
import time 
from configs import config_3d

args = config_3d.args
resize_radio = args.resize_radio
r_resize = args.r_resize
data_shape = args.data_shape


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir, images_dir_list, labels_dir_list, mode):
        self.image_names = sorted(images_dir_list)
        self.labels = sorted(labels_dir_list)
        self.images_dir = images_dir
        self.labels_dir = labels_dir


        self.tx = torchvision.transforms.Compose([])
        self.tx_dis = torchvision.transforms.Compose([])
        self.tx1 = torchvision.transforms.Compose([])
        self.lx = torchvision.transforms.Compose([])
        
        self.tx_total = None
        if mode == 'train':
            self.tx_total = Compose([
                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomVerticalFlip(p=0.5)
            ])
        

        self.lx_radius = torchvision.transforms.Compose([])
        self.lx_exist = torchvision.transforms.Compose([])

        self.lx_vector1 = torchvision.transforms.Compose([])
        self.lx_vector2 = torchvision.transforms.Compose([])
        self.lx_vector3 = torchvision.transforms.Compose([])
        self.lx_vector1_bin = torchvision.transforms.Compose([])
        self.lx_vector2_bin = torchvision.transforms.Compose([])
        self.lx_vector3_bin = torchvision.transforms.Compose([])


            

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        seq_len = 3
        begin_time = time.time()
        i_img = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_img.tif').astype(np.float32)
        depth, height, width = data_shape
        
        img_shape = i_img.shape
        # print(img_shape)
        
        i_new = np.zeros([seq_len, 1, depth, height, width],dtype=np.float32)
        i_new_lab = np.zeros([seq_len, 1, depth, height, width],dtype=np.float32)
        i_new_dis = np.zeros([seq_len, 1, depth, height, width],dtype=np.float32)
        

        l_radius_new = np.zeros([seq_len])
        l_exist_new = np.zeros([seq_len])

        l_vector1_new = np.zeros([seq_len])
        l_vector2_new = np.zeros([seq_len])
        l_vector3_new = np.zeros([seq_len])

        l_vector1_bin_new = np.zeros([seq_len])
        l_vector2_bin_new = np.zeros([seq_len])
        l_vector3_bin_new = np.zeros([seq_len])


        for seq_id in range(seq_len):
            i1 = np.sqrt(copy.deepcopy(i_img[seq_id*3])) / 255
            i_lab1 = copy.deepcopy(i_img[seq_id*3+1]//255)
            i_dis1 = copy.deepcopy(i_img[seq_id*3+2]//255)
            
            

            i_new[seq_id][0] = copy.deepcopy(i1)
            i_new_lab[seq_id][0] = copy.deepcopy(i_lab1)
            i_new_dis[seq_id][0] = copy.deepcopy(i_dis1)

            pd_data = pd.read_csv(self.images_dir + self.image_names[i] + '/' + 'node_matrix_' + str(seq_id+1) + '.txt', delimiter=',', header=None)
            info = pd_data.to_numpy()[0]

        
            if self.image_names[i].split("/")[1].split("_")[1] == 'pos':
                l_exist_new[seq_id] = 1
            elif self.image_names[i].split("/")[1].split("_")[1] == 'neg':
                l_exist_new[seq_id] = 0
            else:
                print("error")
                pause

            
            if info[0]>r_resize:
                info[0] = r_resize
            l_radius_new[seq_id] = info[0]/r_resize

            l_vector1_new[seq_id] = info[1]
            l_vector2_new[seq_id] = info[2]
            l_vector3_new[seq_id] = info[3]
        
            l_vector1_bin_new[seq_id] = info[4]
            l_vector2_bin_new[seq_id] = info[5]
            l_vector3_bin_new[seq_id] = info[6]

        
        
        seed = np.random.randint(0, 2**32) # make a seed with numpy generator
        random.seed(seed) 
        torch.manual_seed(seed)
        
        node_img = torch.from_numpy(i_new).float()
        node_lab = torch.from_numpy(i_new_lab).float()
        node_dis = torch.from_numpy(i_new_dis).float()

        
        # apply this seed to img tranfsorms
        if self.tx_total:
            node_img, node_lab, node_dis = self.tx_total(node_img,node_lab,node_dis)


        node_label_radius = self.lx_radius(l_radius_new.astype(np.float32))
        node_label_exist = self.lx_exist(l_exist_new.astype(np.long))
        node_label_vector1 = self.lx_vector1(l_vector1_new.astype(np.float32))
        node_label_vector2 = self.lx_vector2(l_vector2_new.astype(np.float32))
        node_label_vector3 = self.lx_vector3(l_vector3_new.astype(np.float32))
        node_label_vector1_bin = self.lx_vector1_bin(l_vector1_bin_new.astype(np.long))
        node_label_vector2_bin = self.lx_vector2_bin(l_vector2_bin_new.astype(np.long))
        node_label_vector3_bin = self.lx_vector3_bin(l_vector3_bin_new.astype(np.long))
        
        end_time = time.time()


        return node_img, node_lab, node_dis, node_label_radius, node_label_exist, node_label_vector1, node_label_vector2, node_label_vector3, node_label_vector1_bin, node_label_vector2_bin, node_label_vector3_bin
        
