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
import copy
from tools.Image_Tools import gen_circle_2d, gen_circle_gaussian_2d
import time
import pandas as pd
from tools.dataset import RandomCrop, RandomFlip_LR, RandomFlip_UD, RandomRotate, Compose

from configs import config_2d

args = config_2d.args
resize_radio = args.resize_radio
r_resize = args.r_resize
data_shape = args.data_shape


class Images_Dataset_folder_2d(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, images_dir_list, labels_dir_list, mode):
        self.image_names = sorted(images_dir_list)
        self.labels = sorted(labels_dir_list)
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.tx = torchvision.transforms.Compose([])
        self.tx_lab = torchvision.transforms.Compose([])
        self.tx_dis = torchvision.transforms.Compose([])

        self.tx_total = None
        if mode == 'train':
            self.tx_total = Compose([
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
            RandomRotate()
            ])
        

        self.tx1 = torchvision.transforms.Compose([])

            
        self.lx = torchvision.transforms.Compose([])
        
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
        height, width = data_shape
        
        
        img_shape = i_img.shape        

        i_new = np.zeros([seq_len, 3, height, width],dtype=np.float32)
        i_new_lab = np.zeros([seq_len, 1, height, width],dtype=np.float32)
        i_new_dis = np.zeros([seq_len, 1, height, width],dtype=np.float32)
        
        l_radius_new = np.zeros([seq_len])
        l_exist_new = np.zeros([seq_len])

        l_vector1_new = np.zeros([seq_len])
        l_vector2_new = np.zeros([seq_len])
        l_vector1_bin_new = np.zeros([seq_len])
        l_vector2_bin_new = np.zeros([seq_len])
        
        
        
        for seq_id in range(seq_len):
            i1 = copy.deepcopy(i_img[seq_id*3]) / 255
            i1 = np.transpose(i1, (2, 0, 1))

            i_lab1 = copy.deepcopy(i_img[seq_id*3+1]//255)
            i_dis1 = copy.deepcopy(i_img[seq_id*3+2]//255)
            
            
            i_new[seq_id] = copy.deepcopy(i1)
            i_new_lab[seq_id][0] = copy.deepcopy(i_lab1[:,:,0])
            i_new_dis[seq_id][0] = copy.deepcopy(i_dis1[:,:,0])
            # print(i_new.shape, i_new_lab.shape, i_new_dis.shape)
            
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

        
            l_vector1_bin_new[seq_id] = info[3]
            l_vector2_bin_new[seq_id] = info[4]

            # l_vector1_bin_new[seq_id][1] = info[5]
            # l_vector2_bin_new[seq_id][1] = info[6]

        # pause

        seed = np.random.randint(0, 2**32) # make a seed with numpy generator
        random.seed(seed) 
        torch.manual_seed(seed)
        
        node_img = torch.from_numpy(i_new).float()
        node_lab = torch.from_numpy(i_new_lab).float()
        node_dis = torch.from_numpy(i_new_dis).float()

        if self.tx_total:
            node_img, node_lab, node_dis = self.tx_total(node_img,node_lab,node_dis)
        

        node_label_radius = torch.from_numpy(l_radius_new).float()
        node_label_exist = torch.from_numpy(l_exist_new).long()

        node_label_vector1 = torch.from_numpy(l_vector1_new).float()
        node_label_vector2 = torch.from_numpy(l_vector2_new).float()
        
        node_label_vector1_bin = torch.from_numpy(l_vector1_bin_new).long()
        node_label_vector2_bin = torch.from_numpy(l_vector2_bin_new).long()
        
        
        return node_img, node_lab, node_dis, node_label_radius, node_label_exist, node_label_vector1, node_label_vector2, node_label_vector1_bin, node_label_vector2_bin

