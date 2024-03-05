from __future__ import print_function, division
import os
from os import path
import numpy as np
from PIL import Image
import glob
from rtree import index
import time
import shutil
import random
import pickle
import copy

import torch
from torch import optim
import torch.utils.data
import torch.nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist
import torchvision
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from skimage import morphology, transform

from configs import config_2d
from models.Models_2D import CSFL_Net_2D
from tools.tracing.tracing_tools_2D import get_pos_image_2d, get_network_predict_2d, tracing_strategy_fast_2d, tracing_strategy_lstm_2d
from tools.Data_Loader_2d import Images_Dataset_folder_2d
from tools.Losses import dice_loss, MSE_loss, L1_loss, dice_score, bce_loss_w

from lib.klib.baseio import *
from lib.swclib.swc_io import swc_save_metric, swc_save, read_swc_tree
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.euclidean_point import EuclideanPoint
from lib.swclib.edge_match_utils import get_bounds
from lib.swclib import edge_match_utils
from lib.swclib.re_sample import up_sample_swc_tree



args = config_2d.args
resize_radio = args.resize_radio
r_resize = args.r_resize


def train(args, model_name, device_ids, device):
    
    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    valid_size = args.valid_rate
    epoch = args.epochs
    initial_lr = args.lr
    num_workers = args.n_threads
    data_shape = args.data_shape

    to_restore = args.to_restore
    vector_bins = args.vector_bins

    train_seg = args.train_seg

    if train_seg:
        datasets_plag = 'train'
        freeze_plag = False
    else:
        datasets_plag = 'train_r'
        freeze_plag = True

    if train_seg:
        lambda_seg = 1
        lambda_centerline = 1
        lambda_r = 0
        lambda_d_class = 0
        lambda_d_reg = 0
    else:
        lambda_seg = 0
        lambda_centerline = 0
        lambda_r = 100
        lambda_d_class = 1
        lambda_d_reg = 0.001

    random_seed = random.randint(1, 100)
    shuffle = True
    lossT = []
    lossL = []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch-2
    i_valid = 0

    train_on_gpu = torch.cuda.is_available()
    pin_memory = False
    if train_on_gpu:
        pin_memory = True
    #######################################################
    #               load the data
    #######################################################
    print('loading the train data')
    train_data_dir = args.dataset_img_path
    train_label_dir = args.dataset_img_path
    
    train_data_dir_list = []
    train_label_dir_list = []

    train_data_imagename_dir_list = os.listdir(train_data_dir)

    ssss = 0
    for imagename in train_data_imagename_dir_list:
        train_data_imagename_num_dir_list = os.listdir(train_data_dir + imagename)
        for imagename_num in train_data_imagename_num_dir_list:
            train_data_dir_list.append(imagename + '/' + imagename_num)
            train_label_dir_list.append(imagename + '/' + imagename_num)
            ssss += 1

    Training_Data = Images_Dataset_folder_2d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, mode = datasets_plag)

    num_train = len(Training_Data)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,num_workers=num_workers, pin_memory=pin_memory,)
    valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,num_workers=num_workers, pin_memory=pin_memory,)

    #######################################################
    #               load the test data
    #######################################################
    print('loading the test data')
    train_data_dir = args.dataset_img_test_path
    train_label_dir = args.dataset_img_test_path
    
    train_data_dir_list = []
    train_label_dir_list = []

    train_data_imagename_dir_list = os.listdir(train_data_dir)

    for imagename in train_data_imagename_dir_list:
        train_data_imagename_num_dir_list = os.listdir(train_data_dir + imagename)
        for imagename_num in train_data_imagename_num_dir_list:
            train_data_dir_list.append(imagename + '/' + imagename_num)
            train_label_dir_list.append(imagename + '/' + imagename_num)


    Training_Data = Images_Dataset_folder_2d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, mode = 'test')

    num_train = len(Training_Data)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.5 * num_train))

    test_idx = indices[split:]
    test_sampler = SubsetRandomSampler(test_idx)
    
    test_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=test_sampler,num_workers=num_workers, pin_memory=pin_memory,)

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input, in_channel, out_channel, freeze_net):
        model = model_input(in_channel, out_channel, freeze_net)
        return model

    model_train = model_unet_LSTM(model_name, 3, 1, freeze_plag)
    model_train = torch.nn.DataParallel(model_train, device_ids=device_ids)
    model_train.to(device)

    softmax = torch.nn.Softmax()#.cuda(gpu)
    criterion = torch.nn.CrossEntropyLoss()#.cuda(gpu)
    reg_criterion = torch.nn.MSELoss()#.cuda(gpu)
    bce_criterion = torch.nn.BCELoss()
    sigmoid_layer = torch.nn.Sigmoid()

    NLL_criterion = torch.nn.NLLLoss()


    opt = torch.optim.Adam(model_train.parameters(), lr=initial_lr) # try SGD
    MAX_STEP = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=0)
    
    #######################################################
    #               model and log dir
    #######################################################
    LOG_DIR = str(args.log_save_dir) + str(args.gpu_id) + '/'
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'


    try:
        shutil.rmtree(LOG_DIR)
        print('Model folder there, so deleted for newer one')
        os.mkdir(LOG_DIR)
    except OSError:
        print("Creation of the log directory '%s' failed " % LOG_DIR)
    else:
        print("Successfully created the log directory '%s' " % LOG_DIR)
    writer = SummaryWriter(LOG_DIR)

    try:
        os.mkdir(MODEL_DIR)
    except OSError:
        print("Creation of the model directory '%s' failed " % MODEL_DIR)
    else:
        print("Successfully created the model directory '%s' " % MODEL_DIR)
    
    

    if to_restore:
        print("loading model")
        model_train.load_state_dict(torch.load(MODEL_DIR   + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth'))
    else:
        read_model_path = MODEL_DIR  + str(epoch) + '_' + str(batch_size)
        print(read_model_path)
        if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
            shutil.rmtree(read_model_path)
            print('Model folder there, so deleted for newer one')
        try:
            os.mkdir(read_model_path)
        except OSError:
            print("Creation of the model directory '%s' failed" % read_model_path)
        else:
            print("Successfully created the model directory '%s' " % read_model_path)

    ######################DATA NROM########################

    total_step = train_loader.__len__()


    idx_tensor = [(idx) for idx in range(vector_bins)]
    idx_tensor = torch.autograd.Variable(torch.FloatTensor(idx_tensor)).to(device)
    #=============================================================================
    print("begin training")


    valid_loss_min = np.Inf
    r_loss_min = np.Inf

    n_iter = 1
    global_step_ = 0
    for i in range(epoch):
        train_loss = 0.0
        valid_loss = 0.0
        valid_class_loss = 0.0
        valid_reg_loss = 0.0
        valid_rad_loss = 0.0
        
        since = time.time()
        scheduler.step(i)
        print('learning rate: %f' % (opt.param_groups[0]['lr']))

        train_step_temp = 0
        valid_step_temp = 0
        
        #######################################################
        #                    Training Data
        #######################################################
        model_train.train()

        load_begin_time_1 = time.time()
        for x_img, y_lab, y_dis, y_r, y_exist, y1, y2, y1_bin, y2_bin in train_loader:
            train_begin_time = time.time()
            
            y_exist_rnn = y_exist.view(-1)
            y_r_rnn = y_r.view(-1)
            y1_rnn = y1.view(-1)
            y2_rnn = y2.view(-1)
            y1_bin_rnn = y1_bin.view(-1)
            y2_bin_rnn = y2_bin.view(-1)
            
            exist_id_rnn = np.where(y_exist_rnn==1)
            exist_no_id_rnn = np.where(y_exist_rnn==0)
            
            x_img, y_lab, y_dis = x_img.to(device), y_lab.to(device), y_dis.to(device)
            y_exist_rnn, y1_rnn, y2_rnn, y1_bin_rnn, y2_bin_rnn = y_exist_rnn.to(device), y1_rnn.to(device), y2_rnn.to(device), y1_bin_rnn.to(device), y2_bin_rnn.to(device)
            
            opt.zero_grad()
            y_lab_pred, y_dis_pred, y_d_pred_1_rnn, y_d_pred_2_rnn, y_r_pred_rnn = model_train(x_img)
        
            y_r_pred_rnn = y_r_pred_rnn.view(-1)
            y_d_pred_1_rnn = y_d_pred_1_rnn.view(-1,vector_bins)
            y_d_pred_2_rnn = y_d_pred_2_rnn.view(-1,vector_bins)

            vector1_predicted_softmax_rnn = softmax(y_d_pred_1_rnn)
            vector2_predicted_softmax_rnn = softmax(y_d_pred_2_rnn)

            # Expectation of each bin
            vector_predicted_rnn = torch.zeros([y_d_pred_1_rnn.shape[0],2], dtype=torch.float32)
            vector_predicted_rnn[:,0] = torch.sum(vector1_predicted_softmax_rnn * idx_tensor, 1) * (1/vector_bins) * 2 - 1
            vector_predicted_rnn[:,1] = torch.sum(vector2_predicted_softmax_rnn * idx_tensor, 1) * (1/vector_bins) #* 2 - 1

            y_d_class_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2, vector_bins], dtype=torch.float32)
            y_d_class_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2, vector_bins], dtype=torch.float32)
            y_d_reg_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2], dtype=torch.float32)
            y_d_reg_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2], dtype=torch.float32)
            y_r_exist_rnn = torch.zeros([len(exist_id_rnn[0])], dtype=torch.float32)
            y_r_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0])], dtype=torch.float32)
            
            exist_num = 0
            for exist_id_num in exist_id_rnn[0]:
                onehot_0_0 = y1_bin_rnn[exist_id_num]
                onehot_1_0 = y2_bin_rnn[exist_id_num]
                y_d_class_exist_rnn[exist_num][0][onehot_0_0] = 1
                y_d_class_exist_rnn[exist_num][1][onehot_1_0] = 1
                
                y_d_class_pred_exist_rnn[exist_num][0] = sigmoid_layer(y_d_pred_1_rnn[exist_id_num])
                y_d_class_pred_exist_rnn[exist_num][1] = sigmoid_layer(y_d_pred_2_rnn[exist_id_num])

                y_d_reg_exist_rnn[exist_num][0] = y1_rnn[exist_id_num]
                y_d_reg_exist_rnn[exist_num][1] = y2_rnn[exist_id_num]

                vector_predicted_norm = torch.sqrt(vector_predicted_rnn[exist_id_num][0]**2 + vector_predicted_rnn[exist_id_num][1]**2 + 1e-9)
                y_d_reg_pred_exist_rnn[exist_num][0] = vector_predicted_rnn[exist_id_num][0] / vector_predicted_norm
                y_d_reg_pred_exist_rnn[exist_num][1] = vector_predicted_rnn[exist_id_num][1] / vector_predicted_norm

                y_r_exist_rnn[exist_num] = y_r_rnn[exist_id_num]
                y_r_pred_exist_rnn[exist_num] = y_r_pred_rnn[exist_id_num]

                exist_num += 1
            
            vector_up = torch.sum(torch.multiply(y_d_reg_pred_exist_rnn,y_d_reg_exist_rnn), axis=1)
            vector_down_test = torch.sqrt(torch.multiply(y_d_reg_pred_exist_rnn[:,0],y_d_reg_pred_exist_rnn[:,0]) + torch.multiply(y_d_reg_pred_exist_rnn[:,1],y_d_reg_pred_exist_rnn[:,1])  + 1e-9)
            vector_down_gold = torch.sqrt(torch.multiply(y_d_reg_exist_rnn[:,0],y_d_reg_exist_rnn[:,0]) + torch.multiply(y_d_reg_exist_rnn[:,1],y_d_reg_exist_rnn[:,1]) + 1e-9)

            angle_result = vector_up/(vector_down_test*vector_down_gold + 1e-9)
            angle_result_norm_0 = torch.minimum(angle_result, torch.ones_like(angle_result))
            angle_result_norm_1 = torch.maximum(angle_result_norm_0, -torch.ones_like(angle_result))

            angle_arccos_rnn = torch.arccos(angle_result_norm_1)*180/np.pi
            angle_arccos_gt_rnn = torch.zeros_like(angle_arccos_rnn)

            for ss in range(angle_arccos_rnn.shape[0]):
                if angle_arccos_rnn[ss] > 90:
                    angle_arccos_rnn[ss] = 180 - angle_arccos_rnn[ss]
            
            
            
            # Loss
            loss_img_seg = lambda_seg * bce_loss_w(y_lab_pred, y_lab, 0.8)
            loss_img_centerline = lambda_centerline * bce_loss_w(y_dis_pred, y_dis, 0.9)
            lossdice_seg = dice_score(y_lab_pred, y_lab)
            lossdice_centerline = dice_score(y_dis_pred, y_dis)
            
            loss_img_direction_rnn_class1 = bce_criterion(y_d_class_pred_exist_rnn[:,0,:], y_d_class_exist_rnn[:,0,:])
            loss_img_direction_rnn_class2 = bce_criterion(y_d_class_pred_exist_rnn[:,1,:], y_d_class_exist_rnn[:,1,:])
            loss_img_direction_rnn_class = lambda_d_class * (loss_img_direction_rnn_class1 + loss_img_direction_rnn_class2)/2
            
            loss_img_direction_rnn_reg = lambda_d_reg * MSE_loss(angle_arccos_rnn, angle_arccos_gt_rnn)
            
            loss_img_radius_rnn = lambda_r * MSE_loss(y_r_pred_exist_rnn, y_r_exist_rnn)
            
            lossT = loss_img_seg + loss_img_centerline + loss_img_direction_rnn_class + loss_img_direction_rnn_reg + loss_img_radius_rnn

            train_loss += lossT.item()
            lossT.backward()
            opt.step()


            train_end_time = time.time()
            if (train_step_temp+1) % 20 == 0:
                print("===================================================================================")
                print('Epoch: {}/{} \t Step: {}/{} \t Total Loss: {:.5f} \t Time: {:.5f}/step'.format(i + 1, epoch, train_step_temp, total_step, lossT.item(),  train_end_time-train_begin_time))
                
                if train_seg == False:
                    print('Vector Angle LSTM: {:.5f} \t'.format(torch.mean(angle_arccos_rnn)))
                    print('Radius LSTM Loss: {:.5f} \t Class LSTM Loss: {:.5f} \t Reg LSTM Loss: {:.5f} \t'.format(loss_img_radius_rnn.item(), loss_img_direction_rnn_class.item(), loss_img_direction_rnn_reg.item()))
                else:
                    print('Segmentation Loss: {:.5f} \t F1: {:.5f} \t | Centerline Loss: {:.5f} \t F1: {:.5f} \t'.format(loss_img_seg.item(),lossdice_seg.item(), loss_img_centerline.item(), lossdice_centerline.item()))
                print("-----------------------------------------------------------------------------------")

            train_step_temp += 1
            writer.add_scalar('Training Segmentation Loss', loss_img_seg.item(), global_step=global_step_)
            writer.add_scalar('Training Centerline Loss', loss_img_centerline.item(), global_step=global_step_)
            
            writer.add_scalar('Training Radius Reg LSTM Loss', loss_img_radius_rnn.item(), global_step=global_step_)
            writer.add_scalar('Training Vector Reg LSTM Loss', loss_img_direction_rnn_reg.item(), global_step=global_step_)
            writer.add_scalar('Training Vector Class LSTM Loss', loss_img_direction_rnn_class.item(), global_step=global_step_)
            writer.add_scalar('Training Total Loss', lossT.item(), global_step=global_step_)
            writer.add_scalar('Training Total Loss', lossT.item(), global_step=global_step_)
            global_step_ += 1

            load_begin_time_1 = time.time()


        #######################################################
        #Validation Step
        #######################################################
        model_train.eval()
        torch.no_grad() #to increase the validation process uses less memory

        neg_num = 0
        neg_loss = 0

        pos_num = 0
        pos_loss = 0
        
        seg_acc_list = []
        centerline_acc_list = []

        class_acc_list0 = []
        class_acc_list1 = []
        class_acc_list2 = []
        class_acc_list3 = []
        angle_rnn_list = []

        for x_img, y_lab, y_dis, y_r, y_exist, y1, y2, y1_bin, y2_bin in test_loader:
            train_begin_time = time.time()

            y_exist_rnn = y_exist.view(-1)
            y_r_rnn = y_r.view(-1)
            y1_rnn = y1.view(-1)
            y2_rnn = y2.view(-1)
            
            y1_bin_rnn = y1_bin.view(-1)
            y2_bin_rnn = y2_bin.view(-1)
            
            exist_id_rnn = np.where(y_exist_rnn==1)
            
            x_img, y_lab, y_dis = x_img.to(device), y_lab.to(device), y_dis.to(device)
            y_exist_rnn, y1_rnn, y2_rnn, y1_bin_rnn, y2_bin_rnn = y_exist_rnn.to(device), y1_rnn.to(device), y2_rnn.to(device), y1_bin_rnn.to(device), y2_bin_rnn.to(device)
            
            opt.zero_grad()

            y_lab_pred, y_dis_pred, y_d_pred_1_rnn, y_d_pred_2_rnn, y_r_pred_rnn = model_train(x_img)
                        
            y_r_pred_rnn = y_r_pred_rnn.view(-1)
            y_d_pred_1_rnn = y_d_pred_1_rnn.view(-1,vector_bins)
            y_d_pred_2_rnn = y_d_pred_2_rnn.view(-1,vector_bins)
            
            vector1_predicted_softmax_rnn = softmax(y_d_pred_1_rnn)
            vector2_predicted_softmax_rnn = softmax(y_d_pred_2_rnn)


            vector_predicted_rnn = torch.zeros([y_d_pred_1_rnn.shape[0],2], dtype=torch.float32)
            vector_predicted_rnn[:,0] = torch.sum(vector1_predicted_softmax_rnn * idx_tensor, 1) * (1/vector_bins) * 2 - 1
            vector_predicted_rnn[:,1] = torch.sum(vector2_predicted_softmax_rnn * idx_tensor, 1) * (1/vector_bins) #* 2 - 1


            y_d_class_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2, vector_bins], dtype=torch.float32)
            y_d_class_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2, vector_bins], dtype=torch.float32)
            y_d_reg_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2], dtype=torch.float32)
            y_d_reg_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0]), 2], dtype=torch.float32)
            y_r_exist_rnn = torch.zeros([len(exist_id_rnn[0])], dtype=torch.float32)
            y_r_pred_exist_rnn = torch.zeros([len(exist_id_rnn[0])], dtype=torch.float32)
            
            exist_num = 0
            for exist_id_num in exist_id_rnn[0]:
                onehot_0_0 = y1_bin_rnn[exist_id_num]
                onehot_1_0 = y2_bin_rnn[exist_id_num]
                y_d_class_exist_rnn[exist_num][0][onehot_0_0] = 1
                y_d_class_exist_rnn[exist_num][1][onehot_1_0] = 1
                
                y_d_class_pred_exist_rnn[exist_num][0] = sigmoid_layer(y_d_pred_1_rnn[exist_id_num])
                y_d_class_pred_exist_rnn[exist_num][1] = sigmoid_layer(y_d_pred_2_rnn[exist_id_num])

                y_d_reg_exist_rnn[exist_num][0] = y1_rnn[exist_id_num]
                y_d_reg_exist_rnn[exist_num][1] = y2_rnn[exist_id_num]

                vector_predicted_norm = torch.sqrt(vector_predicted_rnn[exist_id_num][0]**2 + vector_predicted_rnn[exist_id_num][1]**2 + 1e-9)
                y_d_reg_pred_exist_rnn[exist_num][0] = vector_predicted_rnn[exist_id_num][0] / vector_predicted_norm
                y_d_reg_pred_exist_rnn[exist_num][1] = vector_predicted_rnn[exist_id_num][1] / vector_predicted_norm

                y_r_exist_rnn[exist_num] = y_r_rnn[exist_id_num]
                y_r_pred_exist_rnn[exist_num] = y_r_pred_rnn[exist_id_num]

                exist_num += 1
                
            
            vector_up = torch.sum(torch.multiply(y_d_reg_pred_exist_rnn,y_d_reg_exist_rnn), axis=1)
            vector_down_test = torch.sqrt(torch.multiply(y_d_reg_pred_exist_rnn[:,0],y_d_reg_pred_exist_rnn[:,0]) + torch.multiply(y_d_reg_pred_exist_rnn[:,1],y_d_reg_pred_exist_rnn[:,1]) + 1e-9)
            vector_down_gold = torch.sqrt(torch.multiply(y_d_reg_exist_rnn[:,0],y_d_reg_exist_rnn[:,0]) + torch.multiply(y_d_reg_exist_rnn[:,1],y_d_reg_exist_rnn[:,1])  + 1e-9)

            angle_result = vector_up/(vector_down_test*vector_down_gold + 1e-9)
            angle_result_norm_0 = torch.minimum(angle_result, torch.ones_like(angle_result))
            angle_result_norm_1 = torch.maximum(angle_result_norm_0, -torch.ones_like(angle_result))

            angle_arccos_rnn = torch.arccos(angle_result_norm_1)*180/np.pi
            angle_arccos_gt_rnn = torch.zeros_like(angle_arccos_rnn)

            for ss in range(angle_arccos_rnn.shape[0]):
                if angle_arccos_rnn[ss] > 90:
                    angle_arccos_rnn[ss] = 180 - angle_arccos_rnn[ss]
            angle_rnn_list.append(torch.mean(angle_arccos_rnn).cpu().detach().numpy())


            # Loss            
            loss_img_seg = lambda_seg * bce_loss_w(y_lab_pred, y_lab, 0.8)
            loss_img_centerline = lambda_centerline *  bce_loss_w(y_dis_pred, y_dis, 0.9)

            loss_img_direction_rnn_class1 = bce_criterion(y_d_class_pred_exist_rnn[:,0,:], y_d_class_exist_rnn[:,0,:])
            loss_img_direction_rnn_class2 = bce_criterion(y_d_class_pred_exist_rnn[:,1,:], y_d_class_exist_rnn[:,1,:])
            loss_img_direction_rnn_class = lambda_d_class * (loss_img_direction_rnn_class1 + loss_img_direction_rnn_class2)/2

            loss_img_direction_rnn_reg = lambda_d_reg * MSE_loss(angle_arccos_rnn, angle_arccos_gt_rnn)

            loss_img_radius_rnn = lambda_r * MSE_loss(y_r_pred_exist_rnn, y_r_exist_rnn)

            lossL = loss_img_seg + loss_img_centerline + loss_img_direction_rnn_class + loss_img_direction_rnn_reg + loss_img_radius_rnn 

            valid_loss += lossL.item()
            valid_class_loss += loss_img_direction_rnn_class.item()
            valid_reg_loss += loss_img_direction_rnn_reg.item()

            valid_step_temp += 1

            # =========================================================================================

            # seg centerline 计算误差
            lossdice_seg = dice_score(y_lab_pred, y_lab)
            seg_acc_list.append(lossdice_seg.item())
            
            lossdice_centerline = dice_score(y_dis_pred, y_dis)
            centerline_acc_list.append(lossdice_centerline.item())

            # R的计算误差
            y_r1_cpu = y_r_exist_rnn.cpu().detach().numpy()
            y_pred_r_cpu = y_r_pred_exist_rnn.cpu().detach().numpy()
            for s in range(y_pred_r_cpu.shape[0]):
                s_temp = y_r1_cpu[s] - y_pred_r_cpu[s]
                if s_temp < 0:
                    neg_num += 1
                    neg_loss += abs(s_temp)*r_resize
                else:
                    pos_num += 1
                    pos_loss += abs(s_temp)*r_resize
            

            # 分类和角度的计算误差
            # # 分类误差
            y_pred1_cpu = y_d_class_pred_exist_rnn[:,0,:].cpu().detach().numpy()
            y_pred2_cpu = y_d_class_pred_exist_rnn[:,1,:].cpu().detach().numpy()

            y1_bin1_cpu = y_d_class_exist_rnn[:,0,:].cpu().detach().numpy()
            y2_bin1_cpu = y_d_class_exist_rnn[:,1,:].cpu().detach().numpy()

            pred1 = y_pred1_cpu.argsort(axis=1)[:,-1:]  # 获取最大值位置
            pred2 = y_pred2_cpu.argsort(axis=1)[:,-1:]  # 获取最大值位置

            y1_bin1_cpu_min = np.min(y1_bin1_cpu, axis=1)
            y2_bin1_cpu_min = np.min(y2_bin1_cpu, axis=1)

            for ss in range(pred1.shape[0]):
                if pred1[ss][0]>=vector_bins//2:
                    pred1[ss][0] = vector_bins - 1 - pred1[ss][0]
                if pred2[ss][0]>=vector_bins//2:
                    pred2[ss][0] = vector_bins - 1 - pred2[ss][0]

            cur_acc1 = np.mean(np.equal(pred1, y1_bin1_cpu_min)*np.ones_like(pred1))
            cur_acc2 = np.mean(np.equal(pred2, y2_bin1_cpu_min)*np.ones_like(pred2))

            class_acc_list1.append(cur_acc1)
            class_acc_list2.append(cur_acc2)


            
        print("============================================")
        print('total num: %d' % (split))
        print("--------------------------------------------")
        print('分割平均准确率:', np.mean(seg_acc_list))
        print('中线平均准确率:', np.mean(centerline_acc_list))
        # R的计算误差
        print("--------------------------------------------")
        print('neg_num: %d neg_loss:%f' % (neg_num, neg_loss/(neg_num+1e-7)))
        print('pos_num: %d pos_loss:%f' % (pos_num, pos_loss/(pos_num+1e-7)))
        print('average_loss:%f' % ((neg_loss+pos_loss)/(neg_num+pos_num+1e-7)))
        # 分类的计算误差
        print("--------------------------------------------")
        # print('越界平均准确率:', np.mean(class_acc_list0))
        # 角度的计算误差
        print("--------------------------------------------")
        print('acc1平均准确率:', np.mean(class_acc_list1))
        print('acc2平均准确率:', np.mean(class_acc_list2))
        print('平均偏差角度:', valid_reg_loss / valid_step_temp)
        print("============================================")

        writer.add_scalar('Test Loss', lossL.item(), global_step=i)


        #######################################################
        #To write in Tensorboard
        #######################################################

        train_loss = train_loss / train_step_temp
        valid_loss = valid_loss / valid_step_temp
        valid_class_loss = valid_class_loss / valid_step_temp
        valid_reg_loss = valid_reg_loss / valid_step_temp

        if (i+1) % 1 == 0:
            print('Epoch: {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f} \t Class Loss: {:.6f} \t Reg Loss: {:.6f}'.format(i + 1, epoch, train_loss, valid_loss, valid_class_loss, valid_reg_loss))

        #######################################################
        #Early Stopping
        #######################################################
        

        if train_seg:
            r_loss = 2 - np.mean(seg_acc_list) - np.mean(centerline_acc_list)
        else:
            r_loss = valid_loss

        torch.save(model_train.state_dict(), MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')

        if r_loss <= r_loss_min: 
            print('R loss decreased (%6f --> %6f).  Saving model ' % (r_loss_min, r_loss))
            torch.save(model_train.state_dict(), MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '_best.pth')
            r_loss_min = r_loss

        time_elapsed = time.time() - since
        print('this epoch time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        n_iter += 1


def inference_segmentation(args, model_name, device_ids, device):
    predict_centerline_path = args.predict_centerline_path
    if not os.path.exists(predict_centerline_path):
        os.makedirs(predict_centerline_path)
    

    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    epoch = args.epochs
    num_workers = args.n_threads
    vector_bins = args.vector_bins
    data_shape = args.data_shape

    resize_radio = args.resize_radio

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input, in_channel=1, out_channel=64):
        model = model_input(in_channel, out_channel)
        return model

    model_test = model_unet_LSTM(model_name, 3, 1)
    model_test = torch.nn.DataParallel(model_test, device_ids=device_ids)
    model_test.to(device)

    #######################################################
    #               load the checkpoint
    #######################################################
    
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'
    checkpoint = torch.load(MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')
    model_test.load_state_dict(checkpoint)
    model_test.eval()

    #######################################################
    #                tracing the image
    #######################################################
    data_transform = torchvision.transforms.Compose([])

    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '*.tif')

    test_patch_height = args.test_patch_height
    test_patch_width  = args.test_patch_height
    stride_height = args.stride_height
    stride_width = args.stride_width

    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]
        img_new = open_tif(test_img_dir).astype(np.float32)
        img_new = transform.resize(img_new,(round(img_new.shape[0]*resize_radio), round(img_new.shape[1]*resize_radio)))

        stack_org = copy.deepcopy(img_new) / 255 

        h, w, _ = stack_org.shape
        h_new = ((h-(test_patch_height-stride_height))//(stride_height)+1)*(stride_height) + (test_patch_height-stride_height)
        w_new = ((w-(test_patch_height-stride_height))//(stride_width)+1)*(stride_width) + (test_patch_height-stride_height)

        # 重调大小
        stack_org_new = -np.ones([h_new, w_new, 3],dtype=np.float32)
        stack_org_new[0:h,0:w, :] = copy.deepcopy(stack_org)

        shape_lab_image = np.zeros([h_new, w_new],dtype=np.float32)
        shape_dis_image = np.zeros([h_new, w_new],dtype=np.float32)
        count_image = np.zeros([h_new, w_new],dtype=np.float32)

        shape_list = []
        SHAPE = [data_shape[0],data_shape[1]]
        batch_list = []
        batch_lab_list = []
        batch_dis_list = []
        cnt = 0


        for j in range(0, h_new-(test_patch_height-stride_height), stride_height):
            for i in range(0, w_new-(test_patch_width-stride_width), stride_width):
                patch = stack_org_new[j:j + SHAPE[0], i:i + SHAPE[1]]
                if len(patch.shape) == 2:
                    patch = patch.squeeze(0)
                else:
                    patch = np.transpose(patch, (2, 0, 1))

                batch_list.append(patch.astype(np.float32))
        
        if len(stack_org.shape) == 2:
            batch_matrix = np.zeros(dtype=np.float32, shape=[len(batch_list), 1, 1, *SHAPE])
        else:
            batch_matrix = np.zeros(dtype=np.float32, shape=[len(batch_list), 1, 3, *SHAPE])

        for i in range(len(batch_list)):
            batch_matrix[i,:,:,:,:] = copy.deepcopy(batch_list[i])
        
        batch_input = data_transform(batch_matrix)
        train_loader = torch.utils.data.DataLoader(batch_input, batch_size=batch_size, num_workers=num_workers)
        
        num=0
        for x_batch in train_loader:
            num += 1
        num_temp = 0

        for x_batch in train_loader:
            print('num:%d / %d '% (num_temp,num))
            num_temp+=1
            batch_input = x_batch.to(device)

            y_lab_pred, y_dis_pred = model_test(batch_input, 'test_dis')

            pred_lab = y_lab_pred.cpu().detach().numpy()
            pred_dis = y_dis_pred.cpu().detach().numpy()
            
            for i in range(pred_lab.shape[0]):
                batch_lab_list.append(pred_lab[i,0,:,:])
                batch_dis_list.append(pred_dis[i,0,:,:])
        
        num = 0
        
        for j in range(0, h_new-(test_patch_height-stride_height), stride_height):
            for i in range(0, w_new-(test_patch_width-stride_width), stride_width):
                shape_lab_image[j:j + SHAPE[0], i:i + SHAPE[1]] += batch_lab_list[num] 
                shape_dis_image[j:j + SHAPE[0], i:i + SHAPE[1]] += batch_dis_list[num] 
                count_image[j:j + SHAPE[0], i:i + SHAPE[1]] += 1
                num += 1

        shape_lab_image = shape_lab_image / count_image
        shape_dis_image = shape_dis_image / count_image

        shape_lab_image_new = (shape_lab_image[0:h,0:w])*255
        shape_dis_image_new = (shape_dis_image[0:h,0:w])*255

        
        # 裁剪视网膜图像中圆外的部分
        test_image_mask_root_dir = args.test_data_mask_path
        test_image_mask_dir = test_image_mask_root_dir + image_name + '_mask_resize.tif' # for DRIVE
        # test_image_mask_dir = test_image_mask_root_dir + 'Image_' + test_img_dir.split("/")[-1].split(".")[0] + '.png' # for CHASEDB
        img_mask_org = open_tif(test_image_mask_dir)
        if np.max(img_mask_org)==255:
            img_mask_org = img_mask_org/255
        
        _, h, w = img_mask_org.shape
        img_mask_org = img_mask_org.reshape(h, w)
        selem = morphology.disk(5*resize_radio)
        img_mask_org = morphology.binary_erosion(img_mask_org, selem)
        img_mask_org = img_mask_org.reshape(1, h, w)


        file_newname_lab = predict_centerline_path + image_name + '.pro.lab.tif' # for DRIVE
        file_newname_skl = predict_centerline_path + image_name + '.pro.skl.tif'
        save_tif(shape_lab_image_new * img_mask_org, file_newname_lab, np.uint8)
        save_tif(shape_dis_image_new * img_mask_org, file_newname_skl, np.uint8)


def inference_fastdeepbranchtracer(args, model_name, device_ids, device):
    predict_swc_path = args.predict_swc_path
    if not os.path.exists(predict_swc_path):
        os.makedirs(predict_swc_path)
    
    predict_seed_path = args.predict_seed_path
    predict_centerline_path = args.predict_centerline_path
    # gold_seed_path = args.gold_seed_path

    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    valid_size = args.valid_rate
    epoch = args.epochs
    initial_lr = args.lr
    num_workers = args.n_threads
    # hidden_layer_size = args.hidden_layer_size
    vector_bins = args.vector_bins
    data_shape = args.data_shape


    SHAPE = [data_shape[0],data_shape[1]]

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input, in_channel=1, out_channel=64, vector_bins=50):
        model = model_input(in_channel, out_channel)
        return model

    model_test = model_unet_LSTM(model_name, 3, 1, vector_bins)
    model_test = torch.nn.DataParallel(model_test, device_ids=device_ids)
    model_test.to(device)
    
    #######################################################
    #               load the checkpoint
    #######################################################
    
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'
    checkpoint = torch.load(MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')
    model_test.load_state_dict(checkpoint)
    model_test.eval()
    
    #######################################################
    #                load the data
    #######################################################
    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '*.tif')


    #######################################################
    #                tracing the image
    #######################################################
    torch.multiprocessing.set_start_method('spawn')

    device_info = [model_test, device]
    data_info = [SHAPE, vector_bins]



    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]
        # print(image_name)
        # if image_name != '12':
        #     continue
        if image_name != '07':
            continue
        # if image_name != '02R':
        #     print('pass:', image_name)
        #     continue

        print(test_img_dir)
        stack_img = open_tif(test_img_dir).astype(np.float32)
        stack_img = transform.resize(stack_img,(round(stack_img.shape[0]*resize_radio), round(stack_img.shape[1]*resize_radio)))


        
        
        
        # 输入预测的中心线和边界
        stack_lab_dir = predict_centerline_path + image_name + '.pro.lab.tif' # for chasedb and road
        stack_skl_dir = predict_centerline_path + image_name + '.pro.skl.tif' # for road

        stack_lab = open_tif(stack_lab_dir).astype(np.float32)
        stack_skl = open_tif(stack_skl_dir).astype(np.float32)

        # 预处理骨架
        th = 128 
        stack_skl[stack_skl<th]=0
        stack_skl[stack_skl>=th]=1
        stack_skl = morphology.skeletonize(stack_skl)//255
        
        
        # 设置追踪模式
        # tracing_strategy_flag = 'centerline'
        # tracing_strategy_flag = 'angle'
        tracing_strategy_flag = 'anglecenterline'

        
        # ===============resize the image=====================
        h, w, _ = stack_img.shape

        steps = 8
        h_new = ((h-8)//(steps*7)+1)*(steps*7) + 8
        w_new = ((w-8)//(steps*7)+1)*(steps*7) + 8

        # 重调大小
        stack_img_new = np.zeros([h_new, w_new, 3],dtype=np.float32)
        stack_lab_new = np.zeros([h_new, w_new],dtype=np.uint8)
        stack_skl_new = np.zeros([h_new, w_new],dtype=np.uint8)

        stack_img_new[0:h,0:w,:] = copy.deepcopy(stack_img)
        stack_lab_new[0:h,0:w] = copy.deepcopy(stack_lab)
        stack_skl_new[0:h,0:w] = copy.deepcopy(stack_skl)

        org_img_shape = stack_img_new.shape
        org_img_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1],3])
        org_lab_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])
        org_skl_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])
        
        org_img_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], :] = copy.deepcopy(stack_img_new)
        org_lab_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_lab_new)
        org_skl_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_skl_new)
 

        shape_image = np.zeros([h_new, w_new],dtype=np.float32)
        # =====================================================
        # 输入gold seed point        
        seed_list_temp = [[],[]]
        stack_seed_swc_dir = predict_seed_path + image_name + '.swc' # for drive

        seed_tree = read_swc_tree(stack_seed_swc_dir)
        seed_tree = up_sample_swc_tree(seed_tree, 5.0) # drive chasedb (5) 
        for tn in seed_tree.get_node_list():
            seed_list_temp[0].append(round(tn.get_y()*resize_radio)+SHAPE[0])
            seed_list_temp[1].append(round(tn.get_x()*resize_radio)+SHAPE[1])

        indices = list(range(len(seed_list_temp[0])))
        np.random.shuffle(indices)
        seed_list = [[],[]]
        for i in range(len(seed_list_temp[0])):
            num = indices[i]
            seed_list[0].append(seed_list_temp[0][num])
            seed_list[1].append(seed_list_temp[1][num])


        seed_list_flag = np.ones(len(seed_list[0]))
        print('共有 %d 个种子点' % (seed_list_flag.shape[0]))


        pos_list = []

        test_image = copy.deepcopy(org_img_temp).astype(np.uint8) # 此项供debug时使用

        # build an initial tree
        tree_new = SwcTree()
        tree_new_covered = SwcTree()
        # build rtree and id_edge_dict (copy from pyneval)
        tree_new_idedge_dict = {}
        swc_tree_list = tree_new.get_node_list()
        p = index.Property()
        p.dimension = 3
        tree_new_rtree = index.Index(properties=p)
        for node in swc_tree_list:
            if node.is_virtual() or node.parent.is_virtual():
                continue
            tree_new_rtree.insert(node.get_id(), get_bounds(node, node.parent, extra=node.radius()))
            tree_new_idedge_dict[node.get_id()] = tuple([node, node.parent])    

        test_id = 0

        begin_time_a = time.time()
        for i in range(seed_list_flag.shape[0]):
            begin_time = time.time()
            if i == 0:
                glob_node_id = 1
            else:
                try:
                    glob_node_id = max(tree_new.id_set) + 1
                except:
                    glob_node_id = 1
            
            print("------------------- tracing ----------------------", i + 1, " / ", seed_list_flag.shape[0])

            seed_node_z = 0
            seed_node_x = seed_list[0][i]
            seed_node_y = seed_list[1][i]

            # 此项供debug时使用
            # seed_node_z = 0
            # seed_node_x = round((726) * 1) + SHAPE[0]
            # seed_node_y = round((460) * 1) + SHAPE[1]


            
            # 如果当前位置已经追踪过,则跳过该种子点
            son_node_temp = SwcNode(nid=glob_node_id, ntype=0, center=EuclideanPoint(center=[seed_node_y,seed_node_x,seed_node_z]), radius=2)
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=son_node_temp, id_edge_dict=tree_new_idedge_dict, threshold=2)
            if len(node_temp_list) != 0:
                print("this seed is already traced")
                continue
            

            seed_node_img, seed_node_exist = get_pos_image_2d(org_img_temp, org_lab_temp, [seed_node_z, seed_node_x, seed_node_y], SHAPE)
            seed_node_img = seed_node_img.reshape(1,*SHAPE, 3)
            exist, exist_score, seed_node_vector, seed_node_r = get_network_predict_2d(seed_node_img, seed_node_exist, SHAPE, model_test, device, vector_bins)


            if exist==0:
                print("this seed is not exist")
                continue

            if seed_node_r < 0.3 * resize_radio:
                print('radius:', seed_node_r)
                print("this seed is to small")
                continue

            seed_node_dict = {'node_id': glob_node_id, 'node_z': seed_node_z, 'node_x': seed_node_x, 'node_y': seed_node_y, 'node_r': seed_node_r, 'node_p_id': -1}

            seed_node_dict['z_delta'] = 0
            seed_node_dict['x_delta'] = 0
            seed_node_dict['y_delta'] = 0
            
            
            tree_new.id_set.add(seed_node_dict['node_id'])
            seed_node = SwcNode(nid=seed_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[seed_node_dict['node_y'],seed_node_dict['node_x'],seed_node_dict['node_z']]), radius=round(seed_node_dict['node_r'],3), parent = tree_new._root)
            tree_new.get_node_list(update=True)

            r_tree_info = [tree_new_rtree, tree_new_idedge_dict]

            print("seed node information")
            print(seed_node)
            

            # 开始追踪
            end_tracing = False
            test_image, tree_new, branch_node_list, r_tree_info = tracing_strategy_fast_2d(end_tracing, org_img_temp, org_lab_temp, org_skl_temp, seed_node, seed_node_dict, 0, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)

            end_tracing = False
            test_image, tree_new, branch_node_list, r_tree_info = tracing_strategy_fast_2d(end_tracing, org_img_temp, org_lab_temp, org_skl_temp, seed_node, seed_node_dict, 1, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)

            end_time = time.time()
            print('单种子点', end_time-begin_time)
            tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]




            
            # tree_new.relocation([0,-SHAPE[0],-SHAPE[1]])
            # for node in tree_new.get_node_list():
            #     node.set_z(round(node.get_z()/resize_radio,3))
            #     node.set_x(round(node.get_x()/resize_radio,3))
            #     node.set_y(round(node.get_y()/resize_radio,3))
            #     node.set_r(round(node.radius()/resize_radio,3))
            # test_dir = predict_swc_path + image_name + '.pre_vector.swc'
            # swc_save(tree_new, test_dir)
            # pause
            
        #     if test_id>=len(test_seed):
        #         tree_save = tree_new
        #         tree_save.relocation([0,-SHAPE[0],-SHAPE[1]])
        #         for node in tree_save.get_node_list():
        #             node.set_z(round(node.get_z()/resize_radio,3))
        #             node.set_x(round(node.get_x()/resize_radio,3))
        #             node.set_y(round(node.get_y()/resize_radio,3))
        #             node.set_r(round(node.radius()/resize_radio,3))
        #         test_dir = predict_swc_path + image_name + '.pre_vector_'+ str(test_id) +'.swc'
        #         swc_save(tree_save, test_dir)
        #         pause

        # pause

        
        tree_new.relocation([0,-SHAPE[0],-SHAPE[1]])
        for node in tree_new.get_node_list():
            node.set_z(round(node.get_z()/resize_radio,3))
            node.set_x(round(node.get_x()/resize_radio,3))
            node.set_y(round(node.get_y()/resize_radio,3))
            node.set_r(round(node.radius()/resize_radio,3))


        # test_dir = predict_swc_path + test_img_dir.split('/')[-1].split('_')[0] + '.pre_vector.swc' # for drive
        # test_dir = predict_swc_path + test_img_dir.split('/')[-1].split('.')[0] + '_test.pre_vector.swc' # for chasedb
        test_dir = predict_swc_path + '/' + image_name + '.pre_vector.swc' # for road
        swc_save(tree_new, test_dir)

        end_time_a = time.time()
        print('共用时：', end_time_a-begin_time_a)


def inference_deepbranchtracer(args, model_name, device_ids, device):
    predict_swc_path = args.predict_swc_path
    if not os.path.exists(predict_swc_path):
        os.makedirs(predict_swc_path)
    
    predict_seed_path = args.predict_seed_path
    predict_centerline_path = args.predict_centerline_path
    gold_seed_path = args.gold_seed_path

    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    valid_size = args.valid_rate
    epoch = args.epochs
    initial_lr = args.lr
    num_workers = args.n_threads
    hidden_layer_size = args.hidden_layer_size
    vector_bins = args.vector_bins
    data_shape = args.data_shape


    SHAPE = [data_shape[0],data_shape[1]]

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input, in_channel=1, out_channel=64, vector_bins=50):
        model = model_input(in_channel, out_channel)
        return model

    model_test = model_unet_LSTM(model_name, 3, 1, vector_bins)
    model_test = torch.nn.DataParallel(model_test, device_ids=device_ids)
    model_test.to(device)

    #######################################################
    #               load the checkpoint
    #######################################################
    
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'
    checkpoint = torch.load(MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')
    model_test.load_state_dict(checkpoint)

    #######################################################
    #                load the data
    #######################################################
    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '*.tif')


    #######################################################
    #                tracing the image
    #######################################################
    torch.multiprocessing.set_start_method('spawn')

    device_info = [model_test, device]
    data_info = [SHAPE, vector_bins]

    
    connect_branch = False


    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]
        # print(image_name)
        # if image_name != '08':
        #     continue
        # if image_name == '7':
        #     continue
        
        # if image_name == '04R' or image_name == '02R' or image_name == '03R' or image_name == '01L':
        #     print('pass:', image_name)
        #     continue

        print(test_img_dir)
        stack_img = open_tif(test_img_dir).astype(np.float32)
        # stack_img = img_rgb[:,:,0]*0.299 + img_rgb[:,:,1]*0.587 + img_rgb[:,:,2]*0.114
        stack_img = transform.resize(stack_img,(round(stack_img.shape[0]*resize_radio), round(stack_img.shape[1]*resize_radio)))

        # =========== should be replaced =====================
        # 输入gold seed point
        
        seed_list_temp = [[],[]]
        stack_seed_swc_dir = gold_seed_path + image_name + '.swc' # for drive
        # stack_seed_swc_dir = gold_seed_path + test_img_dir.split('/')[-1].split('.')[0] + '.swc' # for chasedb
        # stack_seed_swc_dir = gold_seed_path + test_img_dir.split('/')[-1].split('.')[0].split('-')[1] + '.swc' # for chasedb

        seed_tree = read_swc_tree(stack_seed_swc_dir)
        seed_tree = up_sample_swc_tree(seed_tree, 5.0) # drive chasedb (5) 
        for tn in seed_tree.get_node_list():
            seed_list_temp[0].append(round(tn.get_y()*resize_radio)+SHAPE[0])
            seed_list_temp[1].append(round(tn.get_x()*resize_radio)+SHAPE[1])
        
        
        # 输入gold 中心线
        # stack_skl_dir = '/4T/liuchao/deepneutracing/deepbranchtracer_2d/DRIVE/temp/centerline/' + test_img_dir.split('/')[-1].split('_')[0] + '_manual1.centerline.tif'
        # stack_skl_dir = predict_centerline_path + test_img_dir.split('/')[-1].split('_')[0] + '_test.pro.skl.tif' # for drive
        stack_lab_dir = predict_centerline_path + image_name + '.pro.lab.tif' # for chasedb and road
        # stack_lab_dir = predict_centerline_path + image_name + '_pro.tif' # for chasedb and road
        stack_skl_dir = predict_centerline_path + image_name + '.pro.skl.tif' # for chasedb and road

        # print(stack_skl_dir)
        # pause
        stack_lab = open_tif(stack_lab_dir).astype(np.float32)
        stack_skl = open_tif(stack_skl_dir).astype(np.float32)
        # stack_lab = stack_lab.reshape(1, stack_lab.shape[0], stack_lab.shape[1]) # for road
        
        
        # this code for cut the edge of segmentation
        test_image_mask_root_dir = args.test_data_mask_path
        test_image_mask_dir = test_image_mask_root_dir + image_name + '_mask_resize.tif' # for DRIVE
        img_mask_org = open_tif(test_image_mask_dir)
        if np.max(img_mask_org)==255:
            img_mask_org = img_mask_org/255
        _, h, w = stack_lab.shape
        selem = morphology.disk(5*resize_radio)
        img_mask_org = img_mask_org.reshape(h, w)
        img_mask_org = morphology.binary_erosion(img_mask_org, selem)
        img_mask_org = img_mask_org.reshape(1, h, w)

        stack_lab = stack_lab*img_mask_org
        stack_skl = stack_skl*img_mask_org
        # print(np.max(stack_lab))
        # print(np.max(stack_skl))        
        # pause
        
        
        
        
        th = 100 # 60 
        stack_skl[stack_skl<th]=0
        stack_skl[stack_skl>=th]=1
        
        stack_skl = morphology.skeletonize(stack_skl)//255
        # stack_skl_dir = '/4T/liuchao/deepneutracing/deepbranchtracer_2d/ROAD/temp/pre_centerline_test/7-skl.tif'
        # save_tif(stack_skl, stack_skl_dir, np.uint8)
        # print(np.max(stack_skl))
        # pause


        # tracing_strategy_flag = 'centerline'
        tracing_strategy_flag = 'anglecenterline'
        # tracing_strategy_flag = 'angle'

        
        # ===============resize the image=====================
        h, w, _ = stack_img.shape

        steps = 8
        h_new = ((h-8)//(steps*7)+1)*(steps*7) + 8
        w_new = ((w-8)//(steps*7)+1)*(steps*7) + 8

        # 重调大小
        stack_img_new = np.zeros([h_new, w_new, 3],dtype=np.float32)
        stack_lab_new = np.zeros([h_new, w_new],dtype=np.uint8)
        stack_skl_new = np.zeros([h_new, w_new],dtype=np.uint8)
        # stack_exist_new = np.zeros([h_new, w_new],dtype=np.uint16)

        stack_img_new[0:h,0:w,:] = copy.deepcopy(stack_img)
        stack_lab_new[0:h,0:w] = copy.deepcopy(stack_lab)
        stack_skl_new[0:h,0:w] = copy.deepcopy(stack_skl)
        # stack_exist_new[0:h,0:w] = copy.deepcopy(stack_exist)

        org_img_shape = stack_img_new.shape
        org_img_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1],3])
        org_lab_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])
        org_skl_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])
        # org_exist_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])

        
        org_img_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], :] = copy.deepcopy(stack_img_new)
        org_lab_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_lab_new)
        org_skl_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_skl_new)
        # org_exist_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_exist_new)


        shape_image = np.zeros([h_new, w_new],dtype=np.float32)
        # =====================================================
        
        # 获取种子点
        # label_temp_centerline = binary_image_skeletonize(org_lab_temp)//255
        # stack_seed_dir = '/4T/liuchao/deepneutracing/deepbranchtracer_2d/DRIVE/temp/pre_seed/' + test_img_dir[len(TEST_DIR):-9] + '_test.seed.tif'
        # save_tif(label_temp_centerline, stack_seed_dir, np.uint8)

        # skl_random_mask = np.random.randint(0,3,size=(org_lab_temp.shape[0],org_lab_temp.shape[1]))
        # skl_random = skl_random_mask * label_temp_centerline
        # skl_random[skl_random!=1]=0
        # seed_list_temp = np.where(skl_random == 1)

        indices = list(range(len(seed_list_temp[0])))
        np.random.shuffle(indices)
        seed_list = [[],[]]
        for i in range(len(seed_list_temp[0])):
            num = indices[i]
            seed_list[0].append(seed_list_temp[0][num])
            seed_list[1].append(seed_list_temp[1][num])


        seed_list_flag = np.ones(len(seed_list[0]))
        print('共有 %d 个种子点' % (seed_list_flag.shape[0]))

        # pause
        pos_list = []
        # tracing
        # swc_covered_area = np.zeros_like(org_img_temp)
        test_image = copy.deepcopy(org_img_temp).astype(np.uint8)

        # build tree
        tree_new = SwcTree()
        tree_new_covered = SwcTree()
        # build rtree and id_edge_dict (copy from pyneval)
        tree_new_idedge_dict = {}
        swc_tree_list = tree_new.get_node_list()
        p = index.Property()
        p.dimension = 3
        tree_new_rtree = index.Index(properties=p)
        for node in swc_tree_list:
            if node.is_virtual() or node.parent.is_virtual():
                continue
            tree_new_rtree.insert(node.get_id(), get_bounds(node, node.parent, extra=node.radius()))
            tree_new_idedge_dict[node.get_id()] = tuple([node, node.parent])    

        # glob_node_id = 1


        test_id = 0
        # test_seed = [[81,270], [115,308] ]

        begin_time_a = time.time()
        for i in range(seed_list_flag.shape[0]):
            begin_time = time.time()
            # print(i)
            if i == 0:
                glob_node_id = 1
            else:
                try:
                    glob_node_id = max(tree_new.id_set) + 1
                except:
                    glob_node_id = 1
            
            print("------------------- tracing ----------------------", i + 1, " / ", seed_list_flag.shape[0])

            seed_node_z = 0
            seed_node_x = seed_list[0][i]
            seed_node_y = seed_list[1][i]

            # seed_node_z = 0
            # seed_node_x = round((1270) * resize_radio) + SHAPE[0]
            # seed_node_y = round((1016) * resize_radio) + SHAPE[1]

            # seed_node_z = 0
            # seed_node_x = round((test_seed[test_id][0]) * resize_radio) + SHAPE[0]
            # seed_node_y = round((test_seed[test_id][1]) * resize_radio)  + SHAPE[1]
            # test_id += 1
            # print(seed_node_z, seed_node_x, seed_node_y)
            # pause
            
            # 如果当前位置已经追踪过,则跳过该种子点
            son_node_temp = SwcNode(nid=glob_node_id, ntype=0, center=EuclideanPoint(center=[seed_node_y,seed_node_x,seed_node_z]), radius=2)
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=son_node_temp, id_edge_dict=tree_new_idedge_dict, threshold=2)
            if len(node_temp_list) != 0:
                print("this seed is already traced")
                continue
            

            seed_node_img, seed_node_exist = get_pos_image_2d(org_img_temp, org_lab_temp, [seed_node_z, seed_node_x, seed_node_y], SHAPE)
            seed_node_img = seed_node_img.reshape(1,*SHAPE, 3)
            exist, exist_score, seed_node_vector, seed_node_r = get_network_predict_v3(seed_node_img, seed_node_exist, SHAPE, model_test, device, vector_bins)


            if exist==0:
                print("this seed is not exist")
                continue

            if seed_node_r < 0.3 * resize_radio:
                print('radius:', seed_node_r)
                print("this seed is to small")
                continue

            seed_node_dict = {'node_id': glob_node_id, 'node_z': seed_node_z, 'node_x': seed_node_x, 'node_y': seed_node_y, 'node_r': seed_node_r, 'node_p_id': -1}

            seed_node_dict['z_delta'] = 0
            seed_node_dict['x_delta'] = 0
            seed_node_dict['y_delta'] = 0
            
            
            tree_new.id_set.add(seed_node_dict['node_id'])
            seed_node = SwcNode(nid=seed_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[seed_node_dict['node_y'],seed_node_dict['node_x'],seed_node_dict['node_z']]), radius=round(seed_node_dict['node_r'],3), parent = tree_new._root)
            tree_new.get_node_list(update=True)

            r_tree_info = [tree_new_rtree, tree_new_idedge_dict]

            print("seed node information")
            print(seed_node)
            

            # 开始追踪
            end_tracing = False
            test_image, tree_new, branch_node_list, r_tree_info = tracing_strategy_rnn(end_tracing, org_img_temp, org_lab_temp, org_skl_temp, seed_node, seed_node_dict, 0, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)

            end_tracing = False
            test_image, tree_new, branch_node_list, r_tree_info = tracing_strategy_rnn(end_tracing, org_img_temp, org_lab_temp, org_skl_temp, seed_node, seed_node_dict, 1, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)

            end_time = time.time()
            print('单种子点', end_time-begin_time)
            tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]




            
            # tree_new.relocation([0,-SHAPE[0],-SHAPE[1]])
            # for node in tree_new.get_node_list():
            #     node.set_z(round(node.get_z()/resize_radio,3))
            #     node.set_x(round(node.get_x()/resize_radio,3))
            #     node.set_y(round(node.get_y()/resize_radio,3))
            #     node.set_r(round(node.radius()/resize_radio,3))
            # test_dir = predict_swc_path + image_name + '.pre_vector.swc'
            # swc_save(tree_new, test_dir)
            # pause
            
    

        
        tree_new.relocation([0,-SHAPE[0],-SHAPE[1]])
        for node in tree_new.get_node_list():
            node.set_z(round(node.get_z()/resize_radio,3))
            node.set_x(round(node.get_x()/resize_radio,3))
            node.set_y(round(node.get_y()/resize_radio,3))
            node.set_r(round(node.radius()/resize_radio,3))


        # test_dir = predict_swc_path + test_img_dir.split('/')[-1].split('_')[0] + '.pre_vector.swc' # for drive
        # test_dir = predict_swc_path + test_img_dir.split('/')[-1].split('.')[0] + '_test.pre_vector.swc' # for chasedb
        test_dir = predict_swc_path + 'rnn/' + image_name + '.pre_vector.swc' # for road
        swc_save(tree_new, test_dir)

        end_time_a = time.time()
        print('共用时：', end_time_a-begin_time_a)




if __name__=='__main__':
    #######################################################
    #              load the config of model
    #######################################################
    args = config_2d.args

    #######################################################
    #              Checking if GPU is used
    #######################################################
    device_ids = [0,1,2,3]

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')
    device = torch.device("cuda:" + str(args.gpu_id) if train_on_gpu else "cpu")

    
    #######################################################
    #              Setting up the model
    #######################################################
    model_name = CSFL_Net_2D

    ####### check the model backbone and parameter ########
    # from thop import profile
    # total = sum([param.nelement() for param in model_name(3,1).parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    # input_ = torch.randn(1,3,3,64,64)
    # flops, params = profile(model_name(3,1), {input_,})
    # print("%.2fGFLOPs" % (flops/1e9), "%.2fM" % (params/1e6))

    # from pytorch_model_summary import summary
    # input_ = torch.randn(1,3,3,64,64)
    # xss = summary(model_name(3,1), input_, show_input=True, show_hierarchical=False)
    # print(xss)

    # for name, param in model_name(3,1, freeze_net=False).named_parameters():
    #     print(name, param.requires_grad)
    #######################################################

    print('================================================')
    print('  status   = ' + str(args.train_or_test))
    print('  gpu_id   = ' + str(args.gpu_id))
    print(' img_path  = ' + str(args.dataset_img_path))
    print(' img_t_path= ' + str(args.dataset_img_test_path))
    print('model_name = ' + str(model_name))
    print('to_restore = ' + str(args.to_restore))
    print('batch_size = ' + str(args.batch_size))
    print('   epoch   = ' + str(args.epochs))
    print('    bins   = ' + str(args.vector_bins))
    print('================================================')
    
    train_or_test = str(args.train_or_test)

    if train_or_test == 'train':
        train(args, model_name, device_ids, device)
        print('train')
    elif train_or_test == 'inference_segmentation':
        print('inference_segmentation')
        inference_segmentation(args, model_name, device_ids, device)
    elif train_or_test == 'inference_fastdeepbranchtracer':
        print('inference_fastdeepbranchtracer')
        inference_fastdeepbranchtracer(args, model_name, device_ids, device)
    elif train_or_test == 'inference_deepbranchtracer':
        print('inference_deepbranchtracer')
        inference_deepbranchtracer(args, model_name, device_ids, device)
    else:
        print('choose train or test mode!')


# python train_2D.py --gpu_id 0 --train_or_test train --train_seg True
# python train_2D.py --gpu_id 0 --train_or_test train --lr 2e-4 --to_restore True

# python train_2D.py --gpu_id 0 --train_or_test inference_segmentation 
# python train_2D.py --gpu_id 0 --train_or_test inference_fastdeepbranchtracer
# python train_2D.py --gpu_id 0 --train_or_test inference_deepbranchtracer



