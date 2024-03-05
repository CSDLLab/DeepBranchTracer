import numpy as np
import copy
import cv2 as cv
import multiprocessing as mp
from skimage import morphology, transform
import queue
import time
import math
import random
import tifffile
# import GeodisTK
import os
from shutil import rmtree
from PIL import Image
import sys
import argparse

from scipy import ndimage
from scipy.ndimage import filters as ndfilter

from lib.klib.baseio import *
from lib.klib.glib.DrawSimulationSWCModel import simulate3DTreeModel_dendrite, save_swc

from lib.swclib.swc_io import swc_save, swc_save_preorder, read_swc_tree, read_swc_tree_matrix, swc_save_metric
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.re_sample import up_sample_swc_tree, down_sample_swc_tree
from lib.swclib import euclidean_point, edge_match_utils, point_match_utils


sys.setrecursionlimit(100000)

# python prepare_datasets_2D.py --datasets_name DRIVE --train_dataset_root_dir /4T/liuchao/deepneutracing/deepbranchtracer_2d/DRIVE/training_data/
# python prepare_datasets_2D.py --datasets_name CHASEDB1 --train_dataset_root_dir /4T/liuchao/deepneutracing/deepbranchtracer_2d/CHASEDB1/training_data/ 
# python prepare_datasets_2D.py --datasets_name ROAD --train_dataset_root_dir /4T/liuchao/deepneutracing/deepbranchtracer_2d/ROAD/training_data/ 

def parse_args():
	parser = argparse.ArgumentParser()

	# (input dir) orginal data
	parser.add_argument('--datasets_name', default='DRIVE',help='datasets name') # CHASEDB1
	parser.add_argument('--image_dir', default='/4T/liuchao/deepneutracing/deepbranchtracer_2d/', help='orginal image saved here')
	
	# (output dir)
	parser.add_argument('--train_dataset_root_dir', default='/4T/liuchao/deepneutracing/deepbranchtracer_2d/DRIVE/training_data/',help='orginal centerline saved here')
	parser.add_argument('--N_patches', default=80000,help='Number of training image patches') # 150000

	parser.add_argument('--input_dim', type=int, default=(1,64,64))
	parser.add_argument('--multi_cpu', type=int, default=10)

	args = parser.parse_args()
	
	return args


# def o_distance(point_A,point_B):
#     distance = math.sqrt((point_A[0][0]-point_B[0][0])**2 +(point_A[0][1]-point_B[0][1])**2)
#     return distance

# def find_centerline_flag_0(centerline_flag):
#     point_num = centerline_flag.shape[0]
#     for i in range(point_num):
#         if centerline_flag[i]==0:
#             return False, i
#     return True, 0

# def down_sample_swc(swc_dir):
# 	swc_tree = read_swc_tree(swc_dir)
# 	swc_tree_downsample = down_sample_swc_tree(swc_tree)
# 	swc_tree_downsample.sort_node_list(key="default")

# 	return swc_tree_downsample

def vector_norm_f(vector_org):
	vector0 = vector_org[0]
	vector1 = vector_org[1]

	vector_norm = [0,0]
	for i in range(2):
		vector_norm[i] = vector_org[i] / np.sqrt(vector0 ** 2 + vector1 ** 2 )
	return vector_norm

def up_sample_swc_rescale(swc_dir, length_threshold, resize_radio):
	swc_tree_upsample = read_swc_tree(swc_dir)
	swc_tree_upsample.rescale([resize_radio,resize_radio,resize_radio,resize_radio])
	swc_tree_upsample = up_sample_swc_tree(swc_tree_upsample, length_threshold)
	swc_tree_upsample.sort_node_list(key="default")

	return swc_tree_upsample


def get_centerline_swc(swc_dir, r):
	swc_tree_centerline = read_swc_tree(swc_dir)
	swc_node_list = swc_tree_centerline.get_node_list()
	for node in swc_node_list:
	    if node.is_virtual():
	        continue
	    node.set_r(r=r)
	return swc_tree_centerline

def get_centerline_direction_2d(resample_tree_data):
	centerline_direction = np.zeros([resample_tree_data.shape[0], 2], dtype=np.float32)
	for i in range(resample_tree_data.shape[0]):
		node_id = resample_tree_data[i][0]
		node_id_p = resample_tree_data[i][6]

		if node_id_p == -1:  # parent 是否为root
			if node_id != resample_tree_data.shape[0]:
				node_A = int(node_id) - 1
				node_B = int(node_id + 1) - 1
			else:
				node_A = int(node_id) - 1
				node_B = int(node_id) - 1
		else:
			node_son = int(node_id + 1) - 1

			if node_id != resample_tree_data.shape[0]:
				if resample_tree_data[node_son][6] == node_id:
					node_A = int(node_id) - 1
					node_B = int(node_id_p) - 1
				else:
					node_A = int(node_id) - 1
					node_B = int(node_id_p) - 1
			else:
				node_A = int(node_id) - 1
				node_B = int(node_id_p) - 1

		centerline_direction[i][0] = resample_tree_data[node_A][3] - resample_tree_data[node_B][3]
		centerline_direction[i][1] = resample_tree_data[node_A][2] - resample_tree_data[node_B][2]

	return centerline_direction

def get_centerline_circle_2d(centerline_sample_A, centerline_direction, r=1.0):
	circle_num = centerline_sample_A.shape[0]

	theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

	circle_x = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_y = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_z = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)

	circle_vector = np.zeros([circle_num, 2], dtype=np.float32)
	# circle_angle = np.zeros([circle_num, 2], dtype=np.float32)

	u_vector = np.zeros([circle_num, 2], dtype=np.float32)

	for i in range(circle_num):
		# 获取centerline, norm_vector
		vector_x = centerline_direction[i][0]
		vector_y = centerline_direction[i][1]

		circle_vector[i][0] = vector_y / np.sqrt(vector_x ** 2 + vector_y ** 2)
		circle_vector[i][1] = vector_x / np.sqrt(vector_x ** 2 + vector_y ** 2)

		##############################################
	
		norm_vector_x = centerline_direction[i][0]
		norm_vector_y = centerline_direction[i][1]

		# 获取法向量u
		u_x = - norm_vector_y
		u_y = norm_vector_x


		# 获取u,v的单位向量
		u_n = math.sqrt(u_x ** 2 + u_y ** 2 + 1e-5)
		u_x_tilde = u_x / u_n
		u_y_tilde = u_y / u_n

		u_vector[i][0] = u_x_tilde
		u_vector[i][1] = u_y_tilde

	return circle_vector, u_vector


def prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_sim, label_sim, img_mask, img_skl_distance, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector,swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, PATCH_NUM, seq_len=1, img_gap=1):
	
	image_seq_dir = image_seq_dir + image_name + '/'
	
	if not os.path.exists(image_seq_dir):
		os.mkdir(image_seq_dir)

	img_shape = img_sim.shape
	img_sim_temp = np.zeros([1, img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2], 3])
	img_lab_temp = np.zeros([1, img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2], 3])
	img_mask_temp = np.zeros([1, img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2], 3])
	img_dis_temp = np.zeros([1, img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2], 3])

	z_half = 1
	x_half = BATCH_SHAPE[1]//2
	y_half = BATCH_SHAPE[2]//2

	img_sim_temp[0, 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2], :] = copy.deepcopy(img_sim)
	for i in range(3):	
		img_lab_temp[0, 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2], i] = copy.deepcopy(label_sim)
		img_mask_temp[0, 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2], i] = copy.deepcopy(img_mask)
		img_dis_temp[0, 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2], i] = copy.deepcopy(img_skl_distance)


	swc_tree_node_flag = np.zeros([swc_tree_centerline_matirx.shape[0]])

	# 提取每条枝干的node id
	branch_list = []
	for i in range(swc_tree_centerline_matirx.shape[0]):
		node_id_temp = swc_tree_centerline_matirx.shape[0] - 1 - i
		branch_list_temp = []
		branch_list_temp.append(node_id_temp)

		while node_id_temp != -2 and swc_tree_node_flag[int(node_id_temp)] == 0:
			swc_tree_node_flag[int(node_id_temp)] = 1
			node_id_temp = swc_tree_centerline_matirx[int(node_id_temp)][6] - 1
			branch_list_temp.append(node_id_temp)
		if len(branch_list_temp) > seq_len*2+1:
			branch_list.append(branch_list_temp)
			#print(len(branch_list_temp))
	print('图像id:%s 共有%d条枝干' % (image_name, len(branch_list)))

	
	branch_num = len(branch_list)

	# print(image_seq_id_list)
	for i in range(PATCH_NUM):
		if i%20==0:
			print('图像id:%s 当前第%d组 / 共%d组'% (image_name, i+1, PATCH_NUM))

		branch_id_rand = random.randint(0,branch_num-1)
		branch_temp_list = branch_list[branch_id_rand]
		branch_length = len(branch_temp_list)

		seed_id_rand = random.randint(seq_len,branch_length-1-seq_len)
		node1_id = branch_temp_list[seed_id_rand]
		node2_id = branch_temp_list[seed_id_rand+1]
		node3_id = branch_temp_list[seed_id_rand+2]

		node_id_list = [node1_id, node2_id, node3_id]

		image_seq_single_dir = image_seq_dir + '/' + str(i+1)  + '_pos_0/'
		if os.path.exists(image_seq_single_dir):
			rmtree(image_seq_single_dir)
			os.mkdir(image_seq_single_dir)
		else:
			os.mkdir(image_seq_single_dir)

		image_stack = np.zeros([3*seq_len, BATCH_SHAPE[1], BATCH_SHAPE[2], 3])
		node_img_temp_dir = image_seq_single_dir + 'node_img.tif'

  
		seq_id = 0
		for node_temp_id in node_id_list:
			node_pos_float = swc_tree_centerline_matirx[int(node_temp_id)][2:4]
			node_pos = [int(round(j,0)) for j in node_pos_float]

			node_pos_p_float = swc_tree_centerline_matirx[int(node_temp_id-1)][2:4]
			node_pos_p = [int(round(j,0)) for j in node_pos_p_float]
			node_pos_s_float = swc_tree_centerline_matirx[int(node_temp_id+1)][2:4]
			node_pos_s = [int(round(j,0)) for j in node_pos_s_float]

	
			node_img = copy.deepcopy(img_sim_temp[0, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half, :])
			node_lab = copy.deepcopy(img_lab_temp[0, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])
			node_dis = copy.deepcopy(img_dis_temp[0, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])


			node_radius = swc_tree_centerline_matirx_radius[int(node_temp_id)][0]
			node_vector = node_pos_s_float - node_pos_p_float
			node_vector = vector_norm_f(node_vector)


			node_vector_to_p = node_pos_p_float - node_pos_float
			node_vector_to_p = vector_norm_f(node_vector_to_p)
			node_vector_to_s = node_pos_s_float - node_pos_float
			node_vector_to_s = vector_norm_f(node_vector_to_s)

			image_stack[3*seq_id+0] = copy.deepcopy(node_img)
			image_stack[3*seq_id+1] = copy.deepcopy(node_lab)
			image_stack[3*seq_id+2] = copy.deepcopy(node_dis)
			

			# label
			node_matrix = np.zeros([1,5])
			node_matrix[0][0] = node_radius
		
			for j in range(2):
				node_matrix[0][j+1] = node_vector[j]
		
			# for j in range(2):
			# 	node_matrix[0][3+j] = (node_vector[j]+1)/2//0.02
			# 	node_matrix[0][5+j] = (-node_vector[j]+1)/2//0.02
			if node_vector[1]<0:
				node_matrix[0][3] = (-node_vector[0]+1)/2//0.02
				node_matrix[0][4] = (-node_vector[1])//0.02
			else:
				node_matrix[0][3] = (node_vector[0]+1)/2//0.02
				node_matrix[0][4] = (node_vector[1])//0.02

			node_matrix_temp_dir = image_seq_single_dir + 'node_matrix_' + str(seq_id+1) + '.txt'
			np.savetxt(node_matrix_temp_dir, node_matrix, fmt='%f', delimiter=',')

			# node_swc = np.zeros([5,7])
			# node_swc[0][0] = 1
			# node_swc[1][0] = 2
			# node_swc[2][0] = 3
			# node_swc[3][0] = 4
			# node_swc[4][0] = 5

			# node_swc[0][5] = 1
			# node_swc[1][5] = 1
			# node_swc[2][5] = 1
			# node_swc[3][5] = 1
			# node_swc[4][5] = 1

			# node_swc[0][6] = -1
			# node_swc[1][6] = 1
			# node_swc[2][6] = 1
			# node_swc[3][6] = -1
			# node_swc[4][6] = -1
		
			# node_swc[0][2] = BATCH_SHAPE[1]//2
			# node_swc[0][3] = BATCH_SHAPE[2]//2
			# node_swc[0][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len

			# node_swc[1][2] = BATCH_SHAPE[1]//2 + node_matrix[0][1] * 5
			# node_swc[1][3] = BATCH_SHAPE[2]//2 + node_matrix[0][2] * 5
			# node_swc[1][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len
			# node_swc[2][2] = BATCH_SHAPE[1]//2 - node_matrix[0][1] * 5
			# node_swc[2][3] = BATCH_SHAPE[2]//2 - node_matrix[0][2] * 5
			# node_swc[2][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len

			# node_swc[3][2] = BATCH_SHAPE[1]//2 + node_vector_to_p[0] * 5
			# node_swc[3][3] = BATCH_SHAPE[2]//2 + node_vector_to_p[1] * 5
			# node_swc[3][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len 
			# node_swc[4][2] = BATCH_SHAPE[1]//2 + node_vector_to_s[0] * 5
			# node_swc[4][3] = BATCH_SHAPE[2]//2 + node_vector_to_s[1] * 5
			# node_swc[4][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len 

			# node_matrix_temp_dir = image_seq_single_dir + 'node_swc_' + str(seq_id+1) + '.swc'
			# np.savetxt(node_matrix_temp_dir, node_swc, fmt='%3f', delimiter=',')

			# # node_pos_txt = np.zeros([1,3])
			# # node_pos_txt[0][0] = node_pos[2]
			# # node_pos_txt[0][1] = node_pos[1]
			# # node_pos_txt[0][2] = node_pos[0]
			# # node_matrix_temp_dir = image_seq_single_dir + 'node_pos_' + str(seq_id+1) + '.txt'
			# # np.savetxt(node_matrix_temp_dir, node_pos_txt, fmt='%3f', delimiter=',')

			seq_id+=1

		save_tif(image_stack, node_img_temp_dir, np.uint8)
  

		#===============================neg and pos======================================================

		pos_num = 2
		data_enhance = ['noise', 'shift']
		neg_num = 1
		
		for pos_id in range(pos_num):
			image_seq_single_dir = image_seq_dir + '/' + str(i+1) + '_pos_' + str(pos_id+1) + '/'
			if os.path.exists(image_seq_single_dir):
				rmtree(image_seq_single_dir)
				os.mkdir(image_seq_single_dir)
			else:
				os.mkdir(image_seq_single_dir)
			
			image_stack_pos = np.zeros([3*seq_len, BATCH_SHAPE[1], BATCH_SHAPE[2], 3])
			node_img_temp_dir = image_seq_single_dir + 'node_img.tif'

			# use two different data enhance method
			data_enhance_method = data_enhance[pos_id]
			x_noise = []
			y_noise = []
			if data_enhance_method == 'shift':
				x_rand_temp = random.uniform(-3, 3)
				x_noise.append(x_rand_temp)
				x_noise.append(x_rand_temp)
				x_noise.append(x_rand_temp)

			elif data_enhance_method == 'noise':
				for seq_temp in range(seq_len):
					x_rand_temp = random.uniform(-3, 3)
					x_noise.append(x_rand_temp)


			seq_id = 0
			for node_temp_id in node_id_list:
				node_pos_float = swc_tree_centerline_matirx[int(node_temp_id)][2:4]
				node_pos = [int(round(j,0)) for j in node_pos_float]

				node_pos_p_float = swc_tree_centerline_matirx[int(node_temp_id-1)][2:4]
				node_pos_p = [int(round(j,0)) for j in node_pos_p_float]
				node_pos_s_float = swc_tree_centerline_matirx[int(node_temp_id+1)][2:4]
				node_pos_s = [int(round(j,0)) for j in node_pos_s_float]

				node_radius = swc_tree_centerline_matirx_radius[int(node_temp_id)][0]
				node_vector = node_pos_p_float - node_pos_s_float
				node_vector = vector_norm_f(node_vector)

				node_vector_to_p = node_pos_p_float - node_pos_float
				node_vector_to_p = vector_norm_f(node_vector_to_p)
				node_vector_to_s = node_pos_s_float - node_pos_float
				node_vector_to_s = vector_norm_f(node_vector_to_s)


				node_vector_u = [-node_vector[1], node_vector[0]]
				node_vector_u_n = np.linalg.norm(np.array(node_vector_u))
				node_u = node_vector_u / node_vector_u_n

				x_rand_temp = x_noise[seq_id]

				node_rand_float = [0,0]
				node_rand_float[0] = x_rand_temp * node_u[0] 
				node_rand_float[1] = x_rand_temp * node_u[1] 
				node_rand = [int(round(rad_num,0)) for rad_num in node_rand_float]
				node_pos_float_positive = [a+b for a,b in zip(node_pos_float, node_rand_float)] 
				
    
				node_vector_to_old = node_pos_float_positive - node_pos_float
				node_vector_to_old = vector_norm_f(node_vector_to_old)

				node_img_pos = copy.deepcopy(img_sim_temp[0, node_pos[1]+node_rand[0]+ 3*x_half:node_pos[1]+node_rand[0] + 5*x_half, node_pos[0]+node_rand[1]+ 3*y_half:node_pos[0]+node_rand[1] + 5*y_half, :])
				node_lab_pos = copy.deepcopy(img_lab_temp[0, node_pos[1]+node_rand[0]+ 3*x_half:node_pos[1]+node_rand[0] + 5*x_half, node_pos[0]+node_rand[1]+ 3*y_half:node_pos[0]+node_rand[1] + 5*y_half])
				node_dis_pos = copy.deepcopy(img_dis_temp[0, node_pos[1]+node_rand[0]+ 3*x_half:node_pos[1]+node_rand[0] + 5*x_half, node_pos[0]+node_rand[1]+ 3*y_half:node_pos[0]+node_rand[1] + 5*y_half])
			

				image_stack_pos[3*seq_id+0] = copy.deepcopy(node_img_pos)
				image_stack_pos[3*seq_id+1] = copy.deepcopy(node_lab_pos)
				image_stack_pos[3*seq_id+2] = copy.deepcopy(node_dis_pos)
			
			
				node_matrix = np.zeros([1,5])
				node_matrix[0][0] = node_radius

	
				for j in range(2):
					node_matrix[0][j+1] = node_vector[j]
			
				# for j in range(2):
				# 	node_matrix[0][3+j] = (node_vector[j]+1)/2//0.02
				# 	node_matrix[0][5+j] = (-node_vector[j]+1)/2//0.02

				if node_vector[1]<0:
					node_matrix[0][3] = (-node_vector[0]+1)/2//0.02
					node_matrix[0][4] = (-node_vector[1])//0.02
				else:
					node_matrix[0][3] = (node_vector[0]+1)/2//0.02
					node_matrix[0][4] = (node_vector[1])//0.02
			
				node_matrix_temp_dir = image_seq_single_dir + 'node_matrix_' + str(seq_id+1) + '.txt'
				np.savetxt(node_matrix_temp_dir, node_matrix, fmt='%f', delimiter=',')

				# node_swc = np.zeros([6,7])
				# node_swc[0][0] = 1
				# node_swc[1][0] = 2
				# node_swc[2][0] = 3
				# node_swc[3][0] = 4
				# node_swc[4][0] = 5
				# node_swc[5][0] = 6

				# node_swc[0][5] = 1
				# node_swc[1][5] = 1
				# node_swc[2][5] = 1
				# node_swc[3][5] = 1
				# node_swc[4][5] = 1
				# node_swc[5][5] = 1

				# node_swc[0][6] = -1
				# node_swc[1][6] = 1
				# node_swc[2][6] = 1
				# node_swc[3][6] = -1
				# node_swc[4][6] = -1
				# node_swc[5][6] = -1
		
				# node_swc[0][2] = BATCH_SHAPE[1]//2 
				# node_swc[0][3] = BATCH_SHAPE[2]//2 
				# node_swc[0][4] = BATCH_SHAPE[0]//2 

				# node_swc[1][2] = BATCH_SHAPE[1]//2 + node_matrix[0][1] * 5
				# node_swc[1][3] = BATCH_SHAPE[2]//2 + node_matrix[0][2] * 5
				# node_swc[1][4] = BATCH_SHAPE[0]//2 
				# node_swc[2][2] = BATCH_SHAPE[1]//2 - node_matrix[0][1] * 5
				# node_swc[2][3] = BATCH_SHAPE[2]//2 - node_matrix[0][2] * 5
				# node_swc[2][4] = BATCH_SHAPE[0]//2 

				# node_swc[3][2] = BATCH_SHAPE[1]//2 + node_vector_to_p[0] * 5
				# node_swc[3][3] = BATCH_SHAPE[2]//2 + node_vector_to_p[1] * 5
				# node_swc[3][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len 
				# node_swc[4][2] = BATCH_SHAPE[1]//2 + node_vector_to_s[0] * 5
				# node_swc[4][3] = BATCH_SHAPE[2]//2 + node_vector_to_s[1] * 5
				# node_swc[4][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len 
				# node_swc[5][2] = BATCH_SHAPE[1]//2 + node_vector_to_old[0] * 5
				# node_swc[5][3] = BATCH_SHAPE[2]//2 + node_vector_to_old[1] * 5
				# node_swc[5][4] = BATCH_SHAPE[0]//2 + BATCH_SHAPE[0]*seq_id*seq_len 


				# node_matrix_temp_dir = image_seq_single_dir + 'node_swc_' + str(seq_id+1) + '.swc'
				# np.savetxt(node_matrix_temp_dir, node_swc, fmt='%3f', delimiter=',')

				# node_pos_txt = np.zeros([1,3])
				# node_pos_txt[0][0] = node_pos[2]
				# node_pos_txt[0][1] = node_pos[1]
				# node_pos_txt[0][2] = node_pos[0]
				# node_matrix_temp_dir = image_seq_single_dir + 'node_pos_' + str(seq_id+1) + '.txt'
				# np.savetxt(node_matrix_temp_dir, node_pos_txt, fmt='%3f', delimiter=',')
				seq_id+=1

			save_tif(image_stack_pos, node_img_temp_dir, np.uint8)

	
		for neg_id in range(neg_num):
			image_seq_single_dir = image_seq_dir + '/' + str(i+1) + '_neg_' + str(neg_id+1) + '/'
			if os.path.exists(image_seq_single_dir):
				rmtree(image_seq_single_dir)
				os.mkdir(image_seq_single_dir)
			else:
				os.mkdir(image_seq_single_dir)
			
			image_stack_neg = np.zeros([3*seq_len, BATCH_SHAPE[1], BATCH_SHAPE[2], 3])
			node_img_temp_dir = image_seq_single_dir + 'node_img.tif'

			neg_ok_num = 0
			seq_id = 0
			while neg_ok_num<seq_len:
				# z_rand_temp = random.randint(BATCH_SHAPE[0], img_sim_temp.shape[0]-BATCH_SHAPE[0])
				x_rand_temp = random.randint(BATCH_SHAPE[1], img_sim_temp.shape[1]-BATCH_SHAPE[1])
				y_rand_temp = random.randint(BATCH_SHAPE[2], img_sim_temp.shape[2]-BATCH_SHAPE[2])

				node_img_neg = copy.deepcopy(img_sim_temp[0, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half, :])
				node_lab_neg = copy.deepcopy(img_lab_temp[0, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])
				node_dis_neg = copy.deepcopy(img_dis_temp[0, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])

				image_stack_neg[3*seq_id+0] = copy.deepcopy(node_img_neg)
				image_stack_neg[3*seq_id+1] = copy.deepcopy(node_lab_neg)
				image_stack_neg[3*seq_id+2] = copy.deepcopy(node_dis_neg)

				node_matrix = np.zeros([1,5])
				node_matrix_temp_dir = image_seq_single_dir + 'node_matrix_' + str(seq_id+1) + '.txt'
				np.savetxt(node_matrix_temp_dir, node_matrix, fmt='%f', delimiter=',')

				seq_id+=1
				neg_ok_num += 1

			save_tif(image_stack_neg, node_img_temp_dir, np.uint8)
				
		
	return 0


def main_training_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name
	image_seq_dir = args.train_dataset_root_dir + 'training_datasets/'

	org_image_train_dir = args.image_dir + datasets_name + '/training/images_color/'
	org_label_train_dir = args.image_dir + datasets_name + '/training/labels/'
	org_mask_train_dir = args.image_dir + datasets_name + '/training/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_label_tif_dir = args.image_dir + datasets_name + '/temp/labels/'
	temp_mask_tif_dir = args.image_dir + datasets_name + '/temp/mask/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'

	image_name = input_dir.split("/")[-1].split(".")[0]

	if datasets_name == 'DRIVE':
		patch_num = args.N_patches // 20 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'

		resize_radio = 2.0
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 3
	elif datasets_name == 'CHASEDB1':
		patch_num = args.N_patches // 20 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'

		resize_radio = 1.5
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 3
	elif datasets_name == 'ROAD':
		patch_num = args.N_patches // 804 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'

		resize_radio = 1.0
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 3
	else:
		print("Error")
		return 0


	# 载入灰度图像
	img_resize_dir = temp_image_tif_dir + image_name + '_img_resize.tif'
	img_gray = open_tif(img_dir).astype(np.float32)
	img_new = transform.resize(img_gray,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	img_new = np.expand_dims(img_new,0)
	save_tif(img_new, img_resize_dir, np.uint8)

	# 载入label
	# label_resize_dir = temp_label_tif_dir + image_name + '_label_resize.tif'
	# label_new = open_tif(input_dir).astype(np.float32)
	# label_new = transform.resize(label_new,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	# label_new = np.expand_dims(label_new,0)
	# th = 200
	# label_new[label_new>=th] = 255
	# label_new[label_new<th] = 0
	# save_tif(label_new, label_resize_dir, np.uint8)

	# 载入mask
	mask_resize_dir = temp_mask_tif_dir + image_name + '_mask_resize.tif'
	img_mask = open_tif(img_mask_dir).astype(np.float32)
	img_mask = transform.resize(img_mask,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	img_mask = np.expand_dims(img_mask,0)
	save_tif(img_mask, mask_resize_dir, np.uint8)

	# 载入swc，生成centerline
	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			
	# 中心线
	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)

	# exist
	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)

	# # 向量计算
	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)

	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction_2d(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector = get_centerline_circle_2d(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

	prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_mask, img_skl, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)


def main_test_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name

	image_seq_dir = args.train_dataset_root_dir + 'test_datasets/'
	org_image_train_dir = args.image_dir + datasets_name + '/test/images_color/'
	org_label_train_dir = args.image_dir + datasets_name + '/test/labels/'
	org_mask_train_dir = args.image_dir + datasets_name + '/test/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/test/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_label_tif_dir = args.image_dir + datasets_name + '/temp/labels/'
	temp_mask_tif_dir = args.image_dir + datasets_name + '/temp/mask/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'

	image_name = input_dir.split("/")[-1].split(".")[0]

	if datasets_name == 'DRIVE':
		patch_num = args.N_patches // 20 // 10 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'

		resize_radio = 2.0
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 3
	elif datasets_name == 'CHASEDB1':
		patch_num = args.N_patches // 8 // 10 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'
		
		resize_radio = 1.5
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 3
	elif datasets_name == 'ROAD':
		patch_num = args.N_patches // 13 // 10 // 4
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'
		
		resize_radio = 1.0
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 3
	else:
		print("Error")
		return 0

	# 载入灰度图像
	img_resize_dir = temp_image_tif_dir + image_name + '_img_resize.tif'
	img_gray = open_tif(img_dir).astype(np.float32)
	img_new = transform.resize(img_gray,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	img_new = np.expand_dims(img_new,0)
	save_tif(img_new, img_resize_dir, np.uint8)
	
 
	# 载入label
	# label_resize_dir = temp_label_tif_dir + image_name + '_label_resize.tif'
	# label_new = open_tif(input_dir).astype(np.float32)
	# label_new = transform.resize(label_new,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	# label_new = np.expand_dims(label_new,0)
	# th = 200
	# label_new[label_new>=th] = 255
	# label_new[label_new<th] = 0
	# save_tif(label_new, label_resize_dir, np.uint8)

	# 载入mask
	mask_resize_dir = temp_mask_tif_dir + image_name + '_mask_resize.tif'
	img_mask = open_tif(img_mask_dir).astype(np.float32)
	img_mask = transform.resize(img_mask,(round(img_gray.shape[0]*resize_radio), round(img_gray.shape[1]*resize_radio)))
	img_mask = np.expand_dims(img_mask,0)
	save_tif(img_mask, mask_resize_dir, np.uint8)

	# 载入swc，生成centerline
	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			
	# 中心线
	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)
	
	# exist
	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)

	# 向量计算
	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)

	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction_2d(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector = get_centerline_circle_2d(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

	prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_mask, img_skl, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)



if __name__ == '__main__':
	args = parse_args()
	
	datasets_name = args.datasets_name
	cpu_core_num = args.multi_cpu
	batch_size = args.input_dim
	patch_num = args.N_patches
	BATCH_SHAPE = [batch_size[0],batch_size[1],batch_size[2]]

 
	print("loading " + datasets_name + " datasets")
	
	# set the folder dir
	org_image_train_dir = args.image_dir + datasets_name + '/training/images_color/'
	org_label_train_dir = args.image_dir + datasets_name + '/training/labels/'
	org_mask_train_dir = args.image_dir + datasets_name + '/training/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	org_image_test_dir = args.image_dir + datasets_name + '/test/images_color/'
	org_label_test_dir = args.image_dir + datasets_name + '/test/labels/'
	org_mask_test_dir = args.image_dir + datasets_name + '/test/mask/'
	org_swc_test_dir = args.image_dir + datasets_name + '/test/swc/'


	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_label_tif_dir = args.image_dir + datasets_name + '/temp/labels/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'

	training_datasets_dir = args.train_dataset_root_dir + 'training_datasets/'
	test_datasets_dir = args.train_dataset_root_dir + 'test_datasets/'

	if not os.path.exists(args.image_dir + datasets_name + '/temp'):
		os.makedirs(args.image_dir + datasets_name + '/temp')
	if not os.path.exists(temp_image_tif_dir):
		os.makedirs(temp_image_tif_dir)
	if not os.path.exists(temp_label_tif_dir):
		os.makedirs(temp_label_tif_dir)
	if not os.path.exists(temp_swc_centerline_dir):
		os.makedirs(temp_swc_centerline_dir)
	if not os.path.exists(temp_centerline_dir):
		os.makedirs(temp_centerline_dir)
		
	if not os.path.exists(args.train_dataset_root_dir):
		os.makedirs(args.train_dataset_root_dir)
	if not os.path.exists(training_datasets_dir):
		os.makedirs(training_datasets_dir)
	if not os.path.exists(test_datasets_dir):
		os.makedirs(test_datasets_dir)


	# Generate the training data and test data
	org_label_list = glob.glob(org_label_train_dir + '*.tif')
	org_label_num = len(org_label_list)
	print('find %d images' % (org_label_num))
	pool = mp.Pool(processes=cpu_core_num)  # we set cpu core is 10
	pool.map(main_training_data, org_label_list) 

	org_label_list = glob.glob(org_label_test_dir + '*.tif')
	org_label_num = len(org_label_list)
	print('find %d images' % (org_label_num))
	pool = mp.Pool(processes=cpu_core_num)  
	pool.map(main_test_data, org_label_list)

 
	# Count the number of files
	import shutil
	total_training_data_num = 0
	training_dataset_list = glob.glob(training_datasets_dir + '*/')
	training_dataset_image_num = len(training_dataset_list)
	print('find %d image folders' % (training_dataset_image_num))
	for training_dataset_image_dir in training_dataset_list:
		training_dataset_image_list = glob.glob(training_dataset_image_dir + '*/')
		training_dataset_image_patch_num = len(training_dataset_image_list)
		print('Folder: %s, find %d images patches' % (training_dataset_image_dir.split('/')[-2], training_dataset_image_patch_num))
		total_training_data_num += training_dataset_image_patch_num
		# for training_dataset_image_patch_dir in training_dataset_image_list:
		# 	shutil.rmtree(training_dataset_image_patch_dir)
	print('TOTAL %d images patches' % (total_training_data_num))

	total_test_data_num = 0
	test_dataset_list = glob.glob(test_datasets_dir + '*/')
	test_dataset_image_num = len(test_dataset_list)
	print('find %d image folders' % (test_dataset_image_num))
	for test_dataset_image_dir in test_dataset_list:
		test_dataset_image_list = glob.glob(test_dataset_image_dir + '*/')
		test_dataset_image_patch_num = len(test_dataset_image_list)
		print('Folder: %s, find %d images patches' % (test_dataset_image_dir.split('/')[-2], test_dataset_image_patch_num))
		total_test_data_num += test_dataset_image_patch_num
		# for test_dataset_image_patch_dir in test_dataset_image_list:
		# 	shutil.rmtree(test_dataset_image_patch_dir)
	print('TOTAL %d images patches' % (total_test_data_num))







