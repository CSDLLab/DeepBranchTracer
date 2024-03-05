import torchvision
import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.spatial import distance_matrix
from skimage import morphology
import numpy as np
from numpy import linalg as LA
import copy
from rtree import index

from lib.klib.glib.SWCExtractor import Vertex
from lib.klib.glib.Obj3D import Point3D, Sphere, Cone
from lib.klib.baseio import *
from lib.swclib import edge_match_utils
from lib.swclib.swc_node import SwcNode
from lib.swclib.euclidean_point import EuclideanPoint

from lib.swclib.swc_io import swc_save

import time

from configs import config_2d

args = config_2d.args
resize_radio = args.resize_radio
r_resize = args.r_resize

def find_if_parent(node_A, node_B, k = 3):
	# find if nodeA is nodeB's parent, k is num
	node_current = node_B
	node_target = node_A

	for i in range(k):
		# print(node_current, node_current.parent)
		if node_current.parent == node_target:
			return True
		else:
			# print(node_current, node_current.parent)
			node_current = node_current.parent
			
			# if node_current.get_id() == 1:
			# 	return True
			if node_current.is_virtual():# 找到了根节点
				return False
	return False

def find_close_point(node_current, node_a, node_b):
	node_current_pos = [node_current.get_z(), node_current.get_x(),node_current.get_y()]
	node_a_pos = [node_a.get_z(), node_a.get_x(), node_a.get_y()]
	node_b_pos = [node_b.get_z(), node_b.get_x(), node_b.get_y()]

	distance_a = math.sqrt((node_current_pos[0] - node_a_pos[0]) ** 2 + (node_current_pos[1] - node_a_pos[1]) ** 2 + (
				node_current_pos[2] - node_a_pos[2])**2)
	distance_b = math.sqrt((node_current_pos[0] - node_b_pos[0]) ** 2 + (node_current_pos[1] - node_b_pos[1]) ** 2 + (
				node_current_pos[2] - node_b_pos[2]) ** 2)
	if distance_a < distance_b:
		return node_a
	else:
		return node_b


def o_distance(point_A, point_B):
	distance = math.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2 + (point_A[2] - point_B[2]) ** 2)
	return distance

def in_range(n, start, end = 0):
    return start <= n <= end if end >= start else end <= n <= start

def cosine_similarity(vector_A,vector_B):
    numerator = 0
    denominator_A = 0
    denominator_B = 0
    for i in range(len(vector_A)):
        numerator = numerator + vector_A[i]*vector_B[i]
        denominator_A = denominator_A + vector_A[i] ** 2
        denominator_B = denominator_B + vector_B[i] ** 2
    result = numerator/np.sqrt(denominator_A+1e-6)/np.sqrt(denominator_B+1e-6)
    return result


def binary_image_skeletonize(input):
    skl = np.expand_dims(input,0)
    label_temp_dilation = morphology.binary_dilation(skl, morphology.ball(2))
    label_temp_erosion = morphology.binary_erosion(label_temp_dilation, morphology.ball(2))

    skeletonize = morphology.skeletonize_3d(label_temp_erosion * 1) * 255
    return skeletonize

def get_pos_image_from_pool_2d(IMAGE_POOL, shape):
    if len(IMAGE_POOL) == 3:
        image_stack = np.zeros(dtype=np.float32, shape=[3, *shape, 3])
        for i in range(3):
            image_stack[i,:,:,:] = copy.deepcopy(IMAGE_POOL[i])
    else:
        image_stack = IMAGE_POOL[-1].reshape(1,*shape,3)
    return image_stack

def get_pos_image_2d(image, image_exist, pos, shape):
    x_half = shape[0]//2
    y_half = shape[1]//2
    pos_z, pos_x, pos_y = pos

    node_img = image[pos_x- x_half:pos_x + x_half, pos_y-y_half:pos_y + y_half].copy()
    node_img_exist = image_exist[pos_x- x_half:pos_x + x_half, pos_y-y_half:pos_y + y_half].copy()
    
    return node_img, node_img_exist



def get_network_predict_2d(image, image_exist, shape, model_test, device, vector_bins, exceed_bound=128): 
    image = copy.deepcopy(image) / 255 

    data_transform = torchvision.transforms.Compose([])
    
    seq_len, _,_,_ = image.shape
    image_tensor = np.zeros(dtype=np.float32, shape=[1, seq_len, 3, *shape])
    image = np.transpose(image, (0, 3, 1, 2))
    
    for i in range(seq_len):
        image_tensor[0,i,:,:,:] = copy.deepcopy(image[i])
        

    idx_tensor = [(idx) for idx in range(vector_bins)]
    idx_tensor = torch.autograd.Variable(torch.FloatTensor(idx_tensor)).to(device)
    softmax = torch.nn.Softmax()

    
    image_tensor_input = data_transform(image_tensor)
    test_loader = torch.utils.data.DataLoader(image_tensor_input, batch_size=1)

    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    for x_batch in test_loader:
        batch_input = x_batch.to(device)
        batch_size, seq_len, _, _, _ = batch_input.size()
        
        
        y_d_pred_1, y_d_pred_2, y_r_pred = model_test(batch_input, 'test_d')
        
        if seq_len == 3: # for LSTM mode
            y_d_pred_1 = y_d_pred_1[:,2,:]
            y_d_pred_2 = y_d_pred_2[:,2,:]
            y_r_pred = y_r_pred[:,2:3]
        else: # for fast mode
            y_d_pred_1 = y_d_pred_1[:,0,:]
            y_d_pred_2 = y_d_pred_2[:,0,:]
            y_r_pred = y_r_pred[:,0:1]
            
        vector1_predicted_softmax = softmax(y_d_pred_1)
        vector2_predicted_softmax = softmax(y_d_pred_2)

        vector1_predicted = torch.sum(vector1_predicted_softmax * idx_tensor, 1) * (1/vector_bins) * 2 - 1
        vector2_predicted = torch.sum(vector2_predicted_softmax * idx_tensor, 1) * (1/vector_bins) #* 2 - 1


        vector_predicted_norm = torch.sqrt(vector1_predicted**2 + vector2_predicted**2 + 1e-9)
        vector1_predicted = vector1_predicted/vector_predicted_norm
        vector2_predicted = vector2_predicted/vector_predicted_norm

        # vector1_predicted = vector1_predicted_softmax.argmax(axis=1) * 0.1 - 1 + 0.05
        # vector2_predicted = vector2_predicted_softmax.argmax(axis=1) * 0.1 - 1 + 0.05
        # vector3_predicted = vector3_predicted_softmax.argmax(axis=1) * 0.1 - 1 + 0.05
        
        # =================test=====================
        # pred_tb = y_dis.cpu().detach().numpy()
        # dis_map = pred_tb[0,0,:,:]

        
        # p_dis = np.mean(dis_map[29:33,29:33])
        # p_seg = np.mean(image_exist[29:33,29:33])/255
        
        x_center = shape[0] // 2
        y_center = shape[1] // 2
        range_x = 2
        range_y = 2
        p_seg = np.max(image_exist[x_center-range_x:x_center+range_x,y_center-range_y:y_center+range_y])/255
        p_centerline = p_seg
        
        


        if p_centerline<=exceed_bound/255:
            pred_exist = 0
        else:
            pred_exist = 1
        
        # print(p_dis, p_seg, p_centerline)

        y_r_pred = y_r_pred.cpu().detach().numpy() * r_resize
        pred_1 = vector1_predicted.cpu().detach().numpy()
        pred_2 = vector2_predicted.cpu().detach().numpy()
        # pred_3 = vector3_predicted.cpu().detach().numpy()
        
        

        # pred_1 = pred_1#*2-1
        # pred_2 = pred_2#*2-1
        # pred_3 = pred_3#*2-1


        return pred_exist, p_centerline, [pred_1[0],pred_2[0]], y_r_pred[0][0]


def get_node_centerline_from_image_2d(node_pos, r_temp, org_image_centerline):
    z_temp = int(round(node_pos[0], 0))
    x_temp = int(round(node_pos[1], 0))
    y_temp = int(round(node_pos[2], 0))

    # centerline 扩大范围
    lambda_ = 1.0
    r_temp_ = int(round((r_temp+1) * lambda_, 0))


    radius_region = np.zeros([2 * r_temp_ + 1, 2 * r_temp_ + 1, 2 * r_temp_ + 1])
    if x_temp - r_temp_ >= 0 and x_temp + r_temp_ < org_image_centerline.shape[0] and y_temp - r_temp_ >= 0 and y_temp + r_temp_ < org_image_centerline.shape[1]:
        centerline_temp = copy.deepcopy(org_image_centerline[x_temp - r_temp_:x_temp + r_temp_ + 1, y_temp - r_temp_:y_temp + r_temp_ + 1])
        # print(np.sum(centerline_temp))
        centerline_temp_point_list = np.where(centerline_temp == 1)
    else:
        centerline_temp = np.zeros_like(radius_region)
        # centerline_temp[r_temp_][r_temp_][r_temp_] = 0#org_image_centerline[z_temp][x_temp][y_temp]
        centerline_temp_point_list = np.where(centerline_temp == 1)

    # 矫正归位
    for i in range(len(centerline_temp_point_list[0])):
        centerline_temp_point_list[0][i] = centerline_temp_point_list[0][i] - r_temp_
        centerline_temp_point_list[1][i] = centerline_temp_point_list[1][i] - r_temp_
        # centerline_temp_point_list[2][i] = centerline_temp_point_list[2][i] - r_temp_

    return centerline_temp_point_list

def get_node_centerline_from_image_vector_2d(node_pos, node_vector, r_temp, org_image_centerline):
    z_temp = round(round(node_pos[0], 0))
    x_temp = round(round(node_pos[1], 0))
    y_temp = round(round(node_pos[2], 0))
    
    # print(org_image_centerline.shape)
    # print(node_vector)
    # pause

    node_vector_u = [0, -node_vector[2], node_vector[1]]
    # node_vector_v = [- node_vector[1] ** 2 - node_vector[2] ** 2 + 1e-7, node_vector[1]*node_vector[0], node_vector[2]*node_vector[0]]
    node_vector_u_n = np.linalg.norm(np.array(node_vector_u))
    # node_vector_v_n = np.linalg.norm(np.array(node_vector_v))
    node_vector_u = node_vector_u / node_vector_u_n
    # node_vector_v = node_vector_v / node_vector_v_n
    

    # centerline 扩大范围
    lambda_ = 1.0
    r_temp_ = round(round((r_temp+0.5) * lambda_, 0))


    X = np.arange(-r_temp_, r_temp_, 1)
    # Y = np.arange(-r_temp_, r_temp_, 1)
    # X_mesh, Y_mesh = np.meshgrid(X, Y)

    # Z_new = z_temp + X * node_vector_u[0] #+ Y_mesh * node_vector_v[0]
    X_new = x_temp + X * node_vector_u[1] #+ Y_mesh * node_vector_v[1]
    Y_new = y_temp + X * node_vector_u[2] #+ Y_mesh * node_vector_v[2]


    
    temp_slice = np.zeros_like(X)
    x = X_new.shape[0]
    # print(x)
    # pause
    for x_t in range(x):
        # z_new = round(Z_new[x_t])
        x_new = round(X_new[x_t])
        y_new = round(Y_new[x_t])
        temp_slice[x_t] = org_image_centerline[x_new][y_new]

    # print(temp_slice)
    # temp_slice = morphology.skeletonize(temp_slice) # trick
    # print(temp_slice)

    # 找到所有值为1的像素点
    c = np.array([round(r_temp_), round(r_temp_)])
    min_dist = np.linalg.norm(temp_slice.shape)

    indices = np.argwhere(temp_slice == 1)

    # print(indices)
    if indices.shape[0] == 0:
        return [-1, -1,-1]
    p = None
    for index in indices:
        # 计算像素点与c之间的距离
        # print(index)
        dist = np.linalg.norm(index - c)
        if dist < min_dist:
            # 更新最小距离和p的值
            min_dist = dist
            p = index
    p_x = p[0]


    p_new = [0, round(X_new[p_x]), round(Y_new[p_x])] 
    # print(p_new)
    # pause
    return p_new


def get_bounds(point_a, point_b, extra=0):
    """
    get bounding box of a segment
    Args:
        point_a: two points to identify the square
        point_b:
        extra: float, a threshold
    Return:
        res(tuple):
    """
    point_a = np.array(point_a.get_center()._pos)
    point_b = np.array(point_b.get_center()._pos)
    res = (np.where(point_a > point_b, point_b, point_a) - extra).tolist() + (np.where(point_a > point_b, point_a, point_b) + extra).tolist()

    return tuple(res)


def tracing_strategy_fast_2d(end_tracing, org_image, org_lab, org_skl, seed_node, seed_node_dict, direction_state, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info):
    model_test, device = device_info[0], device_info[1]
    SHAPE, vector_bins = data_info[0], data_info[1]
    tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]

    begin_time = time.time()
    
    img_shape = org_image.shape
    # mark_shape = ((img_shape[0] + 1), img_shape[1] + 1, (img_shape[2] + 1))

    branch_node_list = []
    exist_score_list = []
    
    # 初始化根节点
    current_node_dict = copy.deepcopy(seed_node_dict)
    steps = 1
    begin_node_id = tree_new.size()


    # 根节点z, x, y
    z_temp = round(current_node_dict['node_z'])
    x_temp = round(current_node_dict['node_x'])
    y_temp = round(current_node_dict['node_y'])



    # 根节点处的network预测
    node_pos = [z_temp, x_temp, y_temp]
    local_image, local_image_exist = get_pos_image_2d(org_image, org_lab, node_pos, SHAPE)
    local_image = local_image.reshape(1,*SHAPE, 3)
    _, _, seed_node_vector, r_temp = get_network_predict_2d(local_image, local_image_exist, SHAPE, model_test, device, vector_bins)
    vector_0 = seed_node_vector[0]
    vector_1 = seed_node_vector[1]
    vector_2 = 0



    # 根节点r, angle
    current_node_dict['radius'] = r_temp
    current_node_dict['vector_0'] = vector_0
    current_node_dict['vector_1'] = vector_1
    current_node_dict['vector_2'] = vector_2

    # 结点拓扑信息
    current_node_dict['id'] = 1
    current_node_dict['pid'] = -1

    parent_node = seed_node


    # 开始追踪
    while end_tracing == False:
        
        print(begin_node_id, '--------------------------------', steps)
        steps = steps + 1
        begin_node_id = begin_node_id+1
        # 读取当前节点信息
        z_temp = round(current_node_dict['node_z'])
        x_temp = round(current_node_dict['node_x'])
        y_temp = round(current_node_dict['node_y'])
        z_delta_old = current_node_dict['z_delta']
        x_delta_old = current_node_dict['x_delta']
        y_delta_old = current_node_dict['y_delta']

        r_temp = current_node_dict['node_r']
        vector_0 = current_node_dict['vector_0']
        vector_1 = current_node_dict['vector_1']
        vector_2 = current_node_dict['vector_2']

        node_id = current_node_dict['node_id']
        node_parent_id = current_node_dict['node_p_id']


        # 计算新结点信息
        vector_0_rad = 0
        vector_1_rad = vector_1
        vector_2_rad = vector_0

        # norm vector
        each_step = np.sqrt(vector_1_rad**2 + vector_2_rad**2)
        vector_0_rad = 0
        vector_1_rad = vector_1_rad / each_step
        vector_2_rad = vector_2_rad / each_step

        # print(vector_2_rad, vector_1_rad, vector_0_rad)


        if r_temp < 0.5 * resize_radio:
            r_temp_step = 0.5 * resize_radio
            step_radius_scale = 2
        # elif r_temp > 3:
        #     r_temp_step = r_temp
        #     step_radius_scale = 1
        else:
            r_temp_step = r_temp
            step_radius_scale = 2

        # print(r_temp_step * step_radius_scale)
        # print(vector_0_rad,vector_1_rad,vector_2_rad)

        z_delta_new = r_temp_step * step_radius_scale * vector_0_rad
        x_delta_new = r_temp_step * step_radius_scale * vector_1_rad
        y_delta_new = r_temp_step * step_radius_scale * vector_2_rad

        # print(np.sqrt(x_delta_new**2 + y_delta_new**2), r_temp)
        # while np.sqrt(x_delta_new**2 + y_delta_new**2) < r_temp:
        while np.abs(x_delta_new) < 1 and np.abs(y_delta_new) < 1:
            # z_delta_new = 2 * z_delta_new
            x_delta_new = 2 * x_delta_new
            y_delta_new = 2 * y_delta_new

            if x_delta_new==0 and y_delta_new==0:
                break

        if direction_state == 0:
            # z_delta_new = z_delta_new
            x_delta_new = x_delta_new
            y_delta_new = y_delta_new
        else:
            # z_delta_new = - z_delta_new
            x_delta_new = - x_delta_new
            y_delta_new = - y_delta_new


        old_delta = [z_delta_old, x_delta_old, y_delta_old]
        new_delta = [z_delta_new, x_delta_new, y_delta_new]
        

        # 余弦相似度防止掉头
        cos_sim = cosine_similarity(old_delta, new_delta)
        if cos_sim < 0:
            cos_sim = - cos_sim
            z_delta_new = - z_delta_new
            x_delta_new = - x_delta_new
            y_delta_new = - y_delta_new

        z_temp_new = current_node_dict['node_z'] + z_delta_new
        x_temp_new = current_node_dict['node_x'] + x_delta_new
        y_temp_new = current_node_dict['node_y'] + y_delta_new

        
        # 获取centerline上的最佳点
        node_pos = [round(z_temp_new), round(x_temp_new), round(y_temp_new)]
        # centerline_temp_point_list = get_node_centerline_from_image_2d(node_pos, r_temp_step * step_radius_scale, org_skl)
        best_point = get_node_centerline_from_image_vector_2d(node_pos, new_delta, r_temp_step * step_radius_scale, org_skl)
        
        

        # best_dis = np.inf
        # best_point = [-1, -1, -1]
        # for i in range(len(centerline_temp_point_list[0])):
        #     z_temp_centerline = round(z_temp_new)
        #     x_temp_centerline = centerline_temp_point_list[0][i] + round(x_temp_new)
        #     y_temp_centerline = centerline_temp_point_list[1][i] + round(y_temp_new)
        #     point_A = [z_temp_new, x_temp_new, y_temp_new]
        #     point_B = [z_temp_centerline, x_temp_centerline, y_temp_centerline]

        #     dis_temp = o_distance(point_A, point_B)
        #     if dis_temp < best_dis:
        #         best_dis = dis_temp
        #         best_point = [z_temp_centerline, x_temp_centerline, y_temp_centerline]

        

        # print(z_temp_new, x_temp_new, y_temp_new)
        # print(best_point)
        
        # tracing_strategy_flag = 'angle'
        # tracing_strategy_flag = 'centerline'
        
        if tracing_strategy_flag == 'angle':
            z_temp_new_ave = z_temp_new
            x_temp_new_ave = x_temp_new
            y_temp_new_ave = y_temp_new
        elif tracing_strategy_flag == 'anglecenterline':
            if best_point == [-1, -1, -1]:
                print("warning: centerline point not found")
                z_temp_new_ave = z_temp_new 
                x_temp_new_ave = x_temp_new 
                y_temp_new_ave = y_temp_new
            else:
                z_temp_new_ave = best_point[0]
                x_temp_new_ave = best_point[1]
                y_temp_new_ave = best_point[2]
                # z_temp_new_ave = (z_temp_new + best_point[0]) / 2
                # x_temp_new_ave = (x_temp_new + best_point[1]) / 2
                # y_temp_new_ave = (y_temp_new + best_point[2]) / 2
        elif tracing_strategy_flag == 'centerline':
            if best_point == [-1, -1, -1]:
                print("warning: centerline point not found")
                z_temp_new_ave = z_temp
                x_temp_new_ave = x_temp
                y_temp_new_ave = y_temp
            else:
                z_temp_new_ave = best_point[0]
                x_temp_new_ave = best_point[1]
                y_temp_new_ave = best_point[2]
        else:
            print("error, please select a tracing mode")
            pause



        

        
        # 更新结点信息
        current_node_dict['node_id'] = begin_node_id

        # print(seed_node_dict['node_id'], begin_node_id - 1)

        if steps == 2:
            current_node_dict['node_p_id'] = seed_node_dict['node_id']
        else:
            current_node_dict['node_p_id'] = begin_node_id - 1

        
        every_step_norm = np.sqrt((z_temp_new_ave-current_node_dict['node_z'])**2 + (x_temp_new_ave-current_node_dict['node_x'])**2 + (y_temp_new_ave-current_node_dict['node_y'])**2)
        z_delta_new = (z_temp_new_ave-current_node_dict['node_z']) / every_step_norm * r_temp_step * step_radius_scale
        x_delta_new = (x_temp_new_ave-current_node_dict['node_x']) / every_step_norm * r_temp_step * step_radius_scale
        y_delta_new = (y_temp_new_ave-current_node_dict['node_y']) / every_step_norm * r_temp_step * step_radius_scale
        every_step = r_temp_step * step_radius_scale

        # every_step = np.sqrt((z_temp_new_ave-current_node_dict['node_z'])**2 + (x_temp_new_ave-current_node_dict['node_x'])**2 + (y_temp_new_ave-current_node_dict['node_y'])**2)

        current_node_dict['node_z'] = z_temp_new_ave
        current_node_dict['node_x'] = x_temp_new_ave
        current_node_dict['node_y'] = y_temp_new_ave
        current_node_dict['z_delta'] = z_delta_new
        current_node_dict['x_delta'] = x_delta_new
        current_node_dict['y_delta'] = y_delta_new


        
        
        # 获得下一目标值
        node_pos = [round(z_temp_new_ave), round(x_temp_new_ave), round(y_temp_new_ave)]
        local_image, local_image_exist = get_pos_image_2d(org_image, org_lab, node_pos, SHAPE)
        local_image = local_image.reshape(1,*SHAPE, 3)
        exist, exist_score, seed_node_vector, r_temp = get_network_predict_2d(local_image, local_image_exist, SHAPE, model_test, device, vector_bins)

        vector_0 = seed_node_vector[0]
        vector_1 = seed_node_vector[1]
        vector_2 = 0 # seed_node_vector[1]


        current_node_dict['node_r'] = r_temp
        current_node_dict['vector_0'] = vector_0
        current_node_dict['vector_1'] = vector_1
        current_node_dict['vector_2'] = vector_2


        not_stable = 0
        # 全局的第二个点（由于无法edge_match_utils.get_nearby_edges计算距离，因此单独设置）
        if steps == 2:
            node_new = EuclideanPoint(center=[current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z']])
            node_root = EuclideanPoint(center=[parent_node.get_x(),parent_node.get_y(),parent_node.get_z()])
            distance = node_new.distance_to_point(node_root)
            
            node_range = current_node_dict['node_r']
            # print('parent node:' , parent_node)
            # print('   son node:' , current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z'], current_node_dict['node_r'])
            # print(distance,parent_node._radius)
        else:
            son_node_temp = SwcNode(nid=1, ntype=0, center=EuclideanPoint(center=[current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z']]), radius=current_node_dict['node_r'])
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=son_node_temp, id_edge_dict=tree_new_idedge_dict, threshold=parent_node.radius()*2)#
            

            if len(node_temp_list) == 0:
                distance = -1
            else:
                node_search_flag=True
                node_p = find_close_point(son_node_temp, node_temp_list[0][0][0], node_temp_list[0][0][1])
                if node_p == parent_node and len(node_temp_list)>1:
                    node_p = find_close_point(son_node_temp, node_temp_list[1][0][0], node_temp_list[1][0][1])
                    distance = node_temp_list[1][1]
                else:
                    distance = node_temp_list[0][1]
                node_range = node_p._radius
                
                if son_node_temp._pos.get_x() == parent_node._pos.get_x() and son_node_temp._pos.get_y() == parent_node._pos.get_y() and son_node_temp._pos.get_z() == parent_node._pos.get_z():
                    node_range = np.Inf

                
        
        if in_range(round(x_temp_new_ave), 0, org_image.shape[0]) is False or in_range(round(y_temp_new_ave), 0, org_image.shape[1]) is False:
            print("exceed the bound")
            end_tracing = True

        
        exist_current = 1
        exist_score_final = 1

        gap = 2
        if steps>gap+1:
            exist_score_final = np.mean(exist_score_list[-gap:])
        else:
            exist_current = exist
            
            

        if r_temp < 0.3 * resize_radio:
            print("radius is 0")
            end_tracing = True
        elif exist_current==0: 
            print("exist is 0")
            end_tracing = True
        elif exist_score_final<0.4/gap: # 
            print("exist_score_final is not suitable")
            tree_new.remove_node(branch_node_list[-1])
            tree_new.get_node_list(update=True)
            tree_new.remove_node(branch_node_list[-2])
            tree_new.get_node_list(update=True)
            # tree_new.remove_node(branch_node_list[-3])
            # tree_new.get_node_list(update=True)
            end_tracing = True
        elif distance < node_range:
            print("this branch is traced")
            end_tracing = True
            
            son_node = SwcNode(nid=current_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[round(current_node_dict['node_y'],3),round(current_node_dict['node_x'],3),round(current_node_dict['node_z'],3)]), radius=round(current_node_dict['node_r'],3))
            
            branch_node_list.append(son_node)
            
            tree_new.add_child(parent_node, son_node)
            tree_new.get_node_list(update=True)
            tree_new_rtree.insert(son_node.get_id(), get_bounds(son_node, son_node.parent, extra=son_node.radius()*1.5))
            tree_new_idedge_dict[son_node.get_id()] = tuple([son_node, son_node.parent])
            
            
        else:
            exist_score_list.append(exist_score)
            
            son_node = SwcNode(nid=current_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[round(current_node_dict['node_y'],3),round(current_node_dict['node_x'],3),round(current_node_dict['node_z'],3)]), radius=round(current_node_dict['node_r'],3))
            
            branch_node_list.append(son_node)
            
            tree_new.add_child(parent_node, son_node)
            tree_new.get_node_list(update=True)
            parent_node = son_node

            tree_new_rtree.insert(son_node.get_id(), get_bounds(son_node, son_node.parent, extra=son_node.radius()*1.5))
            tree_new_idedge_dict[son_node.get_id()] = tuple([son_node, son_node.parent])

            # pause
    end_time = time.time()
    print(end_time-begin_time)


    r_tree_info = [tree_new_rtree, tree_new_idedge_dict]

    

    return test_image, tree_new, branch_node_list, r_tree_info


def tracing_strategy_lstm_2d(end_tracing, org_image, org_lab, org_skl, seed_node, seed_node_dict, direction_state, test_image, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info):
    model_test, device = device_info[0], device_info[1]
    SHAPE, vector_bins = data_info[0], data_info[1]
    tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]

    begin_time = time.time()
    
    img_shape = org_image.shape
    # mark_shape = ((img_shape[0] + 1), img_shape[1] + 1, (img_shape[2] + 1))

    branch_node_list = []
    exist_score_list = []
    
    # 初始化根节点
    current_node_dict = copy.deepcopy(seed_node_dict)
    steps = 1
    begin_node_id = tree_new.size()


    # 根节点z, x, y
    z_temp = round(current_node_dict['node_z'])
    x_temp = round(current_node_dict['node_x'])
    y_temp = round(current_node_dict['node_y'])

    # image pool
    IMAGE_POOL = []

    # 根节点处的network预测
    node_pos = [z_temp, x_temp, y_temp]
    local_image, local_image_exist = get_pos_image_2d(org_image, org_lab, node_pos, SHAPE)
    IMAGE_POOL.append(local_image)
    local_image = local_image.reshape(1,*SHAPE, 3)
    _, _, seed_node_vector, r_temp = get_network_predict_2d(local_image, local_image_exist, SHAPE, model_test, device, vector_bins)
    vector_0 = seed_node_vector[0]
    vector_1 = seed_node_vector[1]
    vector_2 = 0



    # 根节点r, angle
    current_node_dict['radius'] = r_temp
    current_node_dict['vector_0'] = vector_0
    current_node_dict['vector_1'] = vector_1
    current_node_dict['vector_2'] = vector_2

    # 结点拓扑信息
    current_node_dict['id'] = 1
    current_node_dict['pid'] = -1

    parent_node = seed_node


    # 开始追踪
    while end_tracing == False:
        
        print(begin_node_id, '--------------------------------', steps)
        steps = steps + 1
        begin_node_id = begin_node_id+1
        # 读取当前节点信息
        z_temp = round(current_node_dict['node_z'])
        x_temp = round(current_node_dict['node_x'])
        y_temp = round(current_node_dict['node_y'])
        z_delta_old = current_node_dict['z_delta']
        x_delta_old = current_node_dict['x_delta']
        y_delta_old = current_node_dict['y_delta']

        r_temp = current_node_dict['node_r']
        vector_0 = current_node_dict['vector_0']
        vector_1 = current_node_dict['vector_1']
        vector_2 = current_node_dict['vector_2']

        node_id = current_node_dict['node_id']
        node_parent_id = current_node_dict['node_p_id']


        # 计算新结点信息
        vector_0_rad = 0
        vector_1_rad = vector_1
        vector_2_rad = vector_0

        # norm vector
        each_step = np.sqrt(vector_1_rad**2 + vector_2_rad**2)
        vector_0_rad = 0
        vector_1_rad = vector_1_rad / each_step
        vector_2_rad = vector_2_rad / each_step

        # print(vector_2_rad, vector_1_rad, vector_0_rad)


        if r_temp < 0.5 * resize_radio:
            r_temp_step = 0.5 * resize_radio
            step_radius_scale = 2
        # elif r_temp > 3:
        #     r_temp_step = r_temp
        #     step_radius_scale = 1
        else:
            r_temp_step = r_temp
            step_radius_scale = 2

        # print(r_temp_step * step_radius_scale)
        # print(vector_0_rad,vector_1_rad,vector_2_rad)

        z_delta_new = r_temp_step * step_radius_scale * vector_0_rad
        x_delta_new = r_temp_step * step_radius_scale * vector_1_rad
        y_delta_new = r_temp_step * step_radius_scale * vector_2_rad

        # print(np.sqrt(x_delta_new**2 + y_delta_new**2), r_temp)
        # while np.sqrt(x_delta_new**2 + y_delta_new**2) < r_temp:
        while np.abs(x_delta_new) < 1 and np.abs(y_delta_new) < 1:
            # z_delta_new = 2 * z_delta_new
            x_delta_new = 2 * x_delta_new
            y_delta_new = 2 * y_delta_new

            if x_delta_new==0 and y_delta_new==0:
                break

        if direction_state == 0:
            # z_delta_new = z_delta_new
            x_delta_new = x_delta_new
            y_delta_new = y_delta_new
        else:
            # z_delta_new = - z_delta_new
            x_delta_new = - x_delta_new
            y_delta_new = - y_delta_new


        old_delta = [z_delta_old, x_delta_old, y_delta_old]
        new_delta = [z_delta_new, x_delta_new, y_delta_new]
        

        # 余弦相似度防止掉头
        cos_sim = cosine_similarity(old_delta, new_delta)
        if cos_sim < 0:
            cos_sim = - cos_sim
            z_delta_new = - z_delta_new
            x_delta_new = - x_delta_new
            y_delta_new = - y_delta_new

        z_temp_new = current_node_dict['node_z'] + z_delta_new
        x_temp_new = current_node_dict['node_x'] + x_delta_new
        y_temp_new = current_node_dict['node_y'] + y_delta_new

        
        # 获取centerline上的最佳点
        node_pos = [round(z_temp_new), round(x_temp_new), round(y_temp_new)]
        # centerline_temp_point_list = get_node_centerline_from_image_2d(node_pos, r_temp_step * step_radius_scale, org_skl)
        best_point = get_node_centerline_from_image_vector_2d(node_pos, new_delta, r_temp_step * step_radius_scale, org_skl)
        
        

        # best_dis = np.inf
        # best_point = [-1, -1, -1]
        # for i in range(len(centerline_temp_point_list[0])):
        #     z_temp_centerline = round(z_temp_new)
        #     x_temp_centerline = centerline_temp_point_list[0][i] + round(x_temp_new)
        #     y_temp_centerline = centerline_temp_point_list[1][i] + round(y_temp_new)
        #     point_A = [z_temp_new, x_temp_new, y_temp_new]
        #     point_B = [z_temp_centerline, x_temp_centerline, y_temp_centerline]

        #     dis_temp = o_distance(point_A, point_B)
        #     if dis_temp < best_dis:
        #         best_dis = dis_temp
        #         best_point = [z_temp_centerline, x_temp_centerline, y_temp_centerline]

        

        # print(z_temp_new, x_temp_new, y_temp_new)
        # print(best_point)
        
        # tracing_strategy_flag = 'angle'
        # tracing_strategy_flag = 'centerline'
        
        if tracing_strategy_flag == 'angle':
            z_temp_new_ave = z_temp_new
            x_temp_new_ave = x_temp_new
            y_temp_new_ave = y_temp_new
        elif tracing_strategy_flag == 'anglecenterline':
            if best_point == [-1, -1, -1]:
                print("warning: centerline point not found")
                z_temp_new_ave = z_temp_new 
                x_temp_new_ave = x_temp_new 
                y_temp_new_ave = y_temp_new
            else:
                z_temp_new_ave = best_point[0]
                x_temp_new_ave = best_point[1]
                y_temp_new_ave = best_point[2]
                # z_temp_new_ave = (z_temp_new + best_point[0]) / 2
                # x_temp_new_ave = (x_temp_new + best_point[1]) / 2
                # y_temp_new_ave = (y_temp_new + best_point[2]) / 2
        elif tracing_strategy_flag == 'centerline':
            if best_point == [-1, -1, -1]:
                print("warning: centerline point not found")
                z_temp_new_ave = z_temp
                x_temp_new_ave = x_temp
                y_temp_new_ave = y_temp
            else:
                z_temp_new_ave = best_point[0]
                x_temp_new_ave = best_point[1]
                y_temp_new_ave = best_point[2]
        else:
            print("error, please select a tracing mode")
            pause



        

        
        # 更新结点信息
        current_node_dict['node_id'] = begin_node_id

        # print(seed_node_dict['node_id'], begin_node_id - 1)

        if steps == 2:
            current_node_dict['node_p_id'] = seed_node_dict['node_id']
        else:
            current_node_dict['node_p_id'] = begin_node_id - 1

        
        every_step_norm = np.sqrt((z_temp_new_ave-current_node_dict['node_z'])**2 + (x_temp_new_ave-current_node_dict['node_x'])**2 + (y_temp_new_ave-current_node_dict['node_y'])**2)
        z_delta_new = (z_temp_new_ave-current_node_dict['node_z']) / every_step_norm * r_temp_step * step_radius_scale
        x_delta_new = (x_temp_new_ave-current_node_dict['node_x']) / every_step_norm * r_temp_step * step_radius_scale
        y_delta_new = (y_temp_new_ave-current_node_dict['node_y']) / every_step_norm * r_temp_step * step_radius_scale
        every_step = r_temp_step * step_radius_scale

        # every_step = np.sqrt((z_temp_new_ave-current_node_dict['node_z'])**2 + (x_temp_new_ave-current_node_dict['node_x'])**2 + (y_temp_new_ave-current_node_dict['node_y'])**2)

        current_node_dict['node_z'] = z_temp_new_ave
        current_node_dict['node_x'] = x_temp_new_ave
        current_node_dict['node_y'] = y_temp_new_ave
        current_node_dict['z_delta'] = z_delta_new
        current_node_dict['x_delta'] = x_delta_new
        current_node_dict['y_delta'] = y_delta_new


        
        
        # 获得下一目标值
        node_pos = [round(z_temp_new_ave), round(x_temp_new_ave), round(y_temp_new_ave)]
        local_image, local_image_exist = get_pos_image_2d(org_image, org_lab, node_pos, SHAPE)
        
        if len(IMAGE_POOL)<3:
            IMAGE_POOL.append(local_image)
        else:
            IMAGE_POOL.pop(0)
            IMAGE_POOL.append(local_image)
            
        local_image_rnn = get_pos_image_from_pool_2d(IMAGE_POOL, SHAPE)
        exist, exist_score, seed_node_vector, r_temp = get_network_predict_2d(local_image_rnn, local_image_exist, SHAPE, model_test, device, vector_bins)

        vector_0 = seed_node_vector[0]
        vector_1 = seed_node_vector[1]
        vector_2 = 0 # seed_node_vector[1]


        current_node_dict['node_r'] = r_temp
        current_node_dict['vector_0'] = vector_0
        current_node_dict['vector_1'] = vector_1
        current_node_dict['vector_2'] = vector_2


        not_stable = 0
        # 全局的第二个点（由于无法edge_match_utils.get_nearby_edges计算距离，因此单独设置）
        if steps == 2:
            node_new = EuclideanPoint(center=[current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z']])
            node_root = EuclideanPoint(center=[parent_node.get_x(),parent_node.get_y(),parent_node.get_z()])
            distance = node_new.distance_to_point(node_root)
            
            node_range = current_node_dict['node_r']
            # print('parent node:' , parent_node)
            # print('   son node:' , current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z'], current_node_dict['node_r'])
            # print(distance,parent_node._radius)
        else:
            son_node_temp = SwcNode(nid=1, ntype=0, center=EuclideanPoint(center=[current_node_dict['node_y'],current_node_dict['node_x'],current_node_dict['node_z']]), radius=current_node_dict['node_r'])
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=son_node_temp, id_edge_dict=tree_new_idedge_dict, threshold=parent_node.radius()*2)#
            

            if len(node_temp_list) == 0:
                distance = -1
            else:
                node_search_flag=True
                node_p = find_close_point(son_node_temp, node_temp_list[0][0][0], node_temp_list[0][0][1])
                if node_p == parent_node and len(node_temp_list)>1:
                    node_p = find_close_point(son_node_temp, node_temp_list[1][0][0], node_temp_list[1][0][1])
                    distance = node_temp_list[1][1]
                else:
                    distance = node_temp_list[0][1]
                node_range = node_p._radius
                
                if son_node_temp._pos.get_x() == parent_node._pos.get_x() and son_node_temp._pos.get_y() == parent_node._pos.get_y() and son_node_temp._pos.get_z() == parent_node._pos.get_z():
                    node_range = np.Inf

                
        
        if in_range(round(x_temp_new_ave), 0, org_image.shape[0]) is False or in_range(round(y_temp_new_ave), 0, org_image.shape[1]) is False:
            print("exceed the bound")
            end_tracing = True

        
        exist_current = 1
        exist_score_final = 1

        gap = 2
        if steps>gap+1:
            exist_score_final = np.mean(exist_score_list[-gap:])
        else:
            exist_current = exist
            
            

        if r_temp < 0.3 * resize_radio:
            print("radius is 0")
            end_tracing = True
        elif exist_current==0: 
            print("exist is 0")
            end_tracing = True
        elif exist_score_final<0.4/gap: # 
            print("exist_score_final is not suitable")
            tree_new.remove_node(branch_node_list[-1])
            tree_new.get_node_list(update=True)
            tree_new.remove_node(branch_node_list[-2])
            tree_new.get_node_list(update=True)
            # tree_new.remove_node(branch_node_list[-3])
            # tree_new.get_node_list(update=True)
            end_tracing = True
        elif distance < node_range:
            print("this branch is traced")
            end_tracing = True
            
            son_node = SwcNode(nid=current_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[round(current_node_dict['node_y'],3),round(current_node_dict['node_x'],3),round(current_node_dict['node_z'],3)]), radius=round(current_node_dict['node_r'],3))
            
            branch_node_list.append(son_node)
            
            tree_new.add_child(parent_node, son_node)
            tree_new.get_node_list(update=True)
            tree_new_rtree.insert(son_node.get_id(), get_bounds(son_node, son_node.parent, extra=son_node.radius()*1.5))
            tree_new_idedge_dict[son_node.get_id()] = tuple([son_node, son_node.parent])
            
            
        else:
            exist_score_list.append(exist_score)
            
            son_node = SwcNode(nid=current_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[round(current_node_dict['node_y'],3),round(current_node_dict['node_x'],3),round(current_node_dict['node_z'],3)]), radius=round(current_node_dict['node_r'],3))
            
            branch_node_list.append(son_node)
            
            tree_new.add_child(parent_node, son_node)
            tree_new.get_node_list(update=True)
            parent_node = son_node

            tree_new_rtree.insert(son_node.get_id(), get_bounds(son_node, son_node.parent, extra=son_node.radius()*1.5))
            tree_new_idedge_dict[son_node.get_id()] = tuple([son_node, son_node.parent])

            # pause
    end_time = time.time()
    print(end_time-begin_time)


    r_tree_info = [tree_new_rtree, tree_new_idedge_dict]

    

    return test_image, tree_new, branch_node_list, r_tree_info

