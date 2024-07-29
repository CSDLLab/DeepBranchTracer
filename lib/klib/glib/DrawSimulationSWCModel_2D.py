import os
from functools import reduce
from collections import deque
import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.spatial import distance_matrix
from .Transformations import rotation_matrix, superimposition_matrix
from .SWCExtractor import Vertex
from .Obj3D import Point3D, Sphere, Cone, calculateBound, calScaleRatio
from .Utils import Timer
from . import Draw3DTools
# from .Draw3DTools import randIntList
from . import ImageUtils
import copy
import math
from scipy.ndimage import filters as ndfilter

def save_swc(save_dir, swc_data):
    with open(save_dir, 'w') as fp:
        for i in range(swc_data.shape[0]):
            fp.write('%d %d %g %g %g %g %d\n' % (
                swc_data[i][0], swc_data[i][1], swc_data[i][2], swc_data[i][3], swc_data[i][4], swc_data[i][5],
                swc_data[i][6]))
        fp.close()

def add_noise(MAX_BOX_WIDTH, mark, forground):
    noise_num = MAX_BOX_WIDTH[0] * 4
    noise_image = np.zeros([MAX_BOX_WIDTH[0], MAX_BOX_WIDTH[1], MAX_BOX_WIDTH[2]])

    radius_list = []
    for i in range(4):
        radius_temp = np.zeros([1 + 2 * (i + 1), 1 + 2 * (i + 1), 1 + 2 * (i + 1)])
        for z in range(radius_temp.shape[0]):
            for x in range(radius_temp.shape[0]):
                for y in range(radius_temp.shape[0]):

                    if math.sqrt((x - (i + 1)) ** 2 + (y - (i + 1)) ** 2 + (z - (i + 1)) ** 2) <= i + 1:
                        radius_temp[z][x][y] = 1
        radius_list.append(radius_temp)

    for i in range(noise_num):
        noise_size = np.random.choice([1, 2, 3, 4], p=[0.7, 0.25, 0.04, 0.01])

        pos_x = np.random.randint(0, MAX_BOX_WIDTH[1] - 3 * noise_size)
        pos_y = np.random.randint(0, MAX_BOX_WIDTH[2] - 3 * noise_size)

        value = np.random.randint(forground * 0.5, forground * 1.5)

        z_list, x_list, y_list = np.where(radius_list[noise_size - 1] == 1)


        for j in range(len(z_list)):
            x_temp = x_list[j]
            y_temp = y_list[j]
            # print(pos_z+z_temp,pos_x+x_temp)
            # print(value)
            if mark[0][pos_x + x_temp][pos_y + y_temp] != 1:
                noise_image[0][pos_x + x_temp][pos_y + y_temp] = value
    noise_image = ndfilter.gaussian_filter(noise_image, [1, 1, 1])
    return noise_image

def normalizeImage16(im):
    im = np.asarray(im, np.float)
    im = np.where(im > 65535, 65535, im)
    im = np.where(im < 0, 0, im)
    # im = im.astype(np.uint16)
    return im


def normalizeImage8(im):
    im = np.asarray(im, np.float)
    im = np.where(im > 255, 255, im)
    im = np.where(im < 0, 0, im)
    # im = im.astype(np.uint8)
    return im

def gaussianNoisyAddGray3D(image,mean,std,data_type):
    row,col,dep= image.shape
    gauss = np.random.normal(mean,std,(row,col,dep))
    gauss = gauss.reshape(row,col,dep)
    noisy = image + gauss
    return noisy

def getRandChildNumber():
    ''' Random generate children number of a tree node
        Input:
            None
        Output:
            (int) : Children number
    '''
    return np.random.choice([1, 2], p=[0.7, 0.3])


def getChildRadius(depth):
    return np.random.uniform(1,3) * (0.9)**(depth)
def getChildRadius_new(radius):
    return np.random.uniform(0.8,1.1) * radius * (0.9)



def getChildLength(base_length, depth):
    return np.random.randint(base_length * 0.8, base_length * 1.2) * (0.9)**(depth)


def setMarkWithSphere(mark, sphere, mark_shape, use_bbox=False):
    bbox = list(sphere.calBBox()) # xmin,ymin,zmin,xmax,ymax,zmax
    for i in range(3):
        j = i+3
        if (bbox[i]<0):
            bbox[i] = 0
        if (bbox[j]>mark_shape[i]):
            bbox[j] = mark_shape[i]
    (xmin,ymin,zmin,xmax,ymax,zmax) = tuple(bbox)
    (x_idxs,y_idxs,z_idxs)=np.where(mark[xmin:xmax,ymin:ymax,zmin:zmax]==0)
    if not use_bbox:
        xs = np.asarray(xmin+x_idxs).reshape((len(x_idxs),1))
        ys = np.asarray(ymin+y_idxs).reshape((len(y_idxs),1))
        zs = np.asarray(zmin+z_idxs).reshape((len(z_idxs),1))
        points=np.hstack((xs,ys,zs))

        sphere_c_mat = np.array([sphere.center_point.toList()]) # 1*3
        # 计算所有点到所有球心的距离
        dis_mat = distance_matrix(points,sphere_c_mat) # M*1

        # 判断距离是否小于半径
        res_idxs = np.where(dis_mat<=sphere.radius)[0]
        for pos in res_idxs:
            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 1
    else:
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):
            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 1



def setMarkWithCone(mark, cone, mark_shape, use_bbox=False):
    bbox = list(cone.calBBox()) # xmin,ymin,zmin,xmax,ymax,zmax
    for i in range(3):
        j = i+3
        if (bbox[i]<0):
            bbox[i] = 0
        if (bbox[j]>mark_shape[i]):
            bbox[j] = mark_shape[i]
    (xmin,ymin,zmin,xmax,ymax,zmax) = tuple(bbox)

    (x_idxs,y_idxs,z_idxs)=np.where(mark[xmin:xmax,ymin:ymax,zmin:zmax]==0)
    if not use_bbox:
        xs = np.asarray(xmin+x_idxs).reshape((len(x_idxs),1))
        ys = np.asarray(ymin+y_idxs).reshape((len(y_idxs),1))
        zs = np.asarray(zmin+z_idxs).reshape((len(z_idxs),1))
        ns = np.ones((len(z_idxs),1))
        points=np.hstack((xs,ys,zs,ns))


        r_min=cone.up_radius
        r_max=cone.bottom_radius
        height=cone.height
        cone_revert_mat = cone.revertMat().T # 4*4


        revert_coor_mat = np.matmul(points, cone_revert_mat) # M*4
        revert_radius_list = LA.norm(revert_coor_mat[:,:2], axis=1) # M

        # Local Indexs
        M = points.shape[0]
        l_idx = np.arange(M) # M (1-dim)
        l_mark = np.ones((M,), dtype=bool)


        res_idxs = np.logical_or(revert_coor_mat[l_idx[l_mark],2]<0, revert_coor_mat[l_idx[l_mark],2]>height)
        l_mark[l_idx[l_mark][res_idxs]]=False


        res_idxs = revert_radius_list[l_idx[l_mark]]>r_max
        l_mark[l_idx[l_mark][res_idxs]]=False


        res_idxs = revert_radius_list[l_idx[l_mark]]<=r_min
        for pos in l_idx[l_mark][res_idxs]:
            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 1
        l_mark[l_idx[l_mark][res_idxs]]=False


        if r_max>r_min:
            res_idxs = ((r_max-revert_radius_list[l_idx[l_mark]])*height/(r_max-r_min)) >= revert_coor_mat[l_idx[l_mark],2]

            for pos in l_idx[l_mark][res_idxs]:
                mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 1
            l_mark[l_idx[l_mark][res_idxs]]=False
    else:
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):
            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 1





def simulate2DTreeModel_dendrite(MAX_BOX_WIDTH, internal_feature, external_feature, data_type):
    if internal_feature == True:
        max_depth = np.random.randint(6, 8)
        base_length = np.random.randint(35, 55)
    else:
        max_depth = 2
        base_length = np.random.randint(100, 150)

    MAX_TREE_DEPTH = max_depth
    BASE_LENGTH = base_length

    # Init space
    mark_shape = (MAX_BOX_WIDTH[0], MAX_BOX_WIDTH[1], MAX_BOX_WIDTH[2])
    mark = np.zeros(mark_shape, dtype=np.uint16)

    # Create root node
    node_count = 0
    # 胞体大小
    root_r = np.random.uniform(5, 10)
    global_r = root_r

    # 起始点位置
    z_pos = 10

    pos_lib = [[z_pos, MAX_BOX_WIDTH[1] // 2, MAX_BOX_WIDTH[2] // 2]]
    pos_random = np.random.randint(0, 1)

    root_pos = (pos_lib[pos_random][0], pos_lib[pos_random][1], pos_lib[pos_random][2])

    node_count += 1
    root_node = Vertex(node_count, 0, root_pos[0], root_pos[1], root_pos[2], root_r, -1)
    setMarkWithSphere(mark, Sphere(Point3D(*root_node.pos), root_node.r), mark_shape)
    is_axons = False
    axons_num = 0  # 在第x个后，分叉

    # Creante dequeue and list to contain result
    dq = deque([(root_node, 0, is_axons)])  # 第二项表示node节点的depth
    nodes = {}
    graph = {}

    id_list = []
    pos_x_list = []
    pos_y_list = []
    pos_z_list = []
    r_list = []
    pid_list = []

    while len(dq):
        root_node = dq[0][0]
        root_depth = dq[0][1]
        root_axons = dq[0][2]

        # print(root_node)
        id_list.append(root_node.idx)
        pos_x_list.append(root_node.pos[2])
        pos_y_list.append(root_node.pos[1])
        pos_z_list.append(root_node.pos[0])
        r_list.append(root_node.r)
        pid_list.append(root_node.p_idx)

        dq.popleft()

        # Add to nodes and graph
        v1 = root_node.idx
        v2 = root_node.p_idx
        if root_node.idx not in nodes:
            nodes[root_node.idx] = root_node
        if v1 > 0 and v2 > 0:
            if not v1 in graph:
                graph[v1] = set([v2])
            else:
                graph[v1].add(v2)

            if not v2 in graph:
                graph[v2] = set([v1])
            else:
                graph[v2].add(v1)

        if root_depth < MAX_TREE_DEPTH:
            # Get children number
            if root_node.idx == 1:  # 根节点单独处理
                mask = np.array([[1, 1, 1],
                                 [-1, 1, 1],
                                 [1, 1, -1],
                                 [-1, 1, -1]])
                for i in range(2):
                    # 获取分支半径和长度

                    if i == 0:
                        child_r = global_r / 2
                        child_length = global_r * 2  # getChildLength(BASE_LENGTH, root_depth + 1, MAX_TREE_DEPTH)
                        theta_z = np.random.uniform(75, 105)
                        theta_y = np.random.uniform(-15, 15)
                    else:
                        child_r = global_r / 2
                        child_length = global_r * 2  # getChildLength(BASE_LENGTH, root_depth + 1, MAX_TREE_DEPTH)
                        theta_z = np.random.uniform(-105, -75)
                        theta_y = np.random.uniform(-15, 15)

                    A = rotation_matrix(theta_z / 180 * np.math.pi, [0, 0, 1])
                    B = rotation_matrix(-theta_y / 180 * np.math.pi, [0, 1, 0])
                    rot_mat = np.matmul(A, B)
                    p0 = np.array([[child_length], [0], [0], [1]])
                    p1 = np.matmul(rot_mat, p0)
                    child_pos = (int(p1[0] * mask[i][0] + root_node.pos[0]), \
                                 int(p1[1] * mask[i][1] + root_node.pos[1]), \
                                 int(p1[2] * mask[i][2] + root_node.pos[2]))
                    if ImageUtils.bboxCheck3D(z_pos, child_pos[1], child_pos[2], child_r, mark_shape):
                        node_count += 1
                        child_node = Vertex(node_count, 0, z_pos, child_pos[1], child_pos[2], child_r,
                                            root_node.idx)
                        # 绘制
                        setMarkWithSphere(mark, Sphere(Point3D(*child_node.pos), child_node.r), mark_shape)
                        setMarkWithCone(mark, Cone(Point3D(*root_node.pos), root_node.r, \
                                                   Point3D(*child_node.pos), child_node.r), mark_shape)

                        # Add to dequeue
                        if i == 0:
                            dq.append((child_node, root_depth + 1, False))
                        else:
                            dq.append((child_node, root_depth + 1, True))

            elif root_axons:
                if axons_num < 3:  # 4个轴突结构之后开始分叉
                    child_num = 1
                else:
                    child_num = getRandChildNumber()
                axons_num += 1
                child_angles_range = Draw3DTools.sliceRange(0, 360, child_num)

                for i in range(child_num):

                    # 获取分支半径和长度
                    # child_r = getChildRadius(root_depth + 1)
                    child_r = getChildRadius_new(root_node.r)

                    child_length = getChildLength(BASE_LENGTH, root_depth + 1)

                    # 获取生长角度
                    if child_num == 1:
                        theta_z = np.random.uniform(0, 360)
                        theta_y = np.random.uniform(75, 105)
                    else:
                        theta_z = np.random.uniform(child_angles_range[i][0], child_angles_range[i][1])
                        theta_y = np.random.uniform(45, 70)

                    A = rotation_matrix(theta_z / 180 * np.math.pi, [0, 0, 1])
                    B = rotation_matrix(-theta_y / 180 * np.math.pi, [0, 1, 0])
                    rot_mat = np.matmul(A, B)
                    p0 = np.array([[child_length], [0], [0], [1]])
                    p1 = np.matmul(rot_mat, p0)

                    grand_node = nodes[root_node.p_idx]  # root节点的父节点
                    p_a = Point3D(0, 0, 0)
                    p_c = Point3D(root_node.pos[0] - grand_node.pos[0], \
                                  root_node.pos[1] - grand_node.pos[1], \
                                  root_node.pos[2] - grand_node.pos[2])
                    p_b = p_a.medianWithPoint(p_c)
                    v1 = np.array([[p_a.x, p_b.x, p_c.x],  # 局部坐标点
                                   [p_a.y, p_b.y, p_c.y],
                                   [p_a.z, p_b.z, p_c.z],
                                   [1, 1, 1]])
                    Dis = p_a.distanceWithPoint(p_c)
                    v0 = np.array([[0, 0, 0],  # 局部坐标点
                                   [0, 0, 0],
                                   [-Dis, -Dis / 2, 0],
                                   [1, 1, 1]])
                    rev_mat = superimposition_matrix(v0, v1)
                    p2 = np.matmul(rev_mat, p1)
                    child_pos = (
                        int(p2[0] + grand_node.pos[0]), int(p2[1] + grand_node.pos[1]), int(p2[2] + grand_node.pos[2]))
                    if ImageUtils.bboxCheck3D(z_pos, child_pos[1], child_pos[2], child_r, mark_shape):
                        node_count += 1
                        child_node = Vertex(node_count, 0, z_pos, child_pos[1], child_pos[2], child_r,
                                            root_node.idx)
                        # 绘制
                        setMarkWithSphere(mark, Sphere(Point3D(*child_node.pos), child_node.r), mark_shape)
                        setMarkWithCone(mark, Cone(Point3D(*root_node.pos), root_node.r, Point3D(*child_node.pos),
                                                   child_node.r), mark_shape)

                        # Add to dequeue
                        dq.append((child_node, root_depth + 1, True))

            else:
                if root_node.idx == 2:
                    child_num = np.random.choice([2, 3, 4], p=[0.85, 0.1, 0.05])
                else:
                    child_num = getRandChildNumber()
                child_angles_range = Draw3DTools.sliceRange(0, 360, child_num)

                for i in range(child_num):

                    # 获取分支半径和长度
                    # child_r = getChildRadius(root_depth + 1)
                    child_r = getChildRadius_new(root_node.r)



                    child_length = getChildLength(BASE_LENGTH, root_depth + 1)

                    # 获取生长角度
                    if child_num == 1:
                        theta_z = np.random.uniform(0, 360)
                        theta_y = np.random.uniform(75, 105)
                    else:
                        theta_z = np.random.uniform(child_angles_range[i][0], child_angles_range[i][1])
                        theta_y = np.random.uniform(45, 70)

                    A = rotation_matrix(theta_z / 180 * np.math.pi, [0, 0, 1])
                    B = rotation_matrix(-theta_y / 180 * np.math.pi, [0, 1, 0])
                    rot_mat = np.matmul(A, B)
                    p0 = np.array([[child_length], [0], [0], [1]])
                    p1 = np.matmul(rot_mat, p0)

                    grand_node = nodes[root_node.p_idx]  # root节点的父节点
                    p_a = Point3D(0, 0, 0)
                    p_c = Point3D(root_node.pos[0] - grand_node.pos[0], \
                                  root_node.pos[1] - grand_node.pos[1], \
                                  root_node.pos[2] - grand_node.pos[2])
                    p_b = p_a.medianWithPoint(p_c)
                    v1 = np.array([[p_a.x, p_b.x, p_c.x],  # 局部坐标点
                                   [p_a.y, p_b.y, p_c.y],
                                   [p_a.z, p_b.z, p_c.z],
                                   [1, 1, 1]])
                    Dis = p_a.distanceWithPoint(p_c)
                    v0 = np.array([[0, 0, 0],  # 局部坐标点
                                   [0, 0, 0],
                                   [-Dis, -Dis / 2, 0],
                                   [1, 1, 1]])
                    rev_mat = superimposition_matrix(v0, v1)
                    p2 = np.matmul(rev_mat, p1)
                    child_pos = (
                    int(p2[0] + grand_node.pos[0]), int(p2[1] + grand_node.pos[1]), int(p2[2] + grand_node.pos[2]))
                    if ImageUtils.bboxCheck3D(z_pos, child_pos[1], child_pos[2], child_r, mark_shape):
                        node_count += 1
                        child_node = Vertex(node_count, 0, z_pos, child_pos[1], child_pos[2], child_r,
                                            root_node.idx)
                        # 绘制
                        setMarkWithSphere(mark, Sphere(Point3D(*child_node.pos), child_node.r), mark_shape)
                        setMarkWithCone(mark, Cone(Point3D(*root_node.pos), root_node.r, Point3D(*child_node.pos),
                                                   child_node.r), mark_shape)

                        # Add to dequeue
                        dq.append((child_node, root_depth + 1, False))

    mark = mark.astype(np.uint8)

    # save swc
    # print(id_list)
    # print(pos_x_list)
    # print(pos_y_list)
    # print(pos_z_list)
    # print(r_list)
    # print(pid_list)

    z_pos_adjust = 0

    swc_data = np.zeros([len(id_list), 7])
    for i in range(swc_data.shape[0]):
        swc_data[i][0] = id_list[i]
        swc_data[i][1] = 0
        swc_data[i][2] = pos_x_list[i]
        swc_data[i][3] = pos_y_list[i]
        swc_data[i][4] = z_pos_adjust # pos_z_list[i]
        swc_data[i][5] = r_list[i]
        swc_data[i][6] = pid_list[i]

    # external feature
    img_sim = copy.deepcopy(mark)
    img_fg_pos = np.where(mark == 1)

    for i in range(len(img_fg_pos[0])):
        is_fg = np.random.choice([0, 1], p=[0.5, 0.5])
        if is_fg == 1:
            z = img_fg_pos[0][i]
            x = img_fg_pos[1][i]
            y = img_fg_pos[2][i]
            img_sim[z][x][y] = 0
    if data_type == np.uint8:  # 8bit image
        if external_feature == True:
            forground = np.random.randint(50, 300)
            background_mean = np.random.randint(1, 20)
            background_std = np.random.randint(0, 5)
        else:
            forground = np.random.randint(20, 250)
            background_mean = 1
            background_std = 0
    else:  # 16bit image
        if external_feature == True:
            forground = np.random.randint(100, 1000)
            background_mean = np.random.randint(1, 200)
            background_std = np.random.randint(0, 20)

        else:
            forground = np.random.randint(100, 1000)
            background_mean = 1
            background_std = 0

    if external_feature == True:
        # foreground -> poisson
        img_sim_fg = np.random.poisson(lam=img_sim * forground)
        img_sim_fg = ndfilter.gaussian_filter(img_sim_fg, [1, 1, 1])

        # background -> add noise
        img_sim_fg = img_sim_fg + add_noise(MAX_BOX_WIDTH, mark, forground)
    else:
        img_sim_fg = mark * forground

    # print(img_sim_fg.shape)

    image = gaussianNoisyAddGray3D(img_sim_fg, background_mean, background_std, data_type)

    print('forground: %3f ,background mean: %3f ,background std: %3f' % (
    forground * 0.5, background_mean, background_std))

    if data_type == np.uint16:
        image = normalizeImage16(image)
    else:
        image = normalizeImage8(image)

    return image[10:11], mark[10:11], swc_data
