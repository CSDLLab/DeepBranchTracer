import math
import glob
import numpy as np
from . import TiffFile
from .glib.SWCExtractor import Vertex
from .glib.Obj3D import Point3D, Sphere, Cone, calculateBound, calScaleRatio
from numpy import linalg as LA
from scipy.spatial import distance_matrix
import copy
import cv2 as cv

def open_swc(file_name):
	return SWC(file_name)

def open_tif(file_name):
	'''
		return numpy array
		z,y,x
	'''
	return TiffFile.imread(file_name)
	
def save_tif(np_array_like,file_name,type, compress = 0):
	'''
		save numpy array like to tiff 
		z,y,x
	'''
	TiffFile.imsave(file_name,np_array_like.astype(type), compress = compress)
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
    # points=img_idxs[:3, xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] # 3*M
    # points=points.T # M*3
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
        # value_list = randIntList(lower,upper,len(res_idxs))
        for pos in res_idxs:
            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
        # mark[xmin+x_idxs[res_idxs], ymin+y_idxs[res_idxs], zmin+z_idxs[res_idxs]] = 255
    else:
        # value_list = randIntList(lower,upper,len(res_idxs))
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):
            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 255
        # mark[xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] = 255



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

        # 每个圆锥的还原矩阵
        r_min=cone.up_radius
        r_max=cone.bottom_radius
        height=cone.height
        cone_revert_mat = cone.revertMat().T # 4*4

        # 每个椎体还原后坐标
        revert_coor_mat = np.matmul(points, cone_revert_mat) # M*4
        revert_radius_list = LA.norm(revert_coor_mat[:,:2], axis=1) # M

        # Local Indexs
        M = points.shape[0]
        l_idx = np.arange(M) # M (1-dim)
        l_mark = np.ones((M,), dtype=bool)

        # 过滤高度在外部的点
        res_idxs = np.logical_or(revert_coor_mat[l_idx[l_mark],2]<0, revert_coor_mat[l_idx[l_mark],2]>height)
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 过滤半径在外部的点
        res_idxs = revert_radius_list[l_idx[l_mark]]>r_max
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 过滤半径在内部的点
        res_idxs = revert_radius_list[l_idx[l_mark]]<=r_min
        # value_list = randIntList(lower,upper,len(l_idx[l_mark][res_idxs]))
        for pos in l_idx[l_mark][res_idxs]:

            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
        # mark[xmin+x_idxs[l_idx[l_mark][res_idxs]], ymin+y_idxs[l_idx[l_mark][res_idxs]], zmin+z_idxs[l_idx[l_mark][res_idxs]]] = 255
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 计算剩余
        if r_max>r_min:
            res_idxs = ((r_max-revert_radius_list[l_idx[l_mark]])*height/(r_max-r_min)) >= revert_coor_mat[l_idx[l_mark],2]
            # value_list = randIntList(lower,upper,len(l_idx[l_mark][res_idxs]))
            for pos in l_idx[l_mark][res_idxs]:

                mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
            # mark[xmin+x_idxs[l_idx[l_mark][res_idxs]], ymin+y_idxs[l_idx[l_mark][res_idxs]], zmin+z_idxs[l_idx[l_mark][res_idxs]]] = 255
            l_mark[l_idx[l_mark][res_idxs]]=False
    else:
        # value_list = randIntList(lower,upper,len(x_idxs))
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):

            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 255
        # mark[xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] = 255


#
def save_swc2tif(swc_file_name, shape, transform_ratio=1.0):
	swc = open_swc(swc_file_name)
	mark = np.zeros([int(shape[0]/transform_ratio),shape[1],shape[2]], dtype=np.uint8)
	mark_new = np.zeros([shape[0],shape[1],shape[2]], dtype=np.uint8)
	mark_shape = ((int(shape[0]/transform_ratio) + 1), shape[1] + 1, (shape[2] + 1))
	# Create root node

	for node in swc.nodes:
		#print(node.id, node.x,node.y,node.z, node.r, node.pid)
		parent_id = node.pid

		if parent_id >0:
			node_p = swc.get_node(parent_id)

			current_node = Vertex(node.id, 0, node.z/transform_ratio, node.y, node.x, node.r, node.pid)
			setMarkWithSphere(mark, Sphere(Point3D(*current_node.pos), current_node.r), mark_shape)

			parent_node = Vertex(node_p.id, 0, node_p.z/transform_ratio, node_p.y, node_p.x, node_p.r, node_p.pid)
			setMarkWithCone(mark, Cone(Point3D(*parent_node.pos), parent_node.r, Point3D(*current_node.pos), current_node.r), mark_shape)

		else:
			current_node = Vertex(node.id, 0, node.z/transform_ratio, node.y, node.x, node.r, node.pid)
			setMarkWithSphere(mark, Sphere(Point3D(*current_node.pos), current_node.r), mark_shape)

	for i in range(shape[2]):
		temp_slice = copy.deepcopy(mark[:,:,i:i+1])
		label_temp_1 = cv.resize(temp_slice, (int(shape[1]), int(shape[0])))
		label_temp_2 = copy.deepcopy(label_temp_1.reshape((int(shape[0]), int(shape[1]), 1)))
		mark_new[:, :, i:i + 1] = copy.deepcopy(label_temp_2).astype(np.uint8)

	return mark_new

# def save_swc2tif(swc_file_name,tif_file_name,shape=None):
# 	swc=open_swc(swc_file_name)
# 	bound_box=swc.get_bound_box()
# 	width=math.ceil(bound_box[3]-bound_box[0])
# 	height=math.ceil(bound_box[4]-bound_box[1])
# 	depth=math.ceil(bound_box[5]-bound_box[2])
# 	if shape:
# 	    depth,height,width=shape
# 	tif=np.zeros(shape=(depth,height,width))
# 	for node in swc.nodes:
# 		_sphere(tif,node.x,node.y,node.z,node.r,255)
# 	for edge in swc.edges:
# 		if edge[0]!=-1 and edge[1]!=-1:
# 			a,b=swc.get_node(edge[0]),swc.get_node(edge[1])
# 			_cone(tif,a.x,a.y,a.z,a.r,b.x,b.y,b.z,b.r,255)
# 	save_tif(tif,tif_file_name)
	
class SWCNode(object):
	
	def __init__(self,id,type,x,y,z,r,pid):
		self.id=id
		self.type=type
		self.x=x
		self.y=y
		self.z=z
		self.r=r
		self.pid=pid
		
class SWC(object):
	
	def __init__(self,file_name=None):
		self.__nodes={}
		self.__edges=[]
		self.bound_box=[0,0,0,0,0,0]# x0,y0,z0,x1,y1,z1
		if file_name:
			self.open(file_name)
			
	def open(self,file_name):
		with open(file_name) as f:
			for line in f.readlines():
				if line.startswith('#'):
					continue
				id,type,x,y,z,r,pid=map(float,line.split())
				self.__nodes[id]=SWCNode(id,type,x,y,z,r,pid)
				self.__edges.append((id,pid))
				if x<self.bound_box[0]:
					self.bound_box[0]=x
				if x>self.bound_box[3]:
					self.bound_box[3]=x
				if y<self.bound_box[1]:
					self.bound_box[1]=y
				if y>self.bound_box[4]:
					self.bound_box[4]=y
				if z<self.bound_box[2]:
					self.bound_box[2]=z
				if z>self.bound_box[5]:
					self.bound_box[5]=z
					
	def get_node(self,id):
		return self.__nodes[id]
		
	def get_bound_box(self):
		return self.bound_box
		
	@property
	def nodes(self):
		return self.__nodes.values()
	
	@property
	def edges(self):
		return self.__edges
	
def _sphere(stack,x,y,z,r,v=1):
	d,h,w=stack.shape
	for k in range(int(max(0,z-r)),int(min(d,z+r))):
		for j in range(int(max(0,y-r)),int(min(h,y+r))):
			for i in range(int(max(0,x-r)),int(min(w,x+r))):
				if (k-z)**2+(j-y)**2+(i-x)**2<=r**2:
					stack[k,j,i]=v
	
def _cone(stack,x0,y0,z0,r0,x1,y1,z1,r1,v=1):
	d,h,w=stack.shape
	'''if r0<r1:
		x0,x1=x1,x0
		y0,y1=y1,y0
		z0,z1=z1,z0
		r0,r1=r1,r0'''
	a,b,c=x1-x0,y1-y0,z1-z0
	r=math.sqrt(a**2+b**2+c**2)+1e-6
	tan_theta=(r0-r1)/r
	points=[]
	for k in range(math.ceil(r)):
		rk=tan_theta*(r-k)+r1
		for j in range(-int(rk)-1,int(rk)+1):
			for i in range(-int(rk)-1,int(rk)+1):
				if j**2+i**2<rk**2:
					points.append((i,j,k))
	n1=math.sqrt(b**2+c**2)
	if n1==0:
		for pt in points:
			x=int(pt[2]+x0)
			y=int(pt[1]+y0)
			z=int(pt[0]+z0)
			if x>=0 and x<w and y>=0 and y<h and z>=0 and z<d:
				stack[z,y,x]=v
	else:	
		n2=math.sqrt((b**2+c**2)**2+(a*b)**2+(a*c)**2)
		Tr=np.array(((0,c/n1,-b/n1),(-(b**2+c**2)/n2,a*b/n2,a*c/n2),(a/r,b/r,c/r)))
		Tr=Tr.T
		for pt in points:
			coord=np.matmul(Tr,pt)
			coord=coord+(x0,y0,z0)
			for x in range(math.floor(coord[0]),math.ceil(coord[0])+1):
				for y in range(math.floor(coord[1]),math.ceil(coord[1])+1):
					for z in range(math.floor(coord[2]),math.ceil(coord[2])+1):
						if x>=0 and x<w and y>=0 and y<h and z>=0 and z<d:
							stack[z,y,x]=v
	
# if __name__=='__main__':
#     for swc_file in glob.glob('./*.swc'):
# 	    save_swc2tif(swc_file,swc_file+'.tif',(85,2048,2048))
