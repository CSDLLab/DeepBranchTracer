import numpy as np
from lib.klib.baseio import *



def gen_circle_2d(size=32, r=5, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    image = ((x - x0)**2 + (y-y0)**2) <= r**2

    return image.astype(np.float32)

def gen_circle_3d(size=32, r=5.0, z_offset=0, x_offset=0, y_offset=0):
    z0 = x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    z0 += z_offset
    z, x, y = np.ogrid[:size, :size, :size]
    
    image = ((x - x0)**2 + (y-y0)**2 + (z-z0)**2) <= r**2

    return image.astype(np.float32)

def gen_circle_gaussian_2d(size=32, r=5, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    image = 1*np.exp(-(((x-x0)**2 /(2*r**2)) + ((y-y0)**2 /(2*r**2))))#*2-1
    
    # K=1
    # image = 1 / (1 + np.exp(-K*image))

    return image.astype(np.float32)

def gen_circle_gaussian_3d(size=32, r=5.0, z_offset=0, x_offset=0, y_offset=0):
    z0 = x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    z0 += z_offset
    z, x, y = np.ogrid[:size, :size, :size]
    
    image = 1*np.exp(-(((x-x0)**2 /(2*r**2)) + ((y-y0)**2 /(2*r**2))+ ((z-z0)**2 /(2*r**2))))

    return image.astype(np.float32)

# usage 
# data_image_dir = 'data/test/temp/test.tif'
# img = gen_circle_3d(size=32, r=5.32, z_offset=0, x_offset=6, y_offset=0)*255
# save_tif(img, data_image_dir, np.uint16)




















