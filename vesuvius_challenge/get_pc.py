from typing import List, Dict, Tuple
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import multiprocessing as mp
# from scipy import sparse
from cupy import sparse
# import numpy as np
import cupy as cp
import numpy as np
from time import time, sleep
# import numba
# from numba import jit, prange

img_path = './data/train/1/surface_volume/'

# Load all images and craete a binary (d,x,y,z) matrix and a corresponding nRGB array.
img_list = os.listdir(img_path)
img_list.sort()
img_list = [img_path + img for img in img_list]
img_nlist = list(zip(img_list, range(len(img_list))))

def load_gray_img(img_path: str) -> np.ndarray:
    """
    Load an image and return a list of 3d coordinates where the pixel is not black.
    
    Args:
        img_path (str): Path to image.

    Returns:
        img: The image.
    """
    img = cv2.imread(img_path)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# @jit(parallel=True) 
def get_3dimg(img_path: Tuple) -> np.ndarray:
    """
    Load an image and return a list of 3d coordinates where the pixel is not black.
    
    Args:
        img_args (Tuple): Tuple of image path and image number.
        
    Returns:
      img_stack: List of 3d coordinates.
    """
    img = load_gray_img(img_path)
    # get dense matrix
    img = cp.asarray(img, dtype=cp.float64)
    dense = sparse.csr_matrix(img)
    del img
    dense = sparse.find(dense)
    # get 3d coordinates
    xy = cp.array([dense[0], dense[1]]).T
    xy = cp.asnumpy(xy)
    return xy

# @jit(nopython=True, parallel=True)
def get_dense(img: np.ndarray) -> np.ndarray:
    """Get the dense bool matrix using where.

    Args:
        img (np.ndarray): The image.

    Returns:
        np.ndarray: xy coordinates.
    """
    dense = np.where(img > 0)
    dense = np.vstack(dense).reshape(-1, 2)
    return dense


# def wrapper(in_args):
#     return get_3dimg(*in_args)

# # @jit(parallel=True)
# def pool_3dimg(img_list: List[str]) -> np.ndarray:
#   """Parallelize get_3dimg() using multiprocessing.

#   Args:
#       img_list (List[str]): List of image paths.

#   Returns:
#       np.ndarray: The 3 dim array.
#   """
#   # img_stack = np.zeros((len(img_list), 3), dtype=np.int32)
#   # img_stack = np.zeros((len(img_list), 3), dtype=np.float64)
#   img_stack = []
#   # for img_path, i in tqdm.tqdm(img_nlist, colour='cyan'):
#   for i in prange(len(img_list)):
#       img_path = img_list[i]    
#       t_start = time()
#       # n_img = np.fromstring(img_path.split('/')[-1].split('.')[0], dtype=np.int32, sep='_')[0]
#       img = cv2.imread(img_path)
#       # convert to grayscale
#       img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#       # get dense matrix
#       # img = np.asarray(img, dtype=np.float64)
#       dense = sparse.csr_matrix(img)
#       dense = sparse.find(dense)
#       # get 3d coordinates
#       xyz = np.array([dense[0], dense[1]]).T
#       xyzd = np.concatenate((xyz, np.ones((xyz.shape[0], 1), dtype=np.int8) * i), axis=1)

#       print('Image {} has {} points.'.format(i, xyzd.shape[0]))
#       print('Time elapsed: {:.2f} s\n'.format(time() - t_start))
#       img_stack.extend(xyzd)

#   img_stack = np.array(img_stack).reshape(-1, 3, dtype=np.float64)
  
#   print('Image stack shape: ', img_stack.shape)
#   print('Image stack dtype: ', img_stack.dtype)
#   return img_stack

# t_start_pool = time()
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     print('Number of processes: ', mp.cpu_count())
#     img_stack = pool.map(get_3dimg, img_nlist)

#     sleep(0.1)
# img_stack = np.array(img_stack).reshape(-1, 3, dtype=np.float64)
# print('Pooling done in {:.2f} s'.format(time() - t_start_pool))

# @jit(nopython=True, parallel=True)
def np_wrapper(img: np.ndarray, i: int) -> np.ndarray:
    xy = get_dense(img)
    xy = xy[np.random.choice(xy.shape[0], int(xy.shape[0] * 0.1), replace=False), :]
    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1), dtype=np.float64) * i), axis=1)
    return xyz

# @jit(parallel=True)
def pool_3dimg(img_nlist: List[Tuple]) -> np.ndarray:
    """Loop for numba.

    Args:
        img_nlist (List[Tuple]): List of path and depth.

    Returns:
        np.ndarray: final array.
    """
    img_stack = []
    for img_path, i in img_nlist:
        img = load_gray_img(img_path)
        xyz = np_wrapper(img, i)
        img_stack.append(xyz)
    img_stack = np.concatenate(img_stack, axis=0)
    return img_stack

def loop_gpu(img_nlist: List[Tuple]) -> np.ndarray:
    """Loop for gpu.

    Args:
        img_nlist (List[Tuple]): List of path and depth.

    Returns:
        np.ndarray: final array.
    """
        
    img_stack = []
    for img_path, i in tqdm.tqdm(img_nlist, colour='cyan'):
        t_start = time()
        # n_img = np.fromstring(img_path.split('/')[-1].split('.')[0], dtype=np.int32, sep='_')[0]
        # xy = get_3dimg(img_path)
        xy = get_3dimg(img_path)
        # Sample 10% of the points
        xy = xy[np.random.choice(xy.shape[0], int(xy.shape[0] * 0.1), replace=False), :]
        xyz = np.concatenate((xy, np.ones((xy.shape[0], 1), dtype=np.float64) * i), axis=1)
        print('Image {} has {} points.'.format(i, xyz.shape[0]))
        print('Time elapsed: {:.2f} s\n'.format(time() - t_start))
        img_stack.append(xyz)
    img_stack = np.concatenate(img_stack, axis=0)
    return img_stack


t_start_gpu = time()
img_stack = loop_gpu(img_nlist)
print('GPU done in {:.2f} s'.format(time() - t_start_gpu))


# t_start_numba = time()
# img_stack = pool_3dimg(img_nlist)
# img_stack = np.concatenate(img_stack, axis=0)
# print('Numba done in {:.2f} s'.format(time() - t_start_numba))

# t_start_numba = time()
# img_stack = pool_3dimg(img_list)
# print('Numba done in {:.2f} s'.format(time() - t_start_numba))

# Checkpoint 1
np.save('./data/img_stack.npy', img_stack)

# # Print stats
print('Image stack shape: ', img_stack.shape)
print('Image stack dtype: ', img_stack.dtype)

# # Create RGB-D image
# img_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img_stack[0]), o3d.geometry.Image(img_stack[1][:, :, 0]), depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)

# # import numpy as np
# img_stack = np.array(img_stack).reshape(-1, 4)
# Point cloud from RGB-D image
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(img_stack)
o3d.io.write_point_cloud('./data/pcd1.ply', pcd)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])

# Save point cloud