from typing import List, Tuple
import sqlite3
import os
import shutil
import numpy as np
import pandas as pd
import tqdm
from time import time, sleep
import cupy as cp
from cupy import sparse
import cv2
from threading import Thread
import asyncio



async def load_np_file(np_file_path: str) -> np.ndarray:
    """Load a numpy file.

    Args:
        np_file_path (str): Path to numpy file.

    Returns:
        np.ndarray: The numpy array.
    """
    np_file = np.load(np_file_path)
    return await np_file

def get_img_list(img_path: str) -> List[Tuple]:
    """Get list of images.

    Args:
        img_path (str): Path to image folder.

    Returns:
        List[Tuple]: List of image paths and image numbers.
    """
    # Load all images and craete a binary (d,x,y,z) matrix and a corresponding nRGB array.
    img_list = os.listdir(img_path)
    img_list.sort()
    img_list = [img_path + img for img in img_list]
    img_nlist = list(zip(img_list, range(len(img_list))))
    return img_nlist

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

# class to store the labels and other meta imformation
class PointCloudDB:
    def __init__(self, piece_id: int, data_type: str, n_points: int =1024, to_int: bool = False, normalize: bool = True):
        """Initialize the class.

        Args:
            piece_id (int): The piece id.
            data_type (str): The data type.
            n_points (int, optional): Number of points per pointcloud. Defaults to 1024.
            to_int (bool, optional): Save as int. Defaults to False.
            normalize (bool, optional): Normalize RGB values. Defaults to True.
        """
        # create npy folder
        if not os.path.exists('./data/{}/{}/npy_points/'.format(data_type, piece_id)):
            os.makedirs('./data/{}/{}/npy_points/'.format(data_type, piece_id))
            
        self.piece_id = piece_id
        self.data_type = data_type
        self.n_points = n_points
        self.to_int = to_int
        self.normalize = normalize
        self.is_test = True if data_type == 'test' else False

        # self.get_mask()
        # self.set_img_labels()

    def get_mask(self):
        """Get the mask for the images with np.where.
        """
        mask_path = './data/{}/{}/mask.png'.format(self.data_type, self.piece_id)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 255)
        self.mask = mask

    def set_img_labels(self,):
        """Get image labels.
        """
        if self.is_test:
            return
        piece_id = self.piece_id
        data_type = self.data_type
        img_labels = cv2.imread('./data/{}/{}/inklabels.png'.format(data_type, piece_id), cv2.IMREAD_GRAYSCALE)
        img_labels = np.where(img_labels == 255, 1, 0)
        # apply mask
        img_labels = img_labels[self.mask]

        self.img_labels = img_labels

    def get_3dimg_gpu(self, img_path: str, n_img: int) -> np.ndarray:
        """
        Load an image and return a stacked array of coordinates, features and labels.
        
        Args:
            img_path (str): image path.
            n_img (int): image number. 
            
        Returns:
            np.ndarray: Stacked array of coordinates, features and labels.
        """
        img = cv2.imread(img_path)
        # convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cp.asarray(img_gray, dtype=cp.float64)
        # get dense matrix
        xy = cp.argwhere(img_gray > 0)
        # get RGB values and flatten
        img = cp.asarray(img, dtype=cp.float64)
        rgb = img[xy[:, 0], xy[:, 1], :].reshape(-1, 3)
        # normalize RGB values
        if self.normalize:
            rgb = rgb / 255 - 0.5
        # get 3d coordinates
        z = cp.ones((xy.shape[0], 1), dtype=cp.float64) * n_img
        xyz = cp.concatenate((xy, z), axis=1)

        if False:
        # if self.is_test:
            img_labels = cp.asarray(self.img_labels, dtype=cp.float64)
            labels = img_labels[xy[:, 0], xy[:, 1]].reshape(-1, 1)
            xyzrgbl = cp.concatenate((xyz, rgb, labels), axis=1)
        else:
            xyzrgbl = cp.concatenate((xyz, rgb), axis=1)

        np_xyzrgbl = cp.asnumpy(xyzrgbl)
        return np_xyzrgbl

    async def save_npy(self, npy_data: np.ndarray, npy_path: str) -> None:
        """Save npy data.

        Args:
            npy_data (np.ndarray): Numpy array.
            npy_path (str): Path to save.

        """
        np.save(npy_path, npy_data) 

    def loop_gpu(self, img_nlist: List[Tuple]) -> None:
        """Loop for gpu.
        send points to elastic search

        Args:
            img_nlist (List[Tuple]): List of path and depth.

        """

        self.n_files = 65

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        np_base_path = 'data/{}/{}/npy_clouds'.format(self.data_type, self.piece_id)
        self.np_base_path = np_base_path
        np_file_path = self.np_base_path + '/pcld_{j}_{i}.npy'
        self.np_file_path = np_file_path

        # for i in tqdm.trange(n_pointclouds, colour='green'):
        for img_path, i in tqdm.tqdm(img_nlist, colour='cyan'):
            t_start = time()
            xyzrgbl = self.get_3dimg_gpu(img_path, i)
            tqdm.tqdm.write('Image {} has {} points.'.format(i, xyzrgbl.shape[0]))
            # Convert to int
            if self.to_int:
                xyzrgbl = xyzrgbl.astype(np.int32)

            n_pointclouds = xyzrgbl.shape[0] / self.n_points
            n_pointclouds = int(np.ceil(n_pointclouds))


            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            # await asyncio.gather(*[self.save_npy(xyzrgbl[j::n_pointclouds, :], np_file_path.format(j=j, i=i)) for j in range(n_pointclouds)])

            for j in tqdm.trange(n_pointclouds, colour='red'):
                np_save = xyzrgbl[j::n_pointclouds, :]
                np_path = np_file_path.format(j=j, i=i)
                np.save(np_path, np_save)
            
            tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - t_start))

    async def get_and_del_npy_files(self, npy_path: str, is_rm: bool = True) -> np.ndarray:
        """Get and delete a npy file.

        Args:
            npy_path (str): Path to the npy file.
            is_rm (bool, optional): If to remove the original file. Defaults to True.

        Returns:
            np.ndarray: Numpy array.
        """
        if npy_path is None:
            return
        npy_path = os.path.join(self.np_base_path, npy_path)
        npy = np.load(npy_path)
        if is_rm:
            os.remove(npy_path)
        return npy
    
    async def name_in_path(self, path: str, name: str) -> bool:
        """Check if a name is in the path.

        Args:
            path (str): Path.
            name (str): Name.

        Returns:
            bool: If the name is in the path.
        """
        return name in path

    async def concat_npy(self, is_rm: bool = True) -> None:
        """Concatenate npy files.
        Remove original files.

        Args:
            is_rm (bool, optional): If to remove the original files. Defaults to True.
        """
        if self.np_base_path is None:
            raise ValueError('Numpy path is not set.')
    
        j = 0
        data_folder = f'data/pointclouds/{self.data_type}'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        path_pointcloud = data_folder + f'/pcld_{self.piece_id}' + '_{j}.npy'

        np_files = os.listdir(self.np_base_path)

        n_pclds = len(np_files) / self.n_files
        n_pclds = int(np.ceil(n_pclds))

        for n_pcld in tqdm.trange(n_pclds, colour='green', desc='Concatenating npy files'):

            pcld_files = []

            # np_files_lambda = lambda x: x if '_{n_pcld}_' in x else None
            # pcld_files_idx = await asyncio.gather(*[self.name_in_path(np_file, f'_{n_pcld}_') for np_file in np_files])
            # pcl_files = [np_file for np_file, idx in zip(np_files, pcld_files_idx) if idx]
            for np_file in np_files:
                if f'_{n_pcld}_' in np_file:
                    pcld_files.append(np_file)

            np_save = await asyncio.gather(*[self.get_and_del_npy_files(np_file, is_rm) for np_file in pcld_files])
            np_save = np.concatenate(np_save, axis=0)

            # np_save = np.load(pcld_files.pop())
            # for np_file in pcld_files:
            #     np_file = np.load(os.path.join(self.np_base_path, np_file))
            #     np_save = np.concatenate((np_save, np_file), axis=0)

            np.save(path_pointcloud.format(j=j), np_save)
            # np.save(path_pointcloud.format(j=j), np_concat)
            j += 1



if __name__ == '__main__':
    
    data_type = 'train'
    n_points = 10_000 # per img. Thus times 65 ir.

    # pcdb = PointCloudDB(2, 'train', n_points, to_int=True, normalize=False)
    # pcdb.n_files = 65
    # pcdb.np_base_path = 'data/train/2/npy_clouds'
    # time_concat = time()
    # asyncio.run(pcdb.concat_npy(is_rm=False))
    # tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - time_concat))

    for data_type in tqdm.tqdm(['train', 'test']):

        piece_ids = ['a', 'b'] if data_type == 'test' else [1,2,3]

        for piece_id in tqdm.tqdm(piece_ids, colour='red'):
            img_path = './data/{}/{}/surface_volume/'.format(data_type, piece_id)
            img_nlist = get_img_list(img_path)

            # # crop first 22 images
            # if data_type == 'train':
            #     if piece_id == 3:
            #         img_nlist = img_nlist[63:]

            np_path = img_path.replace('surface_volume', 'npy_clouds')
            if not os.path.exists(np_path):
                os.makedirs(np_path)
            else:
                shutil.rmtree(np_path)
                os.makedirs(np_path)

            t_start_gpu = time()
            pcdb = PointCloudDB(piece_id, data_type, n_points, to_int=True, normalize=False)
            # asyncio.run(pcdb.loop_gpu(img_nlist))
            pcdb.loop_gpu(img_nlist)
            tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - t_start_gpu))

            time_concat = time()
            asyncio.run(pcdb.concat_npy())
            tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - time_concat))
