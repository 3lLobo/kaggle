# Create a sql lite database to store the point clouds. The schema will contain the xyz coordinates, the rgb values, the label and the piece id, Schema:
#   CREATE TABLE IF NOT EXISTS point_clouds (
#       id INTEGER PRIMARY KEY AUTOINCREMENT,
#       x REAL NOT NULL,
#       y REAL NOT NULL,
#       z REAL NOT NULL,
#       r REAL NOT NULL,
#       g REAL NOT NULL,
#       b REAL NOT NULL,
#       label INTEGER NOT NULL,
#   );

# Imports

from typing import List, Tuple
import sqlite3
import os
import numpy as np
import pandas as pd
import tqdm
from time import time, sleep
import cupy as cp
from cupy import sparse
import cv2
from threading import Thread


# piece_id = 1
# data_type = 'train'
# img_labels = cv2.imread('./data/{}/{}/inklabels.png'.format(data_type, piece_id), cv2.IMREAD_GRAYSCALE)
# img_labels = np.where(img_labels == 255, 1, 0)
# img_labels = cp.asarray(img_labels, dtype=cp.float64)
# img_labels = None




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
    def __init__(self, piece_id: int, data_type: str):
        """Initialize the class.

        Args:
            piece_id (int): The piece id.
            data_type (str): The data type.

        """
        # create npy folder
        if not os.path.exists('./data/{}/{}/npy_points/'.format(data_type, piece_id)):
            os.makedirs('./data/{}/{}/npy_points/'.format(data_type, piece_id))
            
        self.piece_id = piece_id
        self.data_type = data_type

        self.table_name = 'pc_{}_{}'.format(data_type, piece_id)
        self.db_name = './data/vesuvius_pointcloud_int.db'
        self.con = sqlite3.connect(self.db_name)
        self.cur = self.con.cursor()
        self.sql_create = 'CREATE TABLE IF NOT EXISTS {} (x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL, r REAL NOT NULL, g REAL NOT NULL, b REAL NOT NULL, label INTEGER NOT NULL)'.format(self.table_name)
        self.sql_insert = 'INSERT INTO {} (x, y, z, r, g, b, label) VALUES (?, ?, ?, ?, ?, ?, ?)'.format(self.table_name)
        self.sql_select = 'SELECT * FROM {}'.format(self.table_name)
        self.sql_delete = 'DELETE FROM {}'.format(self.table_name)

    def set_img_labels(self, piece_id: int, data_type: str):
        """Get image labels.

        Args:
            piece_id (int): The piece id.
            data_type (str): The data type.
        """
        img_labels = cv2.imread('./data/{}/{}/inklabels.png'.format(data_type, piece_id), cv2.IMREAD_GRAYSCALE)
        img_labels = np.where(img_labels == 255, 1, 0)
        img_labels = cp.asarray(img_labels, dtype=cp.float64)
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
        labels = self.img_labels[xy[:, 0], xy[:, 1]].reshape(-1, 1)
        # normalize RGB values
        rgb = rgb / 255 - 0.5
        # get 3d coordinates
        z = cp.ones((xy.shape[0], 1), dtype=cp.float64) * n_img
        xyz = cp.concatenate((xy, z), axis=1)

        xyzrgbl = cp.concatenate((xyz, rgb, labels), axis=1)
        xyzrgbl = cp.asnumpy(xyzrgbl)
        # Free GPU memory
        del img_gray, xy, img, rgb, labels, z, xyz
        return xyzrgbl


    def loop_gpu(self, img_nlist: List[Tuple]) -> None:
        """Loop for gpu.
        send points to elastic search

        Args:
            img_nlist (List[Tuple]): List of path and depth.

        """
        # Create database
        # con = sqlite3.connect(self.db_name)
        # cur = con.cursor()
        # cur.execute(self.sql_create)

        for img_path, i in tqdm.tqdm(img_nlist, colour='cyan'):
            t_start = time()
            xyzrgbl = self.get_3dimg_gpu(img_path, i)
            tqdm.tqdm.write('Image {} has {} points.'.format(i, xyzrgbl.shape[0]))
            # Convert to int
            # xyzrgbl = xyzrgbl.astype(np.int32)
            # Down-sample points
            xyzrgbl = xyzrgbl[::10, :]
            # Add to database
            # cur.executemany(self.sql_insert, xyzrgbl)
            # if i % 10 == 0:
            #     con.commit()
            # for j in tqdm.trange(xyzrgbl.shape[0], colour='green'):
            #     cur.execute(sql_insert, xyzrgbl[j, :])
            # con.commit()
            np.save(img_path.replace('tiff', 'npy').replace('surface_volume', 'npy_points'), xyzrgbl)
            tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - t_start))
        # con.commit()
        # con.close()
        # img_stack = np.concatenate(img_stack, axis=0)

        # return img_stack


# def get_db(piece_id: int, data_type: str= 'train') -> sqlite3.Connection:
#     """Get database connection.
#     Create the db if necessary.

#     Args:
#         piece_id (int): The piece id.
#         data_type (_type_): The data type.

#     Returns:
#         sqlite3.Connection: Connection to the database.
#     """
#     db_name = './data/vesuvius_pointcloud.db'
#     con = sqlite3.connect(db_name)
#     cur = con.cursor()
#     #  Create tables if they don't exist
#     table_name = 'pc_{}_{}'.format(data_type, piece_id)
#     cur.execute('CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY AUTOINCREMENT, x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL, r REAL NOT NULL, g REAL NOT NULL, b REAL NOT NULL, label INTEGER NOT NULL);'.format(table_name))
#     return con
    

# def insert_point_db(con: sqlite3.Connection, xyz: np.ndarray, rgb: np.ndarray, piece_id: int, data_type: str= 'train') -> None:
#     """Insert point cloud into database.

#     Args:
#         con (sqlite3.Connection): Connection to the database.
#         xyz (np.ndarray): 3d coordinates.
#         rgb (np.ndarray): RGB values.
#         piece_id (int): The piece id.
#         data_type (_type_): The data type.
#     """
#     table_name = 'pc_{}_{}'.format(data_type, piece_id)
#     cur = con.cursor()
#     for i in tqdm.tqdm(range(xyz.shape[0]), colour='cyan'):
#         cur.execute('INSERT INTO {} (x, y, z, r, g, b, label) VALUES (?, ?, ?, ?, ?, ?, ?);'.format(table_name), (xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2], 0))
#     con.commit()


if __name__ == '__main__':
    
    data_type = 'train'
    for piece_id in tqdm.trange(1,4, colour='red'):
        img_path = './data/{}/{}/surface_volume/'.format(data_type, piece_id)
        img_nlist = get_img_list(img_path)

        t_start_gpu = time()
        pcdb = PointCloudDB(piece_id, data_type)
        pcdb.set_img_labels(piece_id, data_type)
        pcdb.loop_gpu(img_nlist)
        tqdm.tqdm.write('Time elapsed: {:.2f} s\n'.format(time() - t_start_gpu))

        # Purge GPU memory
        # cp.get_default_memory_pool().free_all_blocks()
        del pcdb
