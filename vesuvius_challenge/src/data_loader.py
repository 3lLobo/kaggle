import os
import torch
import numpy as np
import os
import torch
# from torch.utils.data import Dataset
from src.transforms import default_transforms, ToTensor
from time import sleep
from torch_geometric.data import Batch, Data, Dataset
from typing import List, Tuple


class PointCloudDataV2(Dataset):
    def __init__(self, data_dir, n_points, is_test: bool = False, do_transform: bool = False, transform: List = default_transforms(), is_unify: bool = True):
        root_dir = data_dir
        self.root = root_dir
        self.data_dir = data_dir
        self.n_points = n_points
        self.is_test = is_test
        self.is_unify = is_unify

        self.num_classes = 2
        # self.num_features = 3
        # self.num_node_features = 3

        self.do_transform = do_transform
        self.transforms = transform
        self.files = []
        self.get_files()

        if not is_test:
            self.label_path = 'data/train/{pc_idx}/inklabels.npy'
            labels = []
            for i in range(3):
                img = np.load(self.label_path.format(pc_idx=i+1))
                labels.append(img)
            self.labels = labels

        super().__init__(root=root_dir, transform=None, pre_transform=None, pre_filter=None)
        
    def download(self):
        pass

    def process(self):
        pass

    def len(self) -> int:
        return len(self.files)
    @property
    def raw_file_names(self):
        return [f'pointcloud_{i}.npy' for i in range(len(self.files))]

    @property
    def processed_file_names(self):
        return self.raw_file_names
    

    def get_files(self):
        self.files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.npy'):
                self.files.append(os.path.join(self.data_dir, file))

    def __len__(self):
        return len(self.files)
    
    # @abstractmethod
    def __getitem__(self, idx: int) -> Data:
        file_path = self.files[idx]
        pointcloud = np.load(file_path)
        if self.is_unify:
            pointcloud = self.unify_num_points(pointcloud)

        labels = self.get_labels(file_path, pointcloud[:, :2])

        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:6]
        xyz = default_transforms()(xyz).unsqueeze(0)
        rgb = default_transforms()(rgb).unsqueeze(0)
        # if self.do_transform:
        #     for t in self.transforms:
        #         pointcloud = t(pointcloud)
        # xyz = pointcloud[:, :3].unsqueeze(0)
        # rgb = pointcloud[:, 3:6].unsqueeze(0)

        # xyz = SparseTensor.from_dense(xyz)

        label_dtype = torch.long if self.is_test else torch.float
        labels = torch.tensor(labels, dtype=label_dtype).unsqueeze(0)

        data = Data(pos=xyz, x=rgb, y=labels)

        return data
    
    def get(self, idx: int) -> Data:
        return self[idx]
    
    def unify_num_points(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        Unify the number of points in the point cloud.
        
        Args:
            pointcloud (np.ndarray): The original point cloud.

        Returns:
            np.ndarray: The point cloud with a unified number of points.
        """
        if pointcloud.shape[0] < self.n_points:
            n_missing = self.n_points - pointcloud.shape[0]
            patch = pointcloud[:n_missing]
            pointcloud = np.concatenate((pointcloud, patch), axis=0)
        elif pointcloud.shape[0] > self.n_points:
            pointcloud = pointcloud[:self.n_points]
        return pointcloud

    
    def get_labels(self, file_path: str, xy: np.ndarray) -> np.ndarray:
        """Get the labels for the corresponding xy coordinates.

        Args:
            file_path (str): File path to the point cloud.
            xy (np.ndarray): xy coordinates of the points.

        Returns:
            np.ndarray: Labels for the points.
        """
        if self.is_test:
            return xy
        pc_idx = int(file_path.split('_')[-2]) - 1
        labels = self.labels[pc_idx]
        labels = labels[xy[:, 0], xy[:, 1]]
        labels = labels.reshape(1, -1)
        return labels
        


class PointCloudData(Dataset):
    def __init__(self, batch_dir, n_points, len_dataset, n_skip_rm, valid=True, test_split: float = 0.2, transform=default_transforms()):
        self.batch_dir = batch_dir
        self.n_points = n_points
        self.len_dataset = len_dataset
        self.classes = 2
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.test_split = test_split
        self.files = []
        # TODO: len should be the number of points times x. X is the probability of all points being sampled at least once given the batch size.
        self.get_batch_files()
        self.batch_files = self.train_files
        self.test_mode = False
        self.toTensor = ToTensor()
        self.n_skip_rm = n_skip_rm
        self.n_skiped = 1

        self.storage = []

        self.build_batch = None

    def set_test_mode(self, test_mode):
        self.test_mode = test_mode
        self.batch_files = self.test_files if test_mode else self.train_files

    def get_batch_files(self):
        batch_files = []
        for file in os.listdir(self.batch_dir):
            if file.endswith('.npy'):
                batch_files.append(os.path.join(self.batch_dir, file))
        
        n_split = int(len(batch_files) * self.test_split)
        if self.valid:
            self.train_files = batch_files[:n_split]
            self.test_files = batch_files[n_split:]
        else:
            self.train_files = batch_files
            self.test_files = []

    def __len__(self):
        n_per_file = 1024 / self.n_points
        if self.test_mode:
            return int(len(self.test_files) * n_per_file)
        else:
            return int(len(self.train_files) * n_per_file)

    def __preproc__(self, pointcloud):
        if self.transforms:
            pointcloud = self.transforms(pointcloud)
        return pointcloud

    def __getitem__(self, idx):
        # Check if there are batches in storage

        if len(self.storage) > 0:
            batch = self.storage.pop()

        else:
            if len(self.batch_files) == 0:
                print('Epoch finished, waiting for new batches.')
                while len(self.batch_files) == 0:
                    sleep(.1)
                    self.get_batch_files()

            # Load pointcloud
            batch_file = self.batch_files.pop()
            batch = np.load(batch_file)
            # Delete file
            if self.n_skiped % self.n_skip_rm == 0:
                os.remove(batch_file)
            self.n_skiped += 1
        if self.build_batch is None:
            self.build_batch = batch
        else:
            batch = np.concatenate((self.build_batch, batch), axis=0)
            self.build_batch = batch
        #  check for too big batches
        if batch.shape[0] > self.n_points:
            n_excess = batch.shape[0] % self.n_points
            m_batch = np.split(batch[:-n_excess, :], batch.shape[0] // self.n_points)
            batch = m_batch.pop()
            self.storage.extend(m_batch)

        if batch.shape[0] < self.n_points:
            n_missing = self.n_points - batch.shape[0]
            add_batch = self.__getitem__(idx)
            self.build_batch = None
            return add_batch


        # Split into pointcloud, features and label
        pointcloud = batch[:, :3]
        # original_xy = pointcloud[:, :2]
        features = batch[:, 3:6]
        if not self.valid:
            label = batch[:, 6].reshape(1, -1)
        else:
            label = np.zeros((1, pointcloud.shape[0]))
        pointcloud = self.__preproc__(pointcloud)
        # pointcloud = np.concatenate((pointcloud, features), axis=1)
        pointcloud = self.toTensor(pointcloud).unsqueeze(0)
        # label = self.toTensor(label)
        features = self.toTensor(features).unsqueeze(0)
        label = self.toTensor(label).unsqueeze(0)
        data = Data(pos=pointcloud, x=features, y=label)

        return data
    
    def get_batch(self, n_pcs):
        batch = []
        for i in range(n_pcs):
            batch.append(self.__getitem__(i))
        return Batch.from_data_list(batch)
