import os
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.transforms import default_transforms, ToTensor
from time import sleep

class PointCloudData(Dataset):
    # TODO: addapt to sql database
    def __init__(self, batch_dir, len_dataset, valid=False, transform=default_transforms()):
        self.batch_dir = batch_dir
        self.len_dataset = len_dataset
        self.classes = 2
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        # TODO: len should be the number of points times x. X is the probability of all points being sampled at least once given the batch size.
        self.batch_files = self.get_batch_files()
        self.toTensor = ToTensor()

    def get_batch_files(self):
        batch_files = []
        for file in os.listdir(self.batch_dir):
            if file.endswith('.npy'):
                batch_files.append(os.path.join(self.batch_dir, file))
        return batch_files

    def __len__(self):
        return self.len_dataset

    def __preproc__(self, pointcloud):
        if self.transforms:
            pointcloud = self.transforms(pointcloud)
        return pointcloud

    def __getitem__(self, idx):
        while len(self.batch_files) == 0:
            sleep(.1)
            self.batch_files = self.get_batch_files()

        # Load pointcloud
        batch_file = self.batch_files.pop()
        batch = np.load(batch_file)
        # Delete file
        os.remove(batch_file)
        # Split into pointcloud, features and label
        pointcloud = batch[:, :3]
        features = batch[:, 3:6]
        if not self.valid:
            label = batch[:, 6]
        else:
            label = np.zeros((pointcloud.shape[0], 1))
        pointcloud = self.__preproc__(pointcloud)
        pointcloud = np.concatenate((pointcloud, features), axis=1)
        pointcloud = self.toTensor(pointcloud)

        return pointcloud, label