from src.sql_data_loader import SQLDataLoader
import numpy as np
import os
import tqdm
from time import sleep

# Provide point clouds from the database by saving batches as npy files.
# If the number of files in the folder is less than the tjreshold, save more batches.

def provide_data_loop(db_path: str, pc_size:int, batch_size: int, piece_id: int, data_type: str, storage_factor: int = 10, threshold: int = 1000) -> None:
    """Provide batches of point clouds from the database.

    Args:
        db_path (str): Path to the database.
        pc_size (int): Size of the point cloud.
        batch_size (int): Batch size.
        piece_id (int): Id of the piece.
        data_type (str): Train or test.
        storage_factor (int, optional): Number of batches to store. Defaults to 10.
        threshold (int, optional): Threshold for the number of batches to store. Defaults to 1000.
    """

    sql_dataloader = SQLDataLoader(db_path, pc_size, batch_size, piece_id, data_type, storage_factor)
    n_batches = len(sql_dataloader)
    print('Number of batches: {}'.format(n_batches))
    # check if folder exists - empty it if it does
    batch_folder = 'data/{}/{}/batches/'.format(data_type, piece_id)
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    else:
        for file in os.listdir(batch_folder):
            os.remove(batch_folder + file)

    for i in tqdm.trange(n_batches, colour='cyan'):
        n_files = len(os.listdir(batch_folder))
        while n_files > threshold:
            sleep(1)
            n_files = len(os.listdir(batch_folder))
        batch = sql_dataloader.get_stored_batch()
        # Squuerze the batch.
        batch = np.squeeze(batch)
        np.save(batch_folder + 'batch_{}.npy'.format(i), batch)


if __name__ == '__main__':
    db_path = 'data/vesuvius_pointcloud.db'
    pc_size = 1024
    batch_size = 1
    piece_id = 1
    data_type = 'train'
    storage_factor = 3200
    threshold = 1000
    provide_data_loop(db_path, pc_size, batch_size, piece_id, data_type, storage_factor, threshold)