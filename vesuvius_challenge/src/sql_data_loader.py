# Load a point-cloud batch from the database.

import sqlite3
import numpy as np
import pandas as pd
import tqdm
import threading
from time import time, sleep


class SQLDataLoader:
  def __init__(self, db_path: str, pc_size:int, batch_size: int, piece_id: int, data_type: str, storage_factor: int = 10) -> None:
    """Initialize the class.

    Args:
        db_path (str): Path to the database.
        batch_size (int): Batch size.
        piece_id (int): Id of the piece.
        data_type (str): Train or test.
    """

    self.db_path = db_path
    self.pc_size = pc_size
    self.batch_size = batch_size
    self.piece_id = piece_id
    self.data_type = data_type

    # Connect to the database.
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    self.con = con
    self.cur = cur
    # assert batch-size times pc size are even
    assert batch_size * pc_size % 2 == 0

    table_name = 'pc_{}_{}'.format(data_type, piece_id)
    self.table_name = table_name

    # Get the number of point-clouds in the database.
    cur.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    self.n_pc = cur.fetchone()[0]

    self.select_columns = 'x, y, z, r, g, b, label'
    self.execute_select_random_ink =  lambda x: 'SELECT {} FROM {} WHERE label = 1 ORDER BY RANDOM() LIMIT {}'.format(self.select_columns, table_name, x//2)
    self.execute_select_random_no_ink = lambda x: 'SELECT {} FROM {} WHERE label = 0 ORDER BY RANDOM() LIMIT {}'.format(self.select_columns, table_name, x//2)

    # Parallel loading of batches.
    self.storage_factor = storage_factor

    t = threading.Thread(target=self.store_batch)
    t.start()
    self.t = t
    self.batch_storage = []
    self.t_has_finished = False

    # Get ratio of ink and no-ink labels.
    cur.execute('SELECT COUNT(*) FROM {} WHERE label = 1'.format(table_name))
    n_ink = cur.fetchone()[0]
    self.n_ink = n_ink
    cur.execute('SELECT COUNT(*) FROM {} WHERE label = 0'.format(table_name))
    n_no_ink = cur.fetchone()[0]
    self.ratio = n_ink / n_no_ink
  
  def get_balanced_batch(self, batch_size: int = None, cur: sqlite3.Cursor = None) -> np.array:
    """Return a balanced batch with equal number of ink and no-ink labels.

    Args:
        batch_size (int, optional): Batch size. Defaults to None.
        cur (sqlite3.Cursor, optional): Cursor to the database. Defaults to None.
    Returns:
        np.array: Batch of point-clouds.
    """
    if batch_size is None:
      batch_size = self.batch_size

    if cur is None:
      cur = self.cur

    n_points = batch_size * self.pc_size // 2
    cur.execute(self.execute_select_random_no_ink(n_points))
    pc_batch_0 = np.array(cur.fetchall())
    cur.execute(self.execute_select_random_ink(n_points))
    pc_batch_1 = np.array(cur.fetchall())
    pc_batch = np.vstack((pc_batch_0, pc_batch_1))
    np.random.shuffle(pc_batch)
    # make batch size equal to batch_size.
    print("SHAPE",pc_batch.shape)
    pc_batch = pc_batch.reshape((batch_size, -1, 7))
    print("SHAPE2",pc_batch.shape)
    return pc_batch  

  def store_batch(self):
    """Store a batch in the storage.
    Threaded function need own connection to the database.
    """
    con = sqlite3.connect(self.db_path)
    cur = con.cursor()

    batch = self.get_balanced_batch(batch_size=self.batch_size*self.storage_factor, cur=cur)
    #  slice the batch into smaller batches.
    batch = np.array_split(batch, self.storage_factor)
    self.batch_storage.extend(batch)
    self.t_has_finished = True  
    con.close()
    print('Stored batch in storage.')
  
  def get_stored_batch(self):
    """Pops a batch from the storage if available and opens a new thread to fill the storage if it is lower than 5 batches.
    """
    # Check if thread has finished.
    if self.t_has_finished:
      self.t.join()
      # Check if storage is below 5 batches.
      if len(self.batch_storage) < self.storage_factor // 2:
        # Start new thread.
        t = threading.Thread(target=self.store_batch)
        t.start()
        self.t = t
        self.t_has_finished = False
    # Pop batch from storage.
    if len(self.batch_storage) == 0:
      while len(self.batch_storage) == 0:
        sleep(0.1)
    batch = self.batch_storage.pop(0)
    return batch



  def __iter__(self):
    return self
  
  def __next__(self):
    return self.get_balanced_batch()
  
  def __len__(self):
    return int(self.n_pc // (self.batch_size * self.ratio))
  
  def __del__(self):
    self.con.close()

  







if __name__ == '__main__':
  db_path = './data/vesuvius_pointcloud.db'
  batch_size = 16
  pc_size = 200000
  piece_id = 1
  data_type = 'train'

  dataloader = SQLDataLoader(db_path, batch_size, pc_size, piece_id, data_type)
  print(dataloader.n_pc)
  print(dataloader.ratio)
  print(dataloader.n_ink)

  for i in tqdm.trange(100, colour='cyan'):
    # batch = dataloader.get_balanced_batch()
    t_start = time()
    batch = dataloader.get_stored_batch()
    # tqdm.tqdm.write('{) , {}'.format(batch.shape[1], batch.dtype))
    print(batch.shape)
    print(batch.dtype)
    mean = np.mean(batch[:,:, :3], axis=0)
    mean = np.mean(mean, axis=0)
    tqdm.tqdm.write('{:.3f}, {:.3f}, {:.3f}'.format(mean[0], mean[1], mean[2]))
    tqdm.tqdm.write('-----------------')
    if time() - t_start > 0.1:
      sleep(0.5)