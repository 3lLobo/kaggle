from elasticsearch import Elasticsearch
import numpy as np
import os
from tqdm import tqdm, trange
from time import sleep,time 

data_type = 'train'
piece_id = 1
n_files = 65
# monitor a folder for new files, load the numpy arrays, and send them to elastic search
npz_path = './data/{}/{}/npy_points/'.format(data_type, piece_id)
# connect to elastic search
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# load numpy array

files = os.listdir(npz_path)

index_name = 'ink_{}_{}'.format(data_type, piece_id)
cols = 'xyzrgbl'.split('')

def send_to_es(xyzrgbl):
    doc = dict(zip(cols, xyzrgbl))
    es.index(index=index_name, body=doc)


for n in trange(n_files):
    t_start = time()
    files = os.listdir(npz_path)
    while len(files) == 0:
        sleep(1)
        files = os.listdir(npz_path)

    file = files.pop()
    xyzrgbl = np.load(npz_path + file)
    xyzrgbl = xyzrgbl.astype(np.int32)
    send_to_es(xyzrgbl)
    
    tqdm.write('Sent %s to elastic search, took %d seconds' % (file, time() - t_start))




