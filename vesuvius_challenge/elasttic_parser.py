from elasticsearch import Elasticsearch

import numpy as np
import os
from tqdm import tqdm, trange
from time import sleep,time 
from multiprocessing import Process, Queue, Pool


def elastic_mp(files):
    data_type = 'train'
    piece_id = 1
    n_files = len(files)
    # monitor a folder for new files, load the numpy arrays, and send them to elastic search
    npz_path = './data/{}/{}/npy_points/'.format(data_type, piece_id)
    # connect to elastic search
    auth = ('elastic', 'changeme')
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}], http_auth=auth)


    index_name = 'testtest_ink_{}_{}'.format(data_type, piece_id)
    cols = [*'xyzrgbl']

    def send_to_es(xyzrgbl):
        doc = dict(zip(cols, xyzrgbl))
        es.index(index=index_name, body=doc)


    for n in trange(n_files):
        t_start = time()
        # files = os.listdir(npz_path)
        while len(files) == 0:
            sleep(1)
            files = os.listdir(npz_path)

        file = files.pop()
        xyzrgbl = np.load(npz_path + file)
        xyzrgbl = xyzrgbl.astype(np.int32)
        for i in trange(xyzrgbl.shape[0]):
            send_to_es(xyzrgbl[i, :].tolist())
        
        tqdm.write('Sent %s to elastic search, took %d seconds' % (file, time() - t_start))



def main():

    data_type = 'train'
    piece_id = 1
    # monitor a folder for new files, load the numpy arrays, and send them to elastic search
    npz_path = './data/{}/{}/npy_points/'.format(data_type, piece_id)   
    all_files = os.listdir(npz_path)
    n_workers = 12

    # split the files into n_workers chunks
    files = []
    for i in range(n_workers):
        files.append(all_files[i::n_workers])

    # Pool
    pool = Pool(processes=n_workers)
    pool.map(elastic_mp, files)
    while True:
        sleep(1)
        print('Waiting for pool to finish')
        if pool._taskqueue.empty():
            break

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()