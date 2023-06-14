import torch
import numpy as np
import time

N_XP = 1000
IMG_SHAPE = (100000, 10000)
N_IDCES = 6000

img = np.arange(0, IMG_SHAPE[0] * IMG_SHAPE[1]).reshape(IMG_SHAPE)

idces = np.random.randint(0, IMG_SHAPE[0], (N_IDCES, 2))

start = time.time()
for i in range(N_XP):
    labels = img[idces[:, 0], idces[:, 1]]
print("Numpy time: ", time.time() - start)

# puch to gpu
device = torch.device('cuda:0')
start = time.time()
img = torch.from_numpy(img).to(device)
idces = torch.from_numpy(idces).to(device)

print("GPU time: ", time.time() - start)

start = time.time()

for i in range(N_XP):
    labels = img[idces[:, 0], idces[:, 1]]
print("Torch time: ", time.time() - start)
    