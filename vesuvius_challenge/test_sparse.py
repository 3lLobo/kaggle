import torch
from torch_sparse import SparseTensor
import numpy as np
import os
import cv2
from src.rle import rle
# Fake data

IMG_SIZE = (8181, 6330)
# IMG_SIZE = (100, 100)
n_points = 640_000_00

img_path = 'test_sparse/'
if not os.path.exists(img_path):
    os.makedirs(img_path)

batch_size = 2
xy = np.random.randint(0, IMG_SIZE[1], size=(batch_size, n_points, 2))
res = np.random.rand(batch_size, n_points, 1)

xy = torch.tensor(xy)
res = torch.tensor(res)
img_files = []
for i in range(batch_size):
  sparse_tensor = SparseTensor(row=xy[i,:, 0], col=xy[i,:, 1], value=res[i], sparse_sizes=IMG_SIZE)
  # save as image
  img = sparse_tensor.to_dense()
  img = img.cpu().detach().numpy()
  img = img.reshape(IMG_SIZE)
  i_img = i
  print("Img {} with shape: ".format(i_img), img.shape)
  img_file = img_path + 'img_{}.npy'.format(i_img)
  np.save(img_file, img)
  img_files.append(img_file)


# Create final prediction image
prediction = np.zeros(IMG_SIZE)
for img_file in img_files:
  img = np.load(img_file)
  prediction += img
prediction = prediction / len(img_files)

img = prediction * 255
print("Pred mean: ", img.mean())
print("Pred shape: ", img.shape)
cv2.imwrite(img_path + 'prediction.png', img)

img = cv2.imread(img_path + 'prediction.png')
print("Img shape: ", img.shape)
print("Img mean: ", img.mean())
# cv2.imwrite(img_path + 'prediction.png', img)

# Create final prediction csv
starts_ix, lengths = rle(prediction)
inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
print(f"Id,Predicted\ntest," + inklabels_rle, file=open(img_path + 'inklabels_rle.csv', 'w'))