import cv2
import numpy as np

label_path = 'data/train/{idx}/inklabels.png'

for idx in range(1,4):
    path = label_path.format(idx=idx)
    print(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.squeeze()
    print(img.shape)
    img = np.where(img > 0, 1, 0)

    np.save(path.replace('inklabels.png', 'inklabels.npy'), img)