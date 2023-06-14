import numpy as np
# from PIL import Image

# Fast run length encoding, from https://www.kaggle.com/code/hackerpoet/even-faster-run-length-encoder/script
def rle (img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths

# inklabels = np.array(Image.open('inklabels.png'), dtype=np.uint8)
# starts_ix, lengths = rle(inklabels)
# inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
# print("Id,Predicted\n1," + inklabels_rle, file=open('inklabels_rle.csv', 'w'))