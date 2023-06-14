import torch
# from src.model import PointNet
from src.data_loader import PointCloudDataV2
from tqdm import trange, tqdm

import numpy as np
import os
import sys
from src.transforms import Normalize, RandomNoise, RandRotation_z, ToTensor
from src.rle import rle
# from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
# from torch.utils.data import DataLoader
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import BCEWithLogitsLoss
from torch_sparse import SparseTensor

IMG_SIZE = (8181, 6330)
PIECE_ID = 'a'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)



def test(model, test_loader, img_path, batch_size):
    writer = SummaryWriter()
    prediction = np.zeros(IMG_SIZE)

    model.eval()
    i = 0
    for data in tqdm(test_loader, colour='green'):
        i += 1
        res = model(data)

        xy = data.y
        
        # For each batch?
        for j in range(batch_size):
            sparse_tensor = SparseTensor(row=xy[i,:, 0], col=xy[i,:, 1], value=res[i], sparse_sizes=IMG_SIZE)
            # save as image
            img = sparse_tensor.to_dense()
            img = img.cpu().detach().numpy()
            img = img.reshape(IMG_SIZE)
            # i_img = i * batch_size + j
            # np.save(img_path + 'img_{}.npy'.format(i_img), img)
            prediction += img
            

        if i % 10 == 0:
                #  add an image to tensorboard
                steps = i * batch_size
                img = prediction / steps
                img = img * 255
                writer.add_image('image', img, i)


    # Create final prediction image
    # img_files = os.listdir(img_path)
    # img_files = [f for f in img_files if f.endswith('.npy')]
    # for img_file in img_files:
    #     img = np.load(img_path + img_file)
    #     prediction += img
    total_steps = i * batch_size
    prediction = prediction / total_steps
    np.save(img_path + 'prediction.npy', prediction)

    # Create final prediction csv
    starts_ix, lengths = rle(prediction)
    inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
    print(f"Id,Predicted\n{PIECE_ID}," + inklabels_rle, file=open(img_path + 'inklabels_rle.csv', 'w'))



def main():
    """Main trainings loop."""
    
    n_points = 640_000

    batch_dir = 'data/pointclouds/test/{}/'.format(PIECE_ID)
    img_path = 'data/pointclouds/test/{}/pred_img/'.format(PIECE_ID)
    batch_size = 3
    epochs = 100

    num_classes = 1
    input_nc = 3

    pointnet = PointNet2(architecture="unet", input_nc=input_nc, num_layers=3, output_nc=num_classes)

    # Load previous checkpoint
    model_files = os.listdir('models/')
    model_files = [x for x in model_files if 'save_p2_' in x]

    if len(model_files) > 0:
        model_files.sort()
        model_files_idx = [int(x.split('_')[-1].split('.')[0]) for x in model_files]
        max_idx = np.argmax(model_files_idx)
        latest_model = model_files[max_idx]
        print("Loading model params from: ", latest_model)
        pointnet.load_state_dict(torch.load('models/'+latest_model))

    pointnet.to(device)

    loader_kwargs = {}
    loader_kwargs = {'num_workers': 1, 'pin_memory': True}

    data_set = PointCloudDataV2(batch_dir, n_points, do_transform=True, transform=[ToTensor()], is_unify=True, is_test=True)
    print("Lenght dataset: ", len(data_set))
    test_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, **loader_kwargs)

    test(pointnet, test_loader, img_path, batch_size)


if __name__ == '__main__':
    main()