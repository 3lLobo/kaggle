import torch
# from src.model import PointNet
from src.data_loader import PointCloudDataV2
from tqdm import trange, tqdm

import numpy as np
import os
import sys
from src.transforms import Normalize, RandomNoise, RandRotation_z, ToTensor
# from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
# from torch.utils.data import DataLoader
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from torch.nn import BCEWithLogitsLoss

BATCH_SIZE = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def get_metrics(data, res):
    """
    Accuracy and F1 score.
    """
    target = data.y.cpu().detach().numpy().reshape(-1)
    predicted = res.x.cpu().detach().numpy().reshape(-1)
    
    # Logits to probabilities
    predicted = 1 / (1 + np.exp(-predicted))
    predicted = np.round(predicted)

    accuracy = accuracy_score(target, predicted)
    # f1 = f1_score(target, predicted, average='macro')
    f1 = fbeta_score(target, predicted, beta=0.5, average='macro')

    return accuracy, f1


def train(model, train_loader, val_loader,  epochs=15, save=True, start_epoch=0):
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEWithLogitsLoss()

    for epoch in trange(start_epoch, epochs, colour='red'):
        model.train()
        running_loss = 0.0
        i = 0
        for data in tqdm(train_loader, colour='green'):
            i += 1

            optimizer.zero_grad()
            res = model(data)
            
            
            loss = criterion(res.x, data.y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 10 == 0:
                    accuracy, f1 = get_metrics(data, res)
                    writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
                    writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_loader) + i)
                    writer.add_scalar('F1/train', f1, epoch * len(train_loader) + i)
                    running_loss = 0.0


        model.eval()
        accs = f1s = total = losses = 0

        # validation
        with torch.no_grad():
            for data in tqdm(val_loader, colour='blue'):
                res = model(data)
                loss = criterion(res.x, data.y)

                accuracy, f1 = get_metrics(data, res)
                accs += accuracy
                f1s += f1
                losses += loss.item()
                total += 1

        writer.add_scalar('Accuracy/val', 100. * accs / total, epoch)
        writer.add_scalar('F1/val', 100. * f1s / total, epoch)
        writer.add_scalar('Loss/val', losses / total, epoch)
        

        # if epoch % 10 == 9:
          # save the model
        if save:
            if not os.path.exists('models/'):
                os.makedirs('models/')
            torch.save(model.state_dict(), "models/save_p2_"+str(epoch)+".pth")




def main():
    """Main trainings loop."""

    # random seed
    torch.manual_seed(11)
    
    n_points = 640_000

    batch_dir = 'data/pointclouds/train/'
    batch_size = 3
    epochs = 100
    start_epoch = 0

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
        start_epoch = model_files_idx[max_idx] + 1

    pointnet.to(device)

    loader_kwargs = {}
    loader_kwargs = {'num_workers': 1, 'pin_memory': True}

    data_set = PointCloudDataV2(batch_dir, n_points, do_transform=True, transform=[ToTensor()], is_unify=True)
    print("Lenght dataset: ", len(data_set))
    train_set, val_set = torch.utils.data.random_split(data_set, [int(0.8 * len(data_set)), len(data_set) - int(0.8 * len(data_set))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, **loader_kwargs)

    train(pointnet, train_loader, epochs=epochs, save=True, val_loader=val_loader, start_epoch=start_epoch)


if __name__ == '__main__':
    main()