import torch
# from src.model import PointNet
from src.data_loader import PointCloudData
from tqdm import trange
import tqdm
import numpy as np
import os
import sys
from src.transforms import Normalize, RandomNoise, RandRotation_z
# from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data
from sklearn.metrics import accuracy_score, f1_score
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
    f1 = f1_score(target, predicted, average='macro')

    return accuracy, f1


def train(model, train_loader, val_loader=None,  epochs=15, save=True):
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEWithLogitsLoss()

    for epoch in trange(epochs, colour='red'):
        model.train()
        running_loss = 0.0
        for i in trange(len(train_loader), colour='green'):
            data =  train_loader.get_batch(BATCH_SIZE)

            optimizer.zero_grad()
            res = model(data)
            
            loss = criterion(res.x, data.y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 10 == 9:    # print every 10 mini-batches
                    accuracy, f1 = get_metrics(data, res)
                    writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
                    writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_loader) + i)
                    writer.add_scalar('F1/train', f1, epoch * len(train_loader) + i)
                    running_loss = 0.0


        model.eval()
        accs = f1s = total = losses = 0

        # validation
        if val_loader:
            train_loader.set_test_mode(True)
            with torch.no_grad():
                for i in trange(len(train_loader), colour='yellow'):
                    data =  train_loader.get_batch(BATCH_SIZE)
                    res = model(data)
                    loss = criterion(res.x, data.y)

                    accuracy, f1 = get_metrics(data, res)
                    accs += accuracy
                    f1s += f1
                    losses += loss.item()
                    total += data.y.size(-1)
            writer.add_scalar('Accuracy/val', 100. * accs / total, epoch)
            writer.add_scalar('F1/val', 100. * f1s / total, epoch)
            writer.add_scalar('Loss/val', losses / total, epoch)
            
            train_loader.set_test_mode(False)

        # if epoch % 10 == 9:
          # save the model
        if save:
            if not os.path.exists('models/'):
                os.makedirs('models/')
            torch.save(model.state_dict(), "models/save_p2_"+str(epoch)+".pth")




def main():
    """Main trainings loop."""

    rec_default = sys.getrecursionlimit()
    print('Default recursion limit: ', rec_default)

    
    n_batches = 969
    pc_factor = 1
    n_points = 1024 * pc_factor

    sys.setrecursionlimit(rec_default * 100)

    batch_size = 311
    n_skip_rm = 10000000000000000

    epochs = 100

    num_classes = 1
    input_nc = 3

    pointnet = PointNet2(architecture="unet", input_nc=input_nc, num_layers=3, output_nc=num_classes)

    pointnet.to(device)


    train_set = PointCloudData('data/train/1/batches/', len_dataset=n_batches, valid=True, n_points=n_points, n_skip_rm=n_skip_rm)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    # val_loader = PointCloudData('data/test/1/batches/', batch_size=32)
    train(pointnet, train_set, epochs=epochs, save=True, val_loader=True)


if __name__ == '__main__':
    main()