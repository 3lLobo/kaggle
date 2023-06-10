import torch
# from src.model import PointNet
from src.data_loader import PointCloudData
from tqdm import trange
import tqdm
import numpy as np
import os
from src.transforms import Normalize, RandomNoise, RandRotation_z
# from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import BCEWithLogitsLoss

BATCH_SIZE = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def train(model, train_loader, val_loader=None,  epochs=15, save=True):
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEWithLogitsLoss()

    # pbar_loss = tqdm.tqdm(total=0, position=0, bar_format='{desc}', colour='red')
    # pbar_acc = tqdm.tqdm(total=1.5, position=1, colour='green', bar_format='{desc}')
    # pbar_f1 = tqdm.tqdm(total=1, position=2, bar_format='{desc}', colour='blue')
    
    for epoch in trange(epochs, colour='red'):
        model.train()
        running_loss = 0.0
        for i in trange(len(train_loader), colour='green'):
            data =  train_loader.get_batch(BATCH_SIZE)
            # inputs, features, labels = data

            # inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            res = model(data)
            print(torch.mean(res.x))
            
            loss = criterion(res.x, data.y)
            loss.backward()

            # loss = model.pointnetloss(outputs, labels, m3x3, m64x64)
            # loss.backward()
            optimizer.step()

            # update the predicted image
            # model.update_predicted_image(outputs, original_xy)

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    target = data.y.cpu().detach().numpy().reshape(-1)
                    predicted = res.x.cpu().detach().numpy().reshape(-1)
                    
                    # Logits to probabilities
                    predicted = 1 / (1 + np.exp(-predicted))
                    predicted = np.round(predicted)

                    accuracy = accuracy_score(target, predicted)
                    f1 = f1_score(target, predicted, average='macro')
                    
                    writer.add_scalar('Loss/train', running_loss / 10, epoch * len(train_loader) + i)
                    writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_loader) + i)
                    writer.add_scalar('F1/train', f1, epoch * len(train_loader) + i)

                    running_loss = 0.0


        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # if epoch % 10 == 9:
          # save the model
        if save:
            if not os.path.exists('models/'):
                os.makedirs('models/')
            torch.save(model.state_dict(), "models/save_p2_"+str(epoch)+".pth")




def main():
    """Main trainings loop."""
    
    n_batches = 969
    n_points = 1024
    batch_size = 311
    n_skip_rm = 10000000000000000

    epochs = 100

    num_points = 1024
    num_classes = 1
    input_nc = 3

    pointnet = PointNet2(architecture="unet", input_nc=input_nc, num_layers=3, output_nc=num_classes)


    # pointnet = PointNet(classes=2, features=3, n_points=n_points)
    # load model weights if they exist
    # if os.path.exists('models'):
    #     files = os.listdir('models')
    #     if len(files) > 0:
    #         # sort files by epoch
    #         files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    #         print('Loading model: ', files[-1])
    #         pointnet.load_state_dict(torch.load('models/'+files[-1]))
    #         pointnet.eval()

    pointnet.to(device)

    # train_transforms = transforms.Compose([
    #                 Normalize(),
    #                 # RandRotation_z(),
    #                 # RandomNoise(),
    #                 ])


    train_set = PointCloudData('data/train/1/batches/', len_dataset=n_batches, valid=False, n_points=n_points, n_skip_rm=n_skip_rm)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    # val_loader = PointCloudData('data/test/1/batches/', batch_size=32)
    train(pointnet, train_set, epochs=epochs, save=True)


if __name__ == '__main__':
    main()