import torch
from src.model import PointNet
from src.data_loader import PointCloudData
from tqdm import trange
import tqdm
import numpy as np
import os
from src.transforms import Normalize, RandomNoise, RandRotation_z
from torchvision import transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def train(model, train_loader, val_loader=None,  epochs=15, save=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
    for epoch in trange(epochs, colour='red'):
        model.train()
        running_loss = 0.0
        for i in trange(len(train_loader), colour='green'):
            data = next(iter(train_loader))
            inputs, labels, original_xy = data
            inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = model.pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # update the predicted image
            model.update_predicted_image(outputs, original_xy)

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    accuracy = model.accuracy(outputs, labels)
                    f1 = model.f1_score(outputs, labels)

                    tqdm.tqdm.write('[Epoch: %d, Batch: %4d / %4d]' % (epoch + 1, i + 1, len(train_loader)))
                    tqdm.tqdm.write('loss: %.3f' % (running_loss / 10))
                    tqdm.tqdm.write('Accuracy: %.3f' % (accuracy))
                    tqdm.tqdm.write('F1: %.3f \n' % (f1))

                    running_loss = 0.0

                    # Save the predicted image
                    if not os.path.exists('images/'):
                        os.makedirs('images/')
                    model.save_predicted_image('images/epoch_'+str(epoch)+'_batch_'+str(i)+'.png')

        model.eval()
        model.reset_predicted_image()
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

        if epoch % 10 == 9:
          # save the model
          if save:
              if not os.path.exists('models/'):
                  os.makedirs('models/')
              torch.save(model.state_dict(), "models/save_"+str(epoch)+".pth")




def main():
    """Main trainings loop."""
    
    n_batches = 8412
    n_points = 1024
    batch_size = 256
    n_skip_rm = 100

    pointnet = PointNet(classes=2, features=3, n_points=n_points)
    # load model weights if they exist
    if os.path.exists('models'):
        files = os.listdir('models')
        if len(files) > 0:
            files.sort()
            pointnet.load_state_dict(torch.load('models/'+files[-1]))
            pointnet.eval()

    pointnet.to(device)

    train_transforms = transforms.Compose([
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ])


    train_set = PointCloudData('data/train/1/batches/', len_dataset=n_batches, valid=False, n_points=n_points, n_skip_rm=n_skip_rm, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    # val_loader = PointCloudData('data/test/1/batches/', batch_size=32)
    train(pointnet, train_loader, epochs=111, save=True)


if __name__ == '__main__':
    main()