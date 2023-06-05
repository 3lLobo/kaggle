import torch
from src.model import PointNet
from src.data_loader import PointCloudData
from tqdm import trange
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def train(model, train_loader, val_loader=None,  epochs=15, save=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in trange(epochs, colour='red'):
        model.train()
        running_loss = 0.0
        for i in trange(len(train_loader), colour='green'):
            data = next(iter(train_loader))
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = model.pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
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

        if epoch % 10 == 9:
          # save the model
          if save:
              if not os.path.exists('models/'):
                  os.makedirs('models/')
              torch.save(model.state_dict(), "models/save_"+str(epoch)+".pth")



def main():
    """Main trainings loop."""
    
    pointnet = PointNet(classes=2, features=3)
    pointnet.to(device)
    train_set = PointCloudData('data/train/1/batches/', len_dataset=841345227, valid=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    # val_loader = PointCloudData('data/test/1/batches/', batch_size=32)
    train(pointnet, train_set, epochs=111, save=True)


if __name__ == '__main__':
    main()