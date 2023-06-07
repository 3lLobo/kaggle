import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torcheval.metrics.functional import binary_f1_score
import cv2

# TODO: k=6 for coordinates plus rgb features
# TODO: matrix3x3 becomes matrix6x6. Check if the model still worqs.
# TODO: Store the 2D predictions.

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input_tensor):
      # input.shape == (bs,n,3)
      bs = input_tensor.size(0)
      xb = F.relu(self.bn1(self.conv1(input_tensor)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self, k=3, n_points=1024):
        super().__init__()
        self.input_transform = Tnet(k=k)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(k,64,1) 

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,n_points,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(n_points)
       
   def forward(self, input_tensor):
        matrix3x3 = self.input_transform(input_tensor)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input_tensor,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 2, features = 3, n_points = 1024):
        super().__init__()
        self.k = features + 3
        self.transform = Transform(k=self.k, n_points=n_points)
        # Use transformer and out put a score for each point.
        self.transformer = nn.TransformerEncoderLayer(d_model=n_points, nhead=16)
        

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(p=0.3)
        # self.logsoftmax = nn.LogSoftmax(dim=1) # Sigmoid ?
        
        # TODO: Loss for multi-point classification
        self.criterion = nn.BCEWithLogitsLoss()

        self.img_size = (7606, 5249)

        self.img_predicted = np.zeros(self.img_size)

    def forward(self, input_tensor):
        xb, matrix3x3, matrix64x64 = self.transform(input_tensor)
        
        xb = xb.unsqueeze(0)
        xb = self.transformer(xb)
        xb = xb.squeeze(0)
        output = xb
        return output, matrix3x3, matrix64x64
    
    def pointnetloss(self, outputs, labels, m3x3, m64x64, alpha = 0.0001):
        bs=outputs.size(0)
        id3x3 = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
        if outputs.is_cuda:
            id3x3=id3x3.cuda()
            id64x64=id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        outputs = outputs.unsqueeze(2)
        bcc_loss = self.criterion(outputs, labels)
        conv_loss = alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
        full_loss = bcc_loss + conv_loss
        return full_loss
        # return self.criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
    
    def sample_prediction(self, outputs):
        outputs = outputs.detach().cpu().numpy()
        outputs = np.where(outputs < 0.5, 0, 1)
        return outputs

    def accuracy(self, outputs, labels):
        outputs = self.sample_prediction(outputs)
        outputs = outputs.reshape(-1)
        labels = labels.cpu().numpy().reshape(-1)
        acc = np.sum(np.equal(outputs, labels)) / len(labels)
        return acc
    
    def f1_score(self, outputs, labels):
        f1 = binary_f1_score(labels.view(-1), outputs.detach().view(-1))
        f1 = f1.cpu().numpy()
        return f1
    
    def update_predicted_image(self, outputs: torch.Tensor, points: np.ndarray):
        """Recreate the image from the predicted points.
        Overlay it to the existing image with a weight of 0.1

        Args:
            outputs (torch.Tensor): the predicted outputs
            points (np.ndarray): the original xy coordinates of the points
        """
        points = points.numpy().astype(np.int8)
        outputs = self.sample_prediction(outputs)
        # Stack along batch dim and remove duplicates
        points = points.reshape(-1, 2)
        points, points_index = np.unique(points, axis=0, return_index=True)
        new_img = np.zeros_like(self.img_predicted)
        outputs = outputs.reshape(-1)
        outputs = outputs[points_index]
        outputs = np.where(outputs == 0, -1, 1)

        new_img[points[:, 1], points[:, 0]] = outputs
        img_predicted = self.img_predicted + new_img
        # min-max normalization
        img_predicted = np.where(img_predicted < -1, -1, img_predicted)
        img_predicted = np.where(img_predicted > 1, 1, img_predicted)
        self.img_predicted = img_predicted

    def save_predicted_image(self, path: str):
        """Save the predicted image to the given path.

        Args:
            path (str): the path to save the image to
        """
        img = self.img_predicted + np.ones_like(self.img_predicted) * 128
        cv2.imwrite(path, self.img_predicted)

    def reset_predicted_image(self):
        """Reset the predicted image to a blank image."""
        self.img_predicted = np.ones(self.img_size, dtype=np.uint8) * 128
