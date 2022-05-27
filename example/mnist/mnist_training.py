
import os
import time 
import numpy as np 
from tqdm import tqdm 
import argparse

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 

from torchvision import datasets 
from torchvision import transforms 
from torchvision.datasets import MNIST 

from metric import *
print('pytorch version:',torch.__version__)
BASE_DIR = os.path.dirname(__file__)
TRAIN_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'train')

parser = argparse.ArgumentParser(description="")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--epoch", type=int, default=10)

args = parser.parse_args()

# model 
class basic_CNN(nn.Module):
    def __init__(self):
        super(basic_CNN, self).__init__()

        def nn_block(in_channels, out_channels, kernel_size):
            layers = []
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            return layers

        self.model = nn.Sequential(*nn_block(1, 32, 3), *nn_block(32, 64, 3))
        self.fc_layer1 = nn.Linear(1600 , 64)
        self.fc_layer2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(args.batch_size,-1)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.softmax(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

train_data = MNIST(BASE_DIR, train=True, transform = transform, download=True)

train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, drop_last = True)

model = basic_CNN().to(args.device)
criterion = nn.CrossEntropyLoss().to(args.device)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

def train(model, dataloader, epochs, optimizer,  criterion, device):
    train_epoch_loss = []
    model.train()
    best_loss = float('inf')
    for epoch in tqdm(range(epochs), desc = 'traing....'):
        start_time = time.time()
        batch_loss = 0
        for image, label in dataloader:
            image = image.to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            pred_y = model(image).to(device)
            loss = criterion(pred_y, label)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        batch_loss /= (len(train_loader) * args.batch_size)
        train_epoch_loss.append(batch_loss)
        end_time = time.time()

        spend_min, spend_sec = spend_time(start_time, end_time)

        if (epoch+1) % 1 == 0:
            print(f'epoch:[{epoch+1}/{epochs}] \t train_epoch_loss: {np.mean(train_epoch_loss):.4f} \t time: {spend_min:03}m {spend_sec:.02f}s')
        
        if not os.path.exists(TRAIN_SAVE_DIR):
            os.makedirs(TRAIN_SAVE_DIR)
        
        if best_loss > batch_loss:
            torch.save(model.state_dict(), os.path.join(TRAIN_SAVE_DIR, 'mnist_parameter.pt'))
            best_loss = batch_loss

    return train_epoch_loss


if __name__ == '__main__':
    train_loss = train(model, train_loader, args.epoch, optimizer, criterion, args.device)
