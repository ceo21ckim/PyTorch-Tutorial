
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

parser = argparse.ArgumentParser(description="")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--epoch", type=int, default=100)

args = parser.parse_args()

BASE_DIR = os.path.dirname(os.getcwd())
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'train/mnist_parameter.pt')

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

def accuracy(pred_y, true_y):
    pred_y = pred_y.argmax(dim = 1)
    acc = (pred_y == true_y).sum()
    return acc 


def test(model, dataloader, criterion, device):
    test_loss = []
    batch_loss = 0
    test_acc = 0
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for id, (image, label) in tqdm(enumerate(dataloader), desc = 'testing...'):
            image = image.to(device)
            label = label.long().to(device)
            pred_y = model(image).to(device)
            loss = criterion(pred_y, label)
            batch_loss += loss.item()
            batch_acc = accuracy(pred_y, label)
            test_acc += batch_acc.item()
            if (id % 10) == 0:
                print(f'batch: [{id}/{len(dataloader)}], batch_loss: {batch_loss/(id+1):.4f}, batch_acc: {(batch_acc / args.batch_size)*100:.2f}%')
    end_time = time.time()
    spend_min, spend_sec = spend_time(start_time, end_time)

    batch_loss /= len(dataloader) * args.batch_size
    test_loss.append(batch_loss)
    test_acc /= len(dataloader) * args.batch_size
    print(f'\t test_loss: {np.mean(test_loss):.4f} time: {spend_min:02}m {spend_sec:.02f}s accuracy: {test_acc*100:.2f}%')
    return test_loss

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

test_data = MNIST(BASE_DIR, train=False, transform = transform, download=True)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle=True, drop_last = True)

model = basic_CNN().to(args.device)
criterion = nn.CrossEntropyLoss().to(args.device)

if __name__ == '__main__':
    model.load_state_dict(torch.load(MODEL_PATH))
    test_loss = test(model, test_loader, criterion, args.device)