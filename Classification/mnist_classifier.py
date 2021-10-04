import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import pandas as pd 
import os, sys
import torchvision
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# torch vision 
# torch.vision.models : AlexNet, VGG, ResNet, DenseNet, torch transforms
# torchvision.datasets. : MNIST, EMNIST

# 28 by 28 / 1 chaanel 
# transform은 data를 불러올때 어떤 상태로 불러올 것인가에 대한 코드.
# pytorch는 tensor로 받기 때문에 totensor
mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = True)
mnist_test
mnist_train


batch_size = 64
data_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)

epochs = 15

linear = torch.nn.Linear(784, 10, bias = True).to('cpu')
criterion = torch.nn.CrossEntropyLoss().to('cpu')
optimizer = optim.SGD(linear.parameters(), lr = 0.1)

for epoch in range(epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for x, y in data_loader:
        # reshape in put image into [batch_size by 784] ; shape!
        # label is not ont-hot encoded

        x = x.view(-1, 28*28).to('cpu') # 784 by 10 ; because mnist label is (0~9)  
        
        optimizer.zero_grad()
        hypothesis = linear(x)
        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch

        print(f'epoch : {epoch+1} / {epochs} cost : {avg_cost:.6f}')


# Test the model using test sets
with torch.no_grad(): # 이 범위 내에서는 gradient를 계산하지 않을 것이다
    x_test = mnist_test.test_data.view(-1, 28*28).float().to('cpu')
    y_test = mnist_test.targets.to('cpu')

    prediction = linear(x_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test 
    accuracy = correct_prediction.float().mean()
    print(f'accuracy: {accuracy.item()}')



# visualization
mnist_test.test_data[0]
r = random.randint(0, len(mnist_test) - 1)
# 라벨 중 하나의 이미지만 가져온다!
x_single_data = mnist_test.test_data[r:r + 1].view(-1, 28*28).float().to('cpu')
y_single_data = mnist_test.targets[r:r + 1].to('cpu')

print('Label : ', y_single_data.item())
single_prediction = linear(x_single_data)
print('Prediction :', torch.argmax(single_prediction, 1).item())

plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap = 'Greys', interpolation='nearest')
plt.show()

