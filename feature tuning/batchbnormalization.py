# Gradient Vanishing : 기울기 손실로 인한 문제
# Gradient Exploding : 기울기가 너무 커서 발생하는 문제


# solution
# Change activation function : use ReLU
# Careful initialization Xavier, He
# small learning rate : Gradient Exploding 

# 간접적인 방법 외 직접적인 방법은 다음과 같다.
# Batch Normalization !

# layer 간 Covariate Shifting이 존재한다. 
# 이런 문제를 Internal Covariate Shift problem이라고 한다. 
# scale and shift 
# minibatch마다 사용한다는 의미로 Batch Normalization이라고 한다. 
# shift, scale 역시 gradient를 backward를 통해서 구하고 지속적으로 학습을 한다. 

import torch.nn as nn 
import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 


linear1 = nn.Linear(784, 32, bias = True)
linear2 = nn.Linear(32, 32, bias = True)
linear3 = nn.Linear(32, 10, bias = True)
relu = nn.ReLU()

bn1 = nn.BatchNorm1d(32)
bn2 = nn.BatchNorm1d(32)

nn_linear1 = nn.Linear(784, 32, bias = True)
nn_linear2 = nn.Linear(32, 32, bias = True)
nn_linear3 = nn.Linear(32, 10, bias = True)

bn_model = torch.nn.Sequential(linear1, bn1, relu, linear2, bn2, relu, linear3)
nn_model = torch.nn.Sequential(nn_linear1, relu, nn_linear2, relu, nn_linear3)


mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = False)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = False)

data_loader = DataLoader(mnist_train, batch_size = 100, shuffle = True, drop_last = True)

lr = 1e-3
epochs = 8



# cost function
criterion = nn.CrossEntropyLoss()

# optimizer 
bn_optimizer = optim.Adam(bn_model.parameters(), lr = lr)
nn_optimizer = optim.Adam(nn_model.parameters(), lr = lr)


for epoch in range(1, epochs + 1):
    # train set 선언
    bn_model.train()
    nn_model.train()

    bn_avg_cost = 0 
    nn_avg_cost = 0 

    total_batchsize = len(data_loader)

    for x, y in data_loader:
        x = x.view(-1, 28*28).float()
        bn_optimizer.zero_grad()
        nn_optimizer.zero_grad()


        # fitting 
        y_bn_pred = bn_model(x)
        y_nn_pred = nn_model(x)

        # cost function
        bn_cost = criterion(y_bn_pred, y)
        nn_cost = criterion(y_nn_pred, y)

        # optimizer 
        bn_cost.backward()
        nn_cost.backward()

        bn_optimizer.step()
        nn_optimizer.step()

        bn_avg_cost += bn_cost
        nn_avg_cost += nn_cost
        
    bn_avg_cost /= total_batchsize
    nn_avg_cost /= total_batchsize

    print(f'epoch : {epoch} / {epochs}, bn_cost = {bn_avg_cost:.6f}, nn_cost : {nn_avg_cost:6f} ')



