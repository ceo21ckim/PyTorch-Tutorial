import torch 
import torch.nn as nn 
import torch.nn.init as init 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# initailization 할 떄 주의해야할 점
# Not all 0's 
# Challenging issue
# Hinton et al. (2006) : Restricted Boltzmann Machine (RBM) !
# Restricted : no connections within a layer 
# 같은 layer 안에서는 연결이 없고, 다른 layer와는 fully connectied 
# input x가 들어왔을 때 y를 만든다. 

# 다음 방식이 제안되어서 RBM은 잘 사용되지 않는다. 
# Xavier(2010) / He initialization(2015)


# Xavier Normal distribution 
# w ~ n (0, var(w))
# Xavier Uniform distribution
# w ~ u (- sqrt(6/(input, output)), + sqrt(6/(input, output)))

# He initialization : Xavier의 변형
# n(out) term이 없어진 모델.

linear1 = nn.Linear(784, 256, bias = True)
linear2 = nn.Linear(256, 256, bias = True)
linear3 = nn.Linear(256, 10, bias = True)
relu = torch.nn.ReLU()

init.xavier_uniform_(linear1.weight)
init.xavier_uniform_(linear2.weight)
init.xavier_uniform_(linear3.weight)

# initialization 을 제대로 하면 성능이 좋아진다. 

linear1 = nn.Linear(784, 512, bias = True)
linear2 = nn.Linear(512, 512, bias = True)
linear3 = nn.Linear(512, 512, bias = True)
linear4 = nn.Linear(512, 512, bias = True)
linear5 = nn.Linear(512, 10, bias = True)


init.xavier_uniform_(linear1.weight)
init.xavier_uniform_(linear2.weight)
init.xavier_uniform_(linear3.weight)
init.xavier_uniform_(linear4.weight)
init.xavier_uniform_(linear5.weight)


mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = False)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = False)

data_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True, drop_last = True)

model = nn.Sequential(linear1, relu, linear2, relu, linear3, relu, linear4, relu, linear5)

# cost function 
criterion = nn.CrossEntropyLoss()

# optimizer 
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 15
for epoch in range(1, epochs + 1):
    avg_cost = 0
    total_batchsize = len(data_loader)

    for x, y in data_loader:
        x = x.view(-1, 28*28).float()

        y_hat = model(x)

        # cost function
        cost = criterion(y_hat, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batchsize

    print(f'epoch : {epoch} / {epochs}, cost : {avg_cost:.6f}')


x_test = mnist_test.test_data.view(-1, 28*28).float()
y_test = mnist_test.targets

with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28*28).float()
    y_test = mnist_test.targets

    y_pred = model(x_test)

    correct_pred = torch.argmax(y_pred, 1) == y_test

    acc = correct_pred.sum() / len(y_test) * 100

    print(f'acc : {acc:.4f}%')
    # acc : 97%