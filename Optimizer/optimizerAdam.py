import torch.optim as optim
import torch 
import torch.nn as nn 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# optim.SGD
# optim.Adadelta
# optim.Adagrad
# optim.Adam
# optim.SparseAdam
# optim.Adamax
# optim.ASGD
# optim.LBFGS
# optim.RMSprop
# optim.Rprop

mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = False)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = False)

data_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True, drop_last = True)
epochs = 2
lr = 1e-3
# initailization 
# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias = True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(linear.parameters(), lr = 1e-3)
nn.init.normal_(linear.weight)

for epoch in range(1, epochs + 1):
    avg_cost = 0
    total_batch = len(data_loader)
    for x, y in data_loader:

        x = x.view(-1, 28*28)

        optimizer.zero_grad()
        y_hat = linear(x)
        # define cost / lost & optimizer 
        cost = criterion(y_hat, y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch

    print(f'epoch : {epoch} / {epochs}, cost = {avg_cost:.6f}')


# mnist_nn
lr = 1e-3
epochs = 15
batch_size = 100 

linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)

relu = torch.nn.ReLU()

nn.init.normal_(linear1.weight)
nn.init.normal_(linear2.weight)
nn.init.normal_(linear3.weight)

model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)


for epoch in range(1, epochs + 1):
    avg_cost = 0 
    total_batch = len(data_loader)
    
    for x, y in data_loader:
        
        x = x.view(-1, 28*28)
        y_hat = model(x)

        cost = criterion(y_hat, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batch_size 

    print(f'epoch : {epoch} / {epochs}, cost : {avg_cost:.6f}')


with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28*28).float()
    y_test = mnist_test.targets 

    y_hat = model(x_test)

    correct_pred = torch.argmax(y_hat, 1) == y_test

    accuracy = correct_pred.sum() / len(y_test) * 100
    true_acc = correct_pred.float().mean() * 100 # boolean value를 float으로 바꾸어준다 0, 1 로
    print(f'my code :{accuracy:.6f}')
    print(f'my code :{true_acc.item():.6f}')


