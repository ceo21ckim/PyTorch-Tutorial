import torch 
import torch.nn as nn 

import torch.optim as optim
from torch.utils.data import DataLoader 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms

mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = False)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = False)

# DataLoader(dataset, batch_size, shuffle, drop_last)
data_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True, drop_last = True)

linear1 = nn.Linear(784, 512, bias = True)
linear2 = nn.Linear(512, 512, bias = True)
linear3 = nn.Linear(512, 512, bias = True)
linear4 = nn.Linear(512, 512, bias = True)
linear5 = nn.Linear(512, 10, bias = True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p = 0.2)

model = torch.nn.Sequential(linear1, relu, dropout, linear2, relu, dropout, linear3, relu, dropout, linear4, relu, dropout, linear5).to('cpu')


# cost function
criterion = nn.CrossEntropyLoss()

# optimizer 
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 3


# training
model.train()
for epoch in range(1, epochs + 1):
    avg_cost = 0
    total_batchsize = len(data_loader)

    for x, y in data_loader:
        optimizer.zero_grad()
        x = x.view(-1, 28*28).float() # resize 
        # fitting 
        y_hat = model(x)

        cost = criterion(y_hat, y)

        cost.backward()
        optimizer.step()

        avg_cost += cost
    
    avg_cost /= total_batchsize
    print(f'epoch : {epoch} / {epochs}, cost : {avg_cost:.6f}')
        

with torch.no_grad():
    model.eval()

    x_test = mnist_test.test_data.view(-1, 28*28).float()
    y_test = mnist_test.targets 

    pred = model(x_test)

    correct_pred = torch.argmax(pred, 1) == y_test 

    acc = correct_pred.sum() / len(y_test) * 100

    print(f'accuracy : {acc:.2f}%')