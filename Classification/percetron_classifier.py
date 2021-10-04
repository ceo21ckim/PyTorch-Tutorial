import torch 
import torch.nn 
import torch.optim as optim
import torch.nn.functional as F 
from  torch.utils.data import DataLoader, Dataset
import pandas as pd

# perceptron 
# XOR 문제는 preceptron으로 해결할 수 없다. 
x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to('cpu')
y = torch.FloatTensor([[0], [1], [1], [0]]).to('cpu')

linear = torch.nn.Linear(2, 1, bias = True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid).to('cpu')

criterion = torch.nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 1)

epochs = 10000
for epoch in range(1, epochs+1):

    optimizer.zero_grad()
    hypothesis = model(x)
    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print(f'epoch : {epoch} / {epochs}, cost : {cost:.6f}')

hypothesis

# multi perceptron 
x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to('cpu')
y = torch.FloatTensor([[0], [1], [1], [0]]).to('cpu')

w1 = torch.Tensor(2, 2)
b1 = torch.Tensor(2)
w2 = torch.Tensor(2, 1)
b2 = torch.Tensor(1)


def simoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

lr = 1
epochs = 10000
for epoch in range(1, epochs +1 ):
    #forward
    l1 = torch.add(torch.matmul(x, w1), b1)
    a1 = sigmoid(l1)
    l2 = torch.add(torch.matmul(a1, w2), b2)
    y_pred = sigmoid(l2)

    cost = F.binary_cross_entropy(y_pred, y)

    d_y_pred = (y_pred - y ) / (y_pred * (1.0 - y_pred) + 1e-7)
    d_l2 = d_y_pred * sigmoid_prime(l2)
    d_b2 = d_l2
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)

    d_a1 = torch.matmul(d_b2, torch.transpose(w2, 0, 1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1 
    d_w1 = torch.matmul(torch.transpose(x, 0, 1), d_b1)


    w1 = w1 - lr * d_w1
    b1 = b1 - lr * torch.mean(d_b1, 0)
    w2 = w2 - lr * d_w2
    b2 = b2 - lr * torch.mean(d_b2, 0)

    if epoch % 100 == 0:
        print(epoch, cost.item())



x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

linear1 = torch.nn.Linear(2, 2, bias = True)
linear2 = torch.nn.Linear(2, 1, bias = True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid)


optimizer = optim.SGD(model.parameters(), lr = 1)
epochs = 10000
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    hypothesis = model(x)

    cost = F.binary_cross_entropy(hypothesis, y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print(f'epoch : {epoch} / {epochs}, cost {cost:.6f}')


hypothesis


x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

linear1 = torch.nn.Linear(2, 10, bias = True)
linear2 = torch.nn.Linear(10, 10, bias = True)
linear3 = torch.nn.Linear(10, 10, bias = True)
linear4 = torch.nn.Linear(10, 1, bias = True)

sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid)

epochs = 10000


optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(1, epochs+1):

    y_hat = model(x)

    cost = F.binary_cross_entropy(y_hat, y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print(f'epoch : {epoch} / {epochs}, cost : {cost:.6f}')


