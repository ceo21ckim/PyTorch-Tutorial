import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)

# Discrete Probability Distribution
# Softmax 

z = torch.FloatTensor([1, 2, 3])
z.shape

# probability
hypothesis = F.softmax(z, dim = 0)
print(hypothesis)

# sum = 1.
hypothesis.sum()

#####################################################################
# Low-level

# class = 5, samples = 3
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim = 1) # 수행되고 난 후 2번째에 있는 값을 가지고 소프트 맥스를 취해라 라는 뜻.
print(hypothesis) 

y = torch.randint(5, (3, )).long()

y_one_hot = torch.zeros_like(hypothesis)

# y_one_hot.scatter_(dim, idx, value) #  1차원에 y인덱스 값을 , 1로 채워줘라 
y_one_hot.scatter_(1, y.unsqueeze(1), 1) #  1차원에 y인덱스 값을 , 1로 채워줘라 

# scatter_ == 똑같은 scatter함수인데 in-place 연산을 진행하는 것이다. 

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
print(cost)


#####################################################################

# Cross-entorpy Loss with torch.nn.functional 

# Low - level 
(y_one_hot * - F.log_softmax(z, dim = 1)).sum(dim = 1).mean()


# High level
# Negative Log Likelihood Loss
F.nll_loss(F.log_softmax(z, dim = 1), y)


F.cross_entropy(z, y)



#####################################################################
# Training with Low-level Cross Entropy Loss

x_train = [[1, 2, 1, 1], 
           [2, 1, 3, 2],
           [3, 1, 3, 4], 
           [4, 1, 5, 5],
           [1, 7, 5, 5], 
           [1, 2, 5, 6],
           [1, 6, 6, 6], 
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# x dim : M x N
# W dim : N x classes(3)
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optim
optimizer = optim.SGD([W, b], lr = 0.1)


epochs = 1000
for epoch in range(epochs+1):

    # cost

    hypothesis = F.softmax(x_train.matmul(W) + b)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot = y_one_hot.scatter(1, y_train.unsqueeze(1), 1)

    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim = 1))).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch) % 100 == 0:
        print( f'epoch : {epoch} / {epochs} cost : {cost.item():6f}')


hypothesis = F.softmax(x_train.matmul(W) + b)
hypothesis
y_train
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot = y_one_hot.scatter(1, y_train.unsqueeze(1), 1)

cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim = 1))).sum(dim=1).mean()
cost.item()



y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)


######################################################################################
# high-level

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optim
optimizer = optim.SGD([W, b], lr = 0.1)


epochs = 1000
for epoch in range(epochs + 1):

    # H(x)

    z = torch.matmul(x_train, W) + b 
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch) % 100 == 0:
        print( f'epoch : {epoch} / {epochs} cost : {cost.item():6f}')



#################################################################
# High - High - level Implementation with nn.Module

# linear model을 통과하고 나면 |x| = (m, 4) => (m, 3)
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

# setting model
model = SoftmaxClassifierModel()

# optim
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 1000
for epoch in range(epochs + 1):

    # H(x)
    y_hat = model(x_train)

    # cost 
    cost = F.cross_entropy(y_hat, y_train)

    # bpp
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch) % 100 == 0:
        print( f'epoch : {epoch} / {epochs} cost : {cost.item():6f}')

