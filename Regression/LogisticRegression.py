import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 

# Logistic Regression 
# Loss function : CrossEntropyLoss


# for reproducibility
torch.manual_seed(42)


# setting Training Data 
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0],[0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
y_train

print(x_train.shape, y_train.shape)


# Computing the Hypothesis
print('e^1 equals : ', torch.exp(torch.FloatTensor([1])))

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# sigmoid function 
hypothesis = 1 / (1 + torch.exp( -x_train.matmul(W) - b))
# hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
print(hypothesis.shape)


# crss entropy 
loss = []
for idx in range(len(y_train)):
    loss.append(-(y_train[idx] * torch.log(hypothesis[idx]) + (1-y_train[idx]) * torch.log(1-hypothesis[idx])).item())

# binaray_cross_entropy(hypothesis, y_train)


# model 
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# optimizer 
optimizer = optim.SGD([W, b], lr = 1)

epochs = 1000
for epoch in range(1, epochs + 1):

    # H(x)
    # hypothesis = 1 / (1 + torch.exp( -x_train.matmul(W) - b))
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # lossfunction
    loss = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 : 
        print( f'epoch : {epoch} / {epochs}, loss : {loss:.4f}')


##########################################################################
# Evaluation

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis[:5])


prediction = hypothesis >= torch.FloatTensor([0.5]) # byteTensor
print(prediction[:5])

print(y_train[:5])

correct_prediction = prediction.float() == y_train 
print(correct_prediction)


############################################################################
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()

# optim
optimizer = optim.SGD(model.parameters(), lr = 1)

epochs = 100
for epoch in range(1, epochs + 1):

    # H(x)
    hypothesis = model(x_train)

    # cost 
    cost = F.binary_cross_entropy(hypothesis, y_train)
 
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train 
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print(f'epoch : {epoch} / {epochs}, cost : {cost:.6f} Accuracy : {accuracy*100:.2f}% ')

