import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train = torch.FloatTensor(
    [[73, 80, 75], 
    [93, 88, 93],
    [89, 91, 90], 
    [96, 98, 100],
    [73, 66, 70]]
)

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros([3, 1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

y_hat = x_train.matmul(W) + b

# learning rate에 따라 발산할 수도 있다. 너무 크면 발산하더라..
optimizer = optim.SGD([W, b], lr = 1e-5)

epochs = 20
for epoch in range(1, epochs + 1):
    
    y_hat = x_train.matmul(W) + b

    cost = F.mse_loss(y_hat, y_train)

    print(f'epoch : {epoch} / {epochs:d}, y_hat : {y_hat.squeeze().detach().tolist()},  cost : {cost.item():.4f}')

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


##########################################################
# nn.Module 이라는 편리한 모델을 사용해보자

x_train = torch.FloatTensor(
    [[73, 80, 75], 
    [93, 88, 93],
    [89, 91, 90], 
    [96, 98, 100],
    [73, 66, 70]]
)

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])




class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3, 1) # 3 : input dimention / 1 : output dimention

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

# create model
model = MyLinear()
# optimizer 
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

epochs = 30
for epoch in range(1, epochs + 1):
    y_hat = model(x_train)

    cost = F.mse_loss(y_hat, y_train)

    print(f'epoch : {epoch} / {epochs}, y_hat : {y_hat.squeeze().detach()}, cost : {cost:.6f}')

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

