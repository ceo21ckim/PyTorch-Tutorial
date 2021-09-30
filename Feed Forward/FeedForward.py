from typing_extensions import Required
import torch
import torch.nn as nn
import numpy as np
class Mymodel(nn.Module):
    def __init__(self, input_size, output_size):
        
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y = self.linear(x)
        return y

def ground_truth(x):
    return 3 * x[:, 0] + x[:, 1] -2 * x[:,2]


def train(model, x, y, optim):
    optim.zero_grad()

    # feed-foward
    y_hat = model(x)

    # get error between answer and inferenced.
    loss = ((y-y_hat)**2).sum() / x.size(0)

    # back-propagation
    loss.backward()

    optim.step()

    return loss.data




batch_size = 1
n_epochs = 1000
n_iter = 10000

model = Mymodel(3, 1)
# model.parameters : dim / lr : learning rate 
optim = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.1)

print(model)

# training

for epoch in range(n_epochs):
    avg_loss = 0 # loss를 계속해서 더해가면서 최소가 되는 지점을 찾는다.

    for i in range(n_iter):
        x = torch.rand(batch_size, 3) # 1 by 3 where batch_size == 1
        y = ground_truth(x.data)

        loss = train(model, x, y, optim) # optim : SGD

        avg_loss += loss
        avg_loss = avg_loss / n_iter

    # simple test sample to check the network.
    x_valid = torch.FloatTensor([[.3, .2, .1]])
    y_valid = ground_truth(x_valid.data)

    model.eval()

    y_hat = model(x_valid)
    model.train()

    print( avg_loss, y_valid.data[0], y_hat.data[0, 0])

    if avg_loss < .001: # finish the training if the loss is smaller than .001 
        break


