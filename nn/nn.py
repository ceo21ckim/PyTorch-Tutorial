import torch
import torch.nn as nn 

class MyLinear(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = torch.Tensor(input_size, output_size)
        self.b = torch.Tensor(output_size)

    def forward(self, x):
        y = torch.mm(x, self.W) + self.b

        return y

x = torch.Tensor(16, 10)
linear = MyLinear(10, 5)
y = linear(x)

y
