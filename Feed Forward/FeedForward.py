import torch

#선형 계층 구현

# torch.mm == 행렬과 행렬 곱
# torch.mv == 행렬과 벡터 곱
# torch.dot == 벡터 내적
# torch.matmul == 인수 설정으로 mm, mv, dot 가능
def linear(x, W, b):
    y = torch.mm(x, W) + b
    return y

x = torch.FloatTensor(16, 10)
W = torch.FloatTensor(10,5)
b = torch.FloatTensor(5)

y = linear(x, W, b)

y

# x1 = torch.FloatTensor(3,4)
# x2 = torch.FloatTensor(4,5)
# torch.mm(x1,x2).size()

# x1 = torch.FloatTensor(10,3,4)
# x2 = torch.FloatTensor(10,4,5)
# torch.bmm(x1,x2).size()