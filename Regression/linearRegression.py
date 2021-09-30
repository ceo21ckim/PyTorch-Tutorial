import torch
import torch.nn as nn 
import torch.optim as optim
# 입력과 출력은 각기 다른 곳에 저장해준다. 
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[4], [5], [6]])


# y = Wx + b
# W = weight, b = bias
W = torch.zeros(1, requires_grad=True) # requires_grad : 학습을 할 것인가?
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b # 0으로 초기화


# Mean Squared Error ( MSE )
# cost function 
loss = []
epochs = 1000
for epoch in range(1, epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean( (hypothesis - y_train )**2 )
    optimizer = optim.SGD([W, b], lr = 0.01)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0 :
        print( f'epoch {epoch} / {epochs},  cost : {cost}')

