import torch 
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[4], [5], [6]])

W = torch.zeros(1)
# b = torch.zeros(1, requires_grad=True)


# w가 1일 때 가장 학습이 잘 되기 떄문에 1에 가까이 가야된다.

# 직접 Gradient Desent 구현하기
lr = 0.1
epochs = 10
for epoch in range(epochs + 1 ):

    hypothesis = x_train * W

    cost = torch.mean((hypothesis - y_train)**2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print( 'Epoch {:4d}/{} W: {:.3f}, Cost {:.6f}'. format(epoch, epochs, W.item(), cost.item()))

    W -= lr * gradient



# Opitm쓰기 
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W], lr = 0.15)


for epoch in range(epochs := 11):

    hypothesis = x_train * W

    cost = torch.mean((hypothesis - y_train)**2)

    print( 'Epoch {:4d}/{} W: {:.3f}, Cost {:.6f}'. format(epoch, epochs-1, W.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()






x_train = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = torch.FloatTensor([[4], [7], [10]])

W = torch.zeros([3,1], requires_grad=True)
b = torch.zeros([3,1], requires_grad=True)

optimizer = optim.Adam([W, b], lr = 0.01)

epochs = 200
for epoch in range(1, epochs + 1):

    y_hat = torch.matmul(x_train, W) + b
    
    cost = torch.mean( (y_hat - y_train)**2 )

    if epoch % 10 == 0 :
        print( f'epoch : {epoch} / {epochs}, cost : {cost.item()}')

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


y_hat = torch.matmul(x_train, W) + b
y_hat