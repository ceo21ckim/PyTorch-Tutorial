import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.optimizer import Optimizer 

torch.manual_seed(42)

# |x_train| = (m , 3)
x_train = torch.FloatTensor(
    [[1, 2, 1], 
    [1, 3, 2], 
    [1, 3, 4], 
    [1, 5, 5], 
    [1, 7, 5], 
    [1, 2, 5], 
    [1, 6, 6], 
    [1, 7, 7]
    ]
)

# |y_train| = (m, )
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

# |x_test| = (m', 3) , |y_test| = (m',)
x_test = torch.FloatTensor([[2, 1, 1 ], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])

class Classifiermodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x) # |x| = (m,3) => (m,3)

model = Classifiermodel()

# optimizer 
optimizer = optim.SGD(model.parameters(), lr = 0.1)

predict = model(x_train)
# 3개 중 값이 가장 높은 값을 가지고와서 예측한다! 
# 가지고온거에서 max를 취하고 dim을 사용해 위치를 찾고 슬라이싱을 통해서 값을 출력한다.
# 첫번째는 value 두번째에는 index
predict.max(1)[1]


def train(model, optimizer, x_train, y_train):
    epochs = 20
    for epoch in range(epochs):
        
        # H(x)
        prediction = model(x_train)

        # cost
        cost = F.cross_entropy(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch {epoch} / {epochs}, cost : {cost:.6f}')


def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]

    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print( f'Accuracy : {correct_count / len(y_test) * 100}%, cost : {cost:.6f}')



train(model, optimizer, x_train, y_train)


# loss값이 높은것을 확인할 수 있다. overfitting 된 것을 볼 수 있다. 
test(model, optimizer, x_test, y_test)


# learning rate가 너무 크면 diverge 하면서 cost가 너무 커진다.
model = Classifiermodel()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)

# learning rate가 너무 작아도 cost가 거의 줄어들지 않아서 안된다. 
model = Classifiermodel()
optimizer = optim.SGD(model.parameters(), lr = 1e-10)

train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)



# 학습이 너무느리거나, 발산해버린다면 learning rate를 조절하면서 하자.

model = Classifiermodel()
optimizer = optim.SGD(model.parameters(), lr = 1e-1)

train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)
