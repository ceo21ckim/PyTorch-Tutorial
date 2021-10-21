import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.optimizer import Optimizer 

torch.manual_seed(42)



# |x_train| = (m, 3), |y_train| = (m, )
x_train = torch.FloatTensor(
    [[73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100], 
    [73, 66, 70]
    ]
)

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# normalization 
mu = x_train.mean(dim=0)
sigma = x_train.std(dim = 0 )

norm_x_train = (x_train - mu) / sigma 
print(norm_x_train)



class Regressionmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 하나의 값을 예측해야하기 때문에 1

    def forward(self, x):
        return self.linear(x) # |x| = (m,3) => (m,3)

model = Regressionmodel()

# optimizer 
optimizer = optim.SGD(model.parameters(), lr = 0.1)


def train(model, optimizer, x_train, y_train):
    epochs = 20
    for epoch in range(epochs):
        
        # H(x)
        prediction = model(x_train) # |x_train| = (m, 3) , |prediction| = (m, 1)

        # cost
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch {epoch} / {epochs}, cost : {cost:.6f}')


def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction

    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.mse_loss(prediction, y_test)

    print( f'Accuracy : {correct_count / len(y_test) * 100}%, cost : {cost:.6f}')

# preprocessing 을 하지 않으면 문제가 발생할 수 있다. 
# 지금은 y_train이 하나의 차원을 받지만, 만약 두개의 값을 반환받을때에는 전처리르 하지 않을떄 엄청난 문제가 된다. 
train(model, optimizer, norm_x_train, y_train)


