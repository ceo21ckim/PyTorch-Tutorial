import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

class MyLinear(nn.Module):
    def __init__(self):    
        super().__init__()
        self.Linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.Linear(x)



# 현실에서는 위와 같은 적은 데이터로는 하지 않는다.
# 엄청난 양의 데이터를 한번에 학습시킬 수 없다. 1. 너무느리다. 2. 하드웨어적으로 불가능하다.

# 그렇기 때문에 일부분의 데이터로만 학습을 하려고 한다. 

# minibatch 형태로 짤라 minibatch 단위로 균일하게 나누어 학습하는 방식이 등장했다.
# 업데이트가 조금 더 빨라질 수 있지만, 전체 데이터를 사용하지 않아서 잘못된 방향으로 업데이트를 할 수도 있다.
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor(
                        [[73, 80, 75], 
                        [93, 88, 93],
                        [89, 91, 90], 
                        [96, 98, 100],
                        [73, 66, 70]]
                        )
        
        self.y_data = [[152], [185], [180], [196], [142]]


    def __len__(self): # 데이터 셋의 총 길이
        return len(self.x_data)

    def __getitem__(self, idx): # 해당 인덱스 idx를 받았을 때 그에 상응하는 입출력 데이터를 반환.
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

dataset = CustomDataset()

# utils data 에서 dataset을 만들었으면 이제 DataLoader를 사용할 수 있다.
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, 
    batch_size = 2, # 각 minibatch의 사이즈 통상적으로 2의 제곱수를 설정한다. 
    shuffle = True # Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다. True를 설정하는 것을 권장
)

model = MyLinear()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

epochs = 20 
for epoch in range(1, epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples 

        # H(x)
        y_hat = model(x_train)

        # cost 
        cost  = F.mse_loss(y_hat, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch : {epoch} / {epochs}, batch {batch_idx + 1} / {len(dataloader)} : {cost:.6f}') # len(dataloader) epoch당 minibatch개수