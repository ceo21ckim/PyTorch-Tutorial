import torch

# 최종적으로 나온 scalar 값에 propagation Algorithm을 통해 미분을 수행했을 때, 자동으로 찾아 계속해서 propagation을 수행하도록 한다.
x = torch.Tensor(2,2) # 공간에 가비지 값으로 할당된다.
y = torch.Tensor(2,2) # https://pythonq.com/so/python/749075
y.requires_grad_(True) # propagation

z = (x + y) + torch.Tensor(2,2)
z
# torch.Tensor(2,2) == torch.empty(2,2)


# 기울기를 구할 필요 없는 연산의 경우 with문법을 사용
# propagation이 필요없는 비 학습 과정, 예측, 추론 등을 수행할 때 유용

x = torch.Tensor(2,2) 
y = torch.Tensor(2,2) 
y.requires_grad_(True)

with torch.no_grad():
    z = (x + y) + torch.FloatTensor(2,2)
