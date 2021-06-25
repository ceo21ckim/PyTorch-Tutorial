import torch
import numpy as np

# torch의 Tensor는 numpy에 np.array랑 같다. 
# 추가로 텐서간 연산에 따른 그래프와 경사도(gradient)를 저장할 수 있다.
x = torch.Tensor([[1,2],[3,4]])
print(x)
x = torch.from_numpy(np.array([[1,2],[3,4]]))
print(x)

x = np.array([[1,2],[3,4]])


#32-bit floating point / default
torch.FloatTensor([[1,2],[3,4]])

#64-bit floating point
torch.DoubleTensor([[1,2],[3,4]])

#16-bit floating point
torch.HalfTensor([[1,2],[3,4]])

#8-bit integer
torch.ByteTensor([[1,2],[3,4]])

#16-bit integet
torch.ShortTensor([[1,2],[3,4]])

#32-bit integer
torch.IntTensor([[1,2],[3,4]])

#64-bit integer
torch.LongTensor([[1,2],[3,4]])