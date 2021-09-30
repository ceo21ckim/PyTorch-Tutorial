import torch
import numpy as np


# View
t = np.array(
    [[[0, 1, 2],
      [3, 4, 5]],
      
      [[6, 7, 8],
      [9, 10, 11]]])


ft = torch.FloatTensor(t)
print(ft.shape)

# view == reshape 
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

ft.reshape(-1, 3) == ft.view([-1, 3])


print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

##############################################################################
# Squeeze 
# dimention의 element  가 1인 경우 없애준다.
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

# squeeze(dim = ?)
# 해당 dimention에 1이 있을 경우 없애주는 역할을 한다.

print(ft.squeeze(dim = 0 )) # 변화가 없다. 
print(ft.squeeze(dim = 1 ))

##############################################################################
# Unsqueeze 
# 내가 지정해준 dimention에 1을 추가해주는 역할을 한다. 

ft = torch.Tensor([0, 1, 2])
print(ft.shape)


print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

# ft.view([1, -1] == ft.unsqueeze(dim = 0))
# view 는 앞에꺼 기준으로 뒤의 차원을 알아서 맞추어준다. ( -1 일 경우 )
print(ft.view([1, -1]))
print(ft.view([1, -1]).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

# 마지막 차원에 1을 해준다
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

##############################################################################
# Type Casting
# tensor의 타입을 바꾸어주겠다!
# LongTensor = 'int64'
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

# boolean 값을 1, 0 로 바꾸어준다. 
bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

# masking 이런 곳에 사용된다. 
bt = (lt == 3)

##############################################################################
# Concatenate 
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

# concatenate by dimention = 0
print(torch.cat([x, y], dim = 0))

print(torch.cat([x, y], dim = 1))


##############################################################################
# Stacking 
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

# 말그대로 쌓아라. list 안에는 tensor들이 들어가 있다. 
print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim = 1)) # 각각 columns으로 들어간다. 

# 위와 동일하다. unsqueez로 차원을 하나 만들어주고 그 차원끼리 concat 해주었다. 
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim = 0 ))



##############################################################################
# Ones and Zeros

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

# 똑같은 shape에 1이나 0으로 넣어준다. 가득채운다!
print(torch.ones_like(x))
print(torch.zeros_like(x))



##############################################################################
#In-place Operation 

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.)) # x 에다가 2를 곱해라
print(x)
print(x.mul_(2.)) # under bar 가 들어가면 임의의 메모리에 할당되지 않고 x에 바로 들어가게 된다.
print(x)