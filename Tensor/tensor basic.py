import numpy as np 
import torch 

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)


print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])


t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.] ])



print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 1D array with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim())
print(t.shape)
print(t.size())
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])


# 2D array with PyTorch
t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.] ])
print(t)

print(t.dim()) # rank
print(t.size()) # shape
print(t[:,1])
print(t[:, 1].size())
print(t[:, :-1])


# Broadcasting 
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)
# tensor([[5., 5.]])


# Vector + scalar
m1 = torch.Tensor([[1, 2]])
m2 = torch.Tensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)
# tensor([[4., 5.]])


# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.Tensor([[1, 2]])
m2 = torch.Tensor([[3], [4]])
print(m1 + m2)
# tensor([[4., 5.],
#         [5., 6.]])


# Multiplication vs Matrix Multiplication 
print()
print('----------')
print('Mul vs Matmul')
print('----------')
m1 = torch.Tensor([[1, 2], [3, 4]])
m2 = torch.Tensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1.matmul(m2))


m1 = torch.Tensor([[1, 2] [3, 4]])
m2 = torch.Tensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1*m2)
print(m1.mul(m2))

# Mean
t = torch.FloatTensor([1, 2])

print(t.mean())

# Can't use mean() on integers
# longtensor == 64-bit integer
t = torch.LongTensor([1, 2])
try :
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())
print(t.mean(dim = 0 ))
# tensor([2., 3.])

print(t.mean(dim=1))
# tensor([1.5000, 3.5000])

print(t.mean(dim=-1))
# tensor([1.5000, 3.5000])

# Sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.sum())
print(t.sum(dim = 0))
print(t.sum(dim = 1))
print(t.sum(dim = -1))

# max and Argmax
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())

# 총 두가지를 반환해준다.
print(t.max(dim= 0 )) # return two values : max and argmax
print('Max : ', t.max(dim=0)[0])
print('Argmax : ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))

