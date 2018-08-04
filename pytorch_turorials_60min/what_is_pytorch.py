# from __future__ import print_function
import torch

# x = torch.empty(5, 3)
# print(x)

# x = torch.rand(5, 3)
# print(x)

# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)


# Torch Tensor and operations

# x = torch.tensor([5.5, 3])
# print(x)
# # print(x.type())
# # print(x.size())
#
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# #
# x = torch.randn_like(x, dtype=torch.float)
# print(x)
# print(x.size())
#
# y = torch.rand(5, 3)
# print(x + y)
#
# print(torch.add(x, y))
#
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
#
# y.add_(x)
# print(y)
#
# print(x[:, 1])


# Resize

# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)
# print(x.shape, y.shape, z.shape)

# x = torch.randn(1)
# print(x)
# print(x.item())


# Numpy Bridge

# a = torch.ones(5)
# print(a)
# print(type(a))
#
# b = a.numpy()
# print(b)
# print(type(b))
#
# a.add_(1)
# print(a)
# print(b)

# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)


# CUDA Tensors
# if torch.cuda.is_available():
#     print("Support cuda")
# else:
#     print("Not support cuda")
#
# x = torch.tensor([5.5, 3])
# x = x.new_ones(5, 3, dtype=torch.double)
# device = torch.device("cuda")
# print(device)
# y = torch.ones_like(x, device=device)
