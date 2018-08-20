import torch

# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
# print(z, out)
#
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b)
# print(b.grad_fn)
#
# out.backward()
# print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)