import torch

t = torch.tensor([[1,2,3],
              [4,5,6]])

t2 = t[1, 1:3]
print(t2.shape)
print(len(t2))

