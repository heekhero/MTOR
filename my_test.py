import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

a = nn.Conv2d(3, 10, 3, 1, 1)
b = nn.Conv2d(10, 10, 3, 1, 1)
for p in b.parameters():
    p.requires_grad = False
c = nn.Linear(1000, 2)

input = Variable(torch.randn(size=[1, 3, 10, 10], dtype=torch.float))

res1 = a(input)

# with torch.no_grad():
res2 = b(res1)

res3 = c(res2.reshape(-1, 1000))
loss = F.cross_entropy(res3, torch.tensor([1], dtype=torch.long))

loss.backward()

pass
