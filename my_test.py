import cv2
import numpy as np
import torch


def relation_matrix(f):
    f = f.float()
    N = f.size(0)

    # relation = f.new(N, N).fill_(0)
    # for i in range(N):
    #     for j in range(N):
    #         relation[i, j] = f[i].dot(f[j]) / (f[i].norm() * f[j].norm())

    f1 = f.clone()
    f2 = f.clone()

    f1 = f1.reshape(N, 1, -1).expand(N, N, -1)
    f2 = f2.reshape(1, N, -1).expand(N, N, -1)

    elem = torch.mul(f1, f2).sum(dim=2)
    f1_norm = torch.norm(f1, dim=2)
    f2_norm = torch.norm(f2, dim=2)

    relation = elem / torch.mul(f1_norm, f2_norm)
    return relation

f = torch.arange(8, requires_grad=True, dtype=torch.float).reshape(2, 4)

a = relation_matrix(f)

print(a)