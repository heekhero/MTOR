import cv2
import numpy as np
import torch

# img = cv2.imread('/data/fuminghao/data/VOCdevkit/VOC2007/JPEGImages/005436.jpg')
# res = np.uint8(np.clip((img + 100.0), 0, 255))
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# cv2.imwrite('res.jpg', res)
# cv2.imwrite('gray.jpg', gray)

from model.utils.net_utils import relation_matrix, guide_matrix

a = torch.randperm(5)

print(a)