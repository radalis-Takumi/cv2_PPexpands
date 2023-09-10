import numpy as np
import cv2

a = np.array([[[0, 1]], [[2, 3]]])
b = a.transpose(1, 0, 2)
print(a)
print(a[:, 0, :])
print(b)
