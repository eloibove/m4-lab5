import cv2
import numpy as np

import reconstruction as rc


X = np.array([[3,3,3,1],[3,3,3,1]]).T
x1 = np.array([[3,3],[2,2]])
x2 = np.array([[2,2],[2,2]])
P1 = np.ones((3,4))
P2 = np.ones((3,4))

x = rc.compute_reproj_error(X,P1,P2,x1,x2)
print(x)