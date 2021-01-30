import cv2
import numpy as np

import reconstruction as rc

# Test compute_proj_camera

F = np.array( [-1.31654128e-06,  5.69009628e-04, -2.71449753e-01,
               -9.81764197e-04, -5.19409197e-05,  1.17026528e+00,
                4.40849083e-01, -1.20122485e+00,  6.19909507e+01] ).reshape(3,3)
P = rc.compute_proj_camera(F, 1)

print(P)


# Test reprojection error
X = np.array([[3,3,3,1],[3,3,3,1]]).T
x1 = np.array([[3,3],[2,2]])
x2 = np.array([[2,2],[2,2]])
P1 = np.ones((3,4))
P2 = np.ones((3,4))

x = rc.compute_reproj_error(X,P1,P2,x1,x2)
print(x)


# Test estimate_aff_hom
vps = [np.array([[10600.322  ,  -381.79193],
                [  602.0181 ,   492.94623],
                [ 2643.6978 , 13842.847  ]]), 
       np.array([[16563.812  ,   457.51215],
                [  993.58014,   457.51215],
                [ 1520.69   , 14892.993  ]]) ]

# TODO

