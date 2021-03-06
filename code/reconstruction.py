import cv2
import numpy as np

import utils as h
import maths as mth

from scipy import optimize as opt


def compute_proj_camera(F, i):
    # Result 9.15 of MVG (v = 0, lambda = 1). It assumes P1 = [I|0]
    # P' = [[e']_x F | e']

    # Find e', epipole such that e'^T F = 0
    et = np.array(mth.nullspace(F.T))

    # Get [e']_x
    e_skew = mth.hat_operator(et)

    # Construct P
    P = np.zeros((3,4))
    P[0:3,0:3] = e_skew @ F
    P[:,-1] = et.T

    return P

def estimate_3d_points_2(P1, P2, xr1, xr2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = xr1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (xr1[0, i] * P1[2, :] - P1[0, :]),
            (xr1[1, i] * P1[2, :] - P1[1, :]),
            (xr2[0, i] * P2[2, :] - P2[0, :]),
            (xr2[1, i] * P2[2, :] - P2[1, :])
        ])

        _, _, V = np.linalg.svd(A)

        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res

def compute_reproj_error(X, P1, P2, xr1, xr2):
    # Check if we need to change to homogeneous
    dim,N = np.shape(X)
    if dim == 3:
        Xp = np.ones((4,N))
        Xp[0:3,:] = X
        X = Xp

    # Project 3D points using Pi
    xpr1 = P1 @ X
    xpr2 = P2 @ X

    # Change to euclidean
    xpr1_e = xpr1[0:2,:]/xpr1[-1,:]
    xpr2_e = xpr2[0:2,:]/xpr2[-1,:]

    # Diff between projected and computed points
    diff1 = (xpr1_e-xr1)**2
    diff2 = (xpr2_e-xr2)**2

    # Average
    error = np.sum(np.sum(diff1 + diff2))/(2*N)

    return error


def transform(aff_hom, Xprj, cams_pr):
    # Algorithm 19.2 of MVG

    Xaff =  aff_hom @ Xprj
    cams_aff = cams_pr @ np.linalg.inv(aff_hom)
    #Xaff =  np.linalg.inv(aff_hom) @ Xprj
    #cams_aff = cams_pr @ aff_hom
    Xaff[:,:]=Xaff[:,:]/Xaff[3,:]

    return Xaff, cams_aff


def resection(tracks, i):
    # extract 3D-2D correspondences from tracks
    # Space points
    pts3d = []

    # Image points
    pts2d = []

    # Get 3D - 2D correspondences from tracks
    for t in tracks:
        pts_3d.append(t.pt)
        pts2d.append(t.ref_views[i])

    # Convert 2D points to homogeneous coordinates
    pts2d = homog(np.array(pts2d))

    # Scale factors  
    sx = np.std(pts2d[:, 0])/np.sqrt(2)
    sy = np.std(pts2d[:, 1])/np.sqrt(2)

    # Translation factors (subtract mean to points)
    tx = np.mean(pts2d[:, 0])
    ty = np.mean(pts2d[:, 1])

    # Similarity transform for image points
    T = np.array([[sx, 0, -tx],
                  [0, sy, -ty],
                  [0, 0, 1]])

    
    # Transform points
    pt2d_n = pts2d @ T

    pts3d = np.array(pts3d)
    # Scale factors
    sx = np.std(pts3d[:, 0])/np.sqrt(2)
    sy = np.std(pts3d[:, 1])/np.sqrt(2)
    sz = np.std(pts3d[:, 2])/np.sqrt(2)

    # Translation factors (subtract mean to points)
    tx = np.mean(pts3d[:, 0])
    ty = np.mean(pts3d[:, 1])
    tz = np.mean(pts3d[:, 2])

    # Similarity transform for space points
    U = np.array([[sx, 0, 0, -tx],
                   [0, sy, 0, -ty],
                   [0, 0, sz, -tz],
                   [0, 0, 0, 1]])


    pt3d_n = pts3d @ U

    # Number of 2D image points
    num_points = np.shape(pts2d)[0]

    # Definition of matrix A, such that Ap = 0
    A = np.zeros((2*num_points,12))

    # Iterate through rows to build matrix
    row = 0

    for i in range(0, num_points):
        # Image points
        x2d = pt2d_n[i, :]
        x3d = pt3d_n[i, :]
        # Homogeneous coordinate of image points
        w2d = -x2d[2]
        # First row of equation 7.2 from MVG
        A[row, :] = np.array([0, 0, 0, 0, #0 transposed
                            #-wiXiT
                            -w2d * x3d[0], -w2d * x3d[1], -w2d * x3d[2], -w2d * x3d[3], 
                            # yiXiT
                            x2d[1] * x3d[0], x2d[1] * x3d[1], x2d[1] * x3d[2], x2d[1] * x3d[3]])
        # Second row of equation 7.2 from MVG
        A[row+1, :] = np.array([w2d * x3d[0], w2d * x3d[1], w2d * x3d[2], w2d * x3d[3], #wiXiT
                                0, 0, 0, 0, # 0 transposed
                                -x2d[0] * x3d[0], -x2d[0] * x3d[1], -x2d[0] * x3d[2], -x2d[0] * x3d[3]]) #-xiXiT
        row = row + 2
    # SVD of matrix A
    U, D, Vt = np.linalg.svd(A)
    # Camera projection matrix P (we know Vt.T contains the entries P1 P2 P3 stacked)
    P = Vt.T[:, -1].reshape((3,4))
    # Denormalization
    P = np.linalg.inv(T) @ P @ U

    return P

def homog(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

def euclid(x):
    return x[:, :-1] / x[:, [-1]]

def compute_eucl_cam(F,x1, x2):

    K = np.array([[2362.12, 0, 1520.69], [0, 2366.12, 1006.81], [0, 0, 1]])
    E = K.T @ F @ K

    # camera projection matrix for the first camera
    P1 = K @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))    
    
    # create matrices (Hartley p 258)
    Z = mth.skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # return all four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
             np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
            np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
            np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]

    ind = 0
    maxres = 0

    for i in range(4):
        # triangulate inliers and compute depth for each camera
        homog_3D = cv2.triangulatePoints(P1, P2[i], x1[:2], x2[:2])
        # the sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]
            
        if sum(d1 > 0) + sum(d2 < 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 < 0)
            ind = i
            infront = (d1 > 0) & (d2 < 0)

    list_cams = []
    list_cams.append(P1)
    list_cams.append(P2[ind])

    return list_cams
