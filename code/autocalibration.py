import numpy as np
import scipy
import reconstruction as rc
import maths as mth
import fundamental as fd


def estimate_aff_hom(cams, vps, vp_3d=False):

    if not vp_3d:
        # Triangulate all vanishing points
        vps = rc.estimate_3d_points_2(cams[0], cams[1], vps[0].T, vps[1].T)

    # Estimate p
    U, D, Vt = np.linalg.svd(vps.T)
    p = Vt.T[:, -1]
    p = p/p[-1]

    # Compose affine homography
    aff_hom = np.zeros((4,4))
    aff_hom[0:3,0:3] = np.eye(3)
    aff_hom[3,0:4] = [p[0], p[1], p[2], 1]

    return aff_hom



def estimate_euc_hom(cams, vps):
    # make points homogeneous
    u = np.array([vps[0,0], vps[0,1], 1])
    v = np.array([vps[1,0], vps[1,1], 1])
    z = np.array([vps[2,0], vps[2,1], 1])

    # Build the matrix A
    A = np.array([[u[0]*v[0], u[0]*v[1] + u[1]*v[0], u[0]*v[2] + u[2]*v[0], u[1]*v[1],
                    u[1]*v[2] + u[2]*v[1], u[2]*v[2]],
                    [u[0]*z[0], u[0]*z[1] + u[1]*z[0], u[0]*z[2] + u[2]*z[0], u[1]*z[1],
                    u[1]*z[2] + u[2]*z[1], u[2]*z[2]],
                    [v[0]*z[0], v[0]*z[1] + v[1]*z[0], v[0]*z[2] + v[2]*z[0], v[1]*z[1],
                    v[1]*z[2] + v[2]*z[1], v[2]*z[2]],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, -1, 0, 0]])

    # Obtain the image of the absolute conic
    w_vec = mth.nullspace(A) 
    w = np.array([w_vec[0], w_vec[1], w_vec[2], 
                  w_vec[1], w_vec[3], w_vec[4],
                  w_vec[2], w_vec[4], w_vec[5]]).reshape(3,3)
    #print(w)

    # Obtain M matrix from P
    M = cams[:,0:3]
    AAT = np.linalg.inv(M.T @ w @ M)
    A = np.linalg.cholesky(AAT)
    #print(A/A[2,2])
    
    # Build euc_hom
    euc_hom = np.zeros((4,4))
    euc_hom[0:3,0:3] = np.linalg.inv(A)
    euc_hom[3,3] = 1

    return euc_hom


def estimate_aff_hom_F(cams, vps, F):
    #Extra: estimate affine homography from F
    #Algorithm 13.1 of MVG

    # Get epipolar line and compute A
    ep = cams[1][:,-1]
    A = mth.hat_operator(ep) @ F

    # Obtain 3D vanishing points
    vps_3D = rc.estimate_3d_points_2(cams[0], cams[1], vps[0].T, vps[1].T)
    vps_3D = vps_3D[0:3,0:3]/vps_3D[-1,:]

    # Assemble M and b
    M = []
    B = []
    for i in range(0,3):
        M.append(vps_3D[i,:].T) 
        xp = np.array([vps[1][i,0],vps[1][i,1],1]) 
        x = np.array([vps[0][i,0],vps[0][i,1],1])
        ax = A @ x
        b = (np.cross(xp,ax).T @ np.cross(xp,ep)) / (np.cross(xp,ep).T @ np.cross(xp,ep))
        B.append(b)

    # Solve the linear problem
    v = np.linalg.solve(M,B)

    # Return the matrix
    aff_hom = np.zeros((4,4))
    aff_hom[0:3,0:3] = A - ep @ v.T
    aff_hom[3,3] = 1

    return aff_hom








