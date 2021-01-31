import numpy as np
import scipy
import reconstruction as rc
import maths as mth
import fundamental as fd


def estimate_aff_hom(cams, vps):
    # Triangulate all vanishing points
    vps_3D = rc.estimate_3d_points_2(cams[0], cams[1], vps[0].T, vps[1].T)

    # Estimate p
    U, D, Vt = np.linalg.svd(vps_3D.T)
    p = Vt.T[:, -1]
    p = p/p[-1]

    # Compose affine homography
    aff_hom = np.zeros((4,4))
    aff_hom[0:3,0:3] = np.eye(3)
    aff_hom[3,0:4] = [-p[0], -p[1], -p[2], 1]

    return aff_hom


def estimate_euc_hom(cams, vps):
    # make points homogeneous
    
    ...

    return euc_hom
