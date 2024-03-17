import numpy as np
from scipy.optimize import least_squares
import cv2


def least_square_opt(pts3d, kps1, kps2, P1, P2):
    def _reproj_err(p3d, uv1, uv2, P1, P2):
        suv1 = P1[:3, :3] @ p3d[:, None] + P1[:3, 3:4]
        e_uv1 = suv1[:2, 0] / suv1[2, 0]

        suv2 = P2[:3, :3] @ p3d[:, None] + P2[:3, 3:4]
        e_uv2 = suv2[:2, 0] / suv2[2, 0]
        err = np.sum(np.power(uv1 - e_uv1, 2)) + np.sum(np.power(uv2 - e_uv2, 2))
        return err
    
    new_pts3d = []
    errors = []
    for p3d, kp1, kp2 in zip(pts3d, kps1, kps2):
        res = least_squares(_reproj_err, p3d, args=(kp1, kp2, P1, P2))
        new_pts3d.append(res.x)
        errors.append(res.cost)
    # def reproj_err(pts3d, kps1, kps2, P1, P2):
    #     uvs1 = pts3d @ P1[:3, :3].T + P1[:3, 3:4].T
    #     uvs1 = uvs1[:, :2] / uvs1[:, 2:3]
    #     uvs2 = pts3d @ P2[:3, :3].T + P2[:3, 3:4].T
    #     uvs2 = uvs2[:, :2] / uvs2[:, 2:3]
    #     err1_sqr = np.sum(np.power(uvs1 - kps1, 2), axis=1)
    #     err2_sqr = np.sum(np.power(uvs2 - kps2, 2), axis=1)
    #     return np.mean(err1_sqr)+np.mean(err2_sqr) 

    # res = least_squares(reproj_err, pts3d, args=(kps1, kps2, P1, P2))
    # new_pts3d = res.x
        
    return np.array(new_pts3d), errors 
    


