import numpy as np
from numpy.typing import NDArray
import cv2
from scipy.optimize import least_squares


def homo_points(xs: NDArray):
    return np.concatenate([xs, np.ones((len(xs), 1))], axis=1)


def to_skew_matrix(x):
    x1, x2, x3 = x
    return np.array([
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ])


def reproj_points(pts, K, R, t):
    cam_pts = pts @ R.T + t.T
    suv = cam_pts @ K.T
    depth = cam_pts[:, 2]
    mask = depth > 0
    uv = suv[:, :2] / suv[:, 2:3]
    return uv, depth, mask


def linear_pnp(pts3d, kps, reproj_threshold=4.0):
    """Linear perspective-n-projection using least square.

    Args:
        pts3d: points 3d,
        kps: normalized camera points: K.inv() @ kps
        reproj_threshold: for mask outliers.
    Return:
        rotation matrix and translate vector.
    """
    def least_square_pnp(tf, kp, pt3d):
        z = sum(tf[8:] * pt3d)
        x = sum(tf[:4] * pt3d) / z
        y = sum(tf[4:8] * pt3d) / z
        return np.power(x-kp[0], 2) + np.power(y-kp[1], 2)

    if pts3d.shape[0] < 6:
        return False, None

    if pts3d.shape[1] == 3:
        pts3d = homo_points(pts3d)

    tf = np.concatenate([np.eye(3), np.zeros(3, 1)], axis=1).flatten()
    print(tf)
    res = least_squares(least_square_pnp, tf, args=(kps, pts3d))

    
