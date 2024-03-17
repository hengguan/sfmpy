import numpy as np
from numpy.typing import NDArray 
import scipy
import cv2

from geometry import reproj_points, homo_points, to_skew_matrix

    
def linear_triangulation(P1, P2, kps1, kps2):
    pts3d = []
    for kp1, kp2 in zip(kps1, kps2):
        T1 = to_skew_matrix(kp1) @ P1
        T2 = to_skew_matrix(kp2) @ P2
        A = np.concatenate([T1[:2], T2[:2]], axis=0) # 4x4
        u, s, vh = scipy.linalg.svd(A)
        p3d = vh[-1, :] / vh[-1, -1]
        pts3d.append(p3d)
    return np.stack(pts3d)
    
    
def normalize_points(xs: NDArray):
    mean = np.mean(xs, axis=0, keepdims=True)
    max_dist = np.max(np.linalg.norm(xs - mean, axis=1))
    scale = 1 / max_dist
    
    trans = np.diag([scale, scale, 1])
    trans[:, 2:3] = - trans @ mean.T
    return xs @ trans.T, trans


def estimate_fundamental_matrix(xs1: NDArray, xs2: NDArray, num_iter=100, min_ratio=0.9, reproj_threshold=1.):
    """Estimate fundamental matrix using eight points method."""
    assert len(xs1) == len(xs2), f"the number of image point is not equal: {len(xs1)}!={len(xs2)}"
    num_pairs = len(xs1)
    count_iter = 0
    curr_ratio = 0.
    choosed = set()
    homo_xs1 = np.concatenate([xs1, np.ones((num_pairs, 1))], axis=1)
    homo_xs2 = np.concatenate([xs2, np.ones((num_pairs, 1))], axis=1)
    norm_xs1, tf1 = normalize_points(homo_xs1)
    norm_xs2, tf2 = normalize_points(homo_xs2)
    ret_mat = None
    max_inlier = 0
    best_inlier_mask = None
    while count_iter < num_iter and curr_ratio < min_ratio:
        idxs = np.random.choice(num_pairs, 8, replace=False)
        if tuple(idxs) in choosed:
            continue
        choosed.add(tuple(idxs))
        A = np.stack([(np.expand_dims(norm_xs2[i], 1) @ np.expand_dims(norm_xs1[i], 0)).flatten() for i in idxs])
        u, s, vh = scipy.linalg.svd(A)
        min_ind = np.argsort(s)[0]
        ret_mat = vh[min_ind].reshape(3, 3)
        rank = np.linalg.matrix_rank(ret_mat)
        if rank == 3:
            u, s, vh = scipy.linalg.svd(ret_mat)
            s[-1] = 0
            ret_mat = u @ np.diag(s) @ vh
        elif rank < 2:
            print("ret matrix rank is not true")
            continue
        
        F21 = tf2.T @ ret_mat @ tf1
        dist, valid = distance_to_epipole_line(F21, homo_xs1, homo_xs2)
        inlier_mask = np.zeros(num_pairs, np.bool_)
        inlier_mask[valid] = dist <= reproj_threshold
        num_inlier = np.count_nonzero(inlier_mask)
        if num_inlier > max_inlier:
            max_inlier = num_inlier
            best_inlier_mask = inlier_mask
        curr_ratio = num_inlier / num_pairs

        count_iter += 1
    return F21, best_inlier_mask


def distance_to_epipole_line(F12, homo_xs1, homo_xs2):
    line = F12 @ homo_xs1.T
    m = np.linalg.norm(line[:2].T, axis=1)
    v = np.sum(homo_xs2 * line.T, axis=1)
    valid = m != 0
    dist = np.abs(v[valid] / m[valid])
    return dist, valid
    

def estimate_E_from_F(F, K1, K2):
    return K2.T @ F @ K1


from viz import show_points_3d, show_points
def decompose_E(E, K1, K2, kps1, kps2, reproj_threshold=4.):
    u, s, vh = scipy.linalg.svd(E)
    y = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R1 = u @ y @ vh
    R2 = u @ y.T @ vh
    t = u[:, -1]

    Rs, ts = [R1, R1, R2, R2], [t, -t, t, -t]
    # skew_kps1 = np.stack([to_skew_matrix(x) for x in kps1])
    # skew_kps2 = np.stack([to_skew_matrix(x) for x in kps2])
    best_valid = 0
    best_idx = -1
    best_pts3d = None
    best_mask = None
    for idx, (R, t) in enumerate(zip(Rs, ts)):
        P1 = np.concatenate([K1, np.zeros((3, 1))], axis=1)
        P2 = K2 @ np.concatenate([R, t[:, None]], axis=1)
        pts3d = linear_triangulation(P1, P2, kps1, kps2)
        mask = pts3d[:, 2] > 0
        cam_pts2 = R @ pts3d[:, :3].T + t[:, None]
        cam2_mask = cam_pts2[2, :] > 0
        mask = np.logical_and(mask, cam2_mask)
        num_valid = np.count_nonzero(mask) + np.count_nonzero(cam2_mask)
        # print(num_valid, len(kps1))
        if num_valid > best_valid:
            best_valid = num_valid
            best_idx = idx 
            best_pts3d = pts3d[mask, :3]
            best_mask = mask

    uv1, d1, m1 = reproj_points(best_pts3d, K1, np.eye(3), np.zeros((3, 1)))
    reproj_err1 = np.linalg.norm(uv1 - kps1[best_mask, :2], axis=1)

    uv2, d2, m2 = reproj_points(best_pts3d, K2, Rs[best_idx], ts[best_idx])
    reproj_err2 = np.linalg.norm(uv2 - kps2[best_mask, :2], axis=1)
    valid = np.logical_and(reproj_err1<reproj_threshold, reproj_err2<reproj_threshold)
    valid = np.logical_and(m1, valid)
    valid = np.logical_and(m2, valid)

    # show_points_3d(best_pts3d[valid, :3])
    return Rs[best_idx], ts[best_idx][:, None]


def recover_pose_from_F(F, K1, K2, kps1, kps2):
    # E = estimate_E_from_F(F, K1, K2)
    E, mask = cv2.findEssentialMat(
        kps1,
        kps2,
        K1,
    )
    ret, R, t, mask = cv2.recoverPose(
        E,
        kps1,
        kps2,
        K1
    )
    # print(R.shape, t.shape)
    # points4d = cv2.triangulatePoints(
    #     np.concatenate([K1, np.zeros((3, 1))], axis=1),
    #     K2 @ np.concatenate([R, t], axis=1),
    #     kps1.T,
    #     kps2.T
    # )
    # print(points4d.shape)
    # pts3d = points4d[:3, :].T / points4d[3:4, :].T
    # pts3d = linear_triangulation(
    #     np.concatenate([K1, np.zeros((3, 1))], axis=1),
    #     K2 @ np.concatenate([R, t], axis=1),
    #     homo_points(kps1),
    #     homo_points(kps2)
    # )
    # show_points_3d(pts3d)
    R, t = decompose_E(E, K1, K2, homo_points(kps1), homo_points(kps2))
    return ret, R, t, mask
    