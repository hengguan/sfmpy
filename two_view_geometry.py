import numpy as np
from numpy.typing import NDArray 
import scipy


def normalize_points(xs: NDArray):
    mean = np.mean(xs, axis=0, keepdims=True)
    max_dist = np.max(np.linalg.norm(xs - mean, axis=1))
    scale = 1 / max_dist
    
    trans = np.diag([scale, scale, 1])
    trans[:, 2:3] = - trans @ mean.T
    print(trans)
    return xs @ trans.T, trans


def estimate_fundamental_matrix(xs1: NDArray, xs2: NDArray, num_iter=100, min_ratio=0.9, threshold=1.):
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
        print(s, vh.shape)
        min_ind = np.argsort(s)[0]
        print(s[min_ind], min_ind)
        ret_mat = vh[min_ind].reshape(3, 3)
        rank = np.linalg.matrix_rank(ret_mat)
        if rank == 3:
            u, s, vh = scipy.linalg.svd(ret_mat)
            print(s)
            s[-1] = 0
            ret_mat = u @ np.diag(s) @ vh
        elif rank < 2:
            print("ret matrix rank is not true")
            continue
        
        F21 = tf2.T @ ret_mat @ tf1
        dist, valid = distance_to_epipole_line(F21, homo_xs1, homo_xs2)
        inlier_mask = np.zeros(num_pairs, np.bool_)
        inlier_mask[valid] = dist <= threshold
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