import numpy as np
from numpy.typing import NDArray 
import scipy


def estimate_fundamental_matrix(xs1: NDArray, xs2: NDArray, num_iter=100, min_ratio=0.9, threshold=1.):
    """Estimate fundamental matrix using eight points method."""
    assert len(xs1) == len(xs2), f"the number of image point is not equal: {len(xs1)}!={len(xs2)}"
    num_pairs = len(xs1)
    count_iter = 0
    curr_ratio = 0.
    choosed = set()
    homo_xs1 = np.concatenate([xs1, np.ones((num_pairs, 1))], axis=1)
    homo_xs2 = np.concatenate([xs2, np.ones((num_pairs, 1))], axis=1)
    inlier_mask = np.zeros(num_pairs, np.bool_)
    ret_mat = None
    while count_iter < num_iter and curr_ratio < min_ratio:
        idxs = np.random.choice(num_pairs, 8, replace=False)
        if tuple(idxs) in choosed:
            continue
        choosed.add(tuple(idxs))
        A = np.stack([(np.expand_dims(homo_xs2[i], 1) @ np.expand_dims(homo_xs1[i], 0)).flatten() for i in idxs])
        u, s, vh = scipy.linalg.svd(A)
        print(u.shape, s.shape, vh.shape)
        min_ind = np.argsort(s)[0]
        print(s[min_ind])
        if s[min_ind] == 0:
            ret_mat = vh[-1].reshape(3, 3)
        else:
            s[min_ind] = 0
            s_mat = np.concatenate([np.diag(s), np.ones((8, 1))], axis=1)
            print(u.shape, s_mat.shape, vh.shape)
            new_A = u @ s_mat @ vh
            u, s, vh = scipy.linalg.svd(new_A)
            min_ind = np.argsort(s)[0]
            print(s[min_ind])
            assert s[min_ind] == 0, "svd not true"
            ret_mat = vh[-1].reshape(3, 3)
        
        line = ret_mat @ homo_xs1.T
        m = np.linalg.norm(line[:2].T, axis=1)
        v = np.sum(homo_xs2 * line.T, axis=1)
        valid = m != 0
        dist = np.abs(v[valid] / m[valid])
        inlier_mask = dist <= threshold
        curr_ratio = np.count_nonzeros(inlier_mask) / num_pairs

        num_iter += 1
    return ret_mat, inlier_mask

    