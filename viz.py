import cv2
import numpy as np
import matplotlib.pyplot as plt

from geometry import reproj_points


def show_points_3d(pts3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], marker="o", s=0.1)
    plt.show()


def show_points(pts):
    plt.scatter(pts[:, 0], pts[:, 1], marker="*", s=0.01)
    plt.show()
    
    
def viz_match(left_img, right_img, left_kps, right_kps, mask=None):
    viz = np.concatenate([left_img, right_img], axis=1)
    right_kps += right_img.shape[1]
    for idx, (lpt, rpt) in enumerate(zip(left_kps, right_kps)):
        cv2.circle(viz, lpt, 3, (0, 0, 255), thickness=-1)
        cv2.circle(viz, rpt, 3, (0, 0, 255), thickness=-1)

        color = (0, 255, 0) if mask is not None and mask[idx] else (255, 0, 0)
        cv2.line(viz, lpt, rpt, color, thickness=1)
    return viz


def viz_epipolar_line(left_img, right_img, left_kps, right_kps, F, mask, random_select=10):
    left_kps = np.concatenate([left_kps, np.ones((len(left_kps), 1))], axis=1)
    right_kps = np.concatenate([right_kps, np.ones((len(right_kps), 1))], axis=1)
    idxs = np.argwhere(mask)[:, 0]
    idxs = np.random.choice(idxs, random_select, replace=False)
    left_ep_lines = right_kps[idxs] @ F.T
    right_ep_lines = left_kps[idxs] @ F

    lh, lw = left_img.shape[:2]
    rh, rw = right_img.shape[:2]
    
    for ll, rl in zip(left_ep_lines, right_ep_lines):
        random_color = np.random.randint(256, size=3).tolist()
        cv2.line(
            left_img, 
            (0, int(-ll[2] / ll[1])), 
            (lw, int(-(ll[0]*lw + ll[2])/ ll[1])), 
            random_color, 
            thickness=1
        )
        cv2.line(
            right_img, 
            (0, int(-rl[2] / rl[1])), 
            (rw, int(-(rl[0]*rw + rl[2])/ rl[1])), 
            random_color, 
            thickness=1
        )

    viz = np.concatenate([left_img, right_img], axis=1)
    return viz
    # left_pts1 = np.zeros((len(left_ep_line), 2))
    # valid = left_ep_line[:, 1] != 0 
    # left_pts1[valid, 1] = - left_ep_line[valid, 2] / left_ep_line[valid, 1]
    # left_pts2 = np.ones_like(left_pts1) * lw
    # left_pts2[valid, 1] = -(left_ep_line[valid, 0] * lw + left_ep_line[valid, 2]) / left_ep_line[valid, 1]
    # if not np.all(valid):
    #     left_pts1[~valid, 0] = -
    #     left_pts1[]


def vis_reproj(img, pts3d, kps, K, R, t):
    uvs, d, m = reproj_points(pts3d, K, R, t)
    mean_err = np.mean(np.linalg.norm(uvs - kps, axis=1))
    print(f"mean error: {mean_err}")
    # viz = images[j]
    for euv, uv in zip(uvs, kps):
        cv2.circle(img, euv.astype(np.int32), 3, (255, 0, 0), thickness=-1)
        cv2.circle(img, uv.astype(np.int32), 2, (0, 0, 255), thickness=-1)
    return img    
    