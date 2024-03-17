import argparse
import numpy as np
from pathlib import Path
import cv2
import itertools
from tqdm import tqdm

from feature_process import LightGlueInfer, SuperPointInfer
# from pyrecon.utils import rbd
from two_view_geometry import (
    estimate_fundamental_matrix, 
    recover_pose_from_F,
    linear_triangulation,
    homo_points,
)
from optimization import least_square_opt
from geometry import reproj_points, linear_pnp
from reconstructor import (
    Camera,
    Image,
    TwoViewGeometry,
    Reconstructor,
)
from viz import viz_match, viz_epipolar_line, show_points_3d, vis_reproj


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, help="Path to dataset directory.")
parser.add_argument("--model-dir", type=str, help="Path to model weights directory.")
args = parser.parse_args()

data_dir = Path(args.data_dir) 
image_dir = data_dir / "images"

img_paths = [i for i in image_dir.glob("*png")]
matches_files = [i for i in data_dir.glob("*txt")]
with open(data_dir/"calibration.txt", "r") as f:
    K = np.array([[float(val) for val in line.strip("\n").split(" ")] for line in f.readlines()])
cameras = dict()
images = dict()
two_view_geometries = dict()

cameras[0] = Camera(0, K)

model_dir = args.model_dir
superpoint_cfg = {
    'descriptor_dim': 256,
    'nms_radius': 4,
    'max_num_keypoints': 2048,
    'detection_threshold': 0.0005,
    'remove_borders': 4,
    'weights': f'{model_dir}/superpoint_v1.pth'
}
extractor = SuperPointInfer(superpoint_cfg)

lightglue_cfg = {
    'features': 'superpoint',
    'weights': f'{model_dir}/superpoint_lightglue.pth'
}
matcher = LightGlueInfer(lightglue_cfg)

feats = []
# images = dict() 
# extract features
for idx, imgf in enumerate(img_paths):
    img = cv2.imread(str(imgf))
    feat = extractor(img)
    feats.append(feat)
    kps = feat["keypoints"].cpu().numpy()[0]
    images[idx] = Image(
        id_=idx,
        R=np.eye(3),
        t=np.zeros((3,1)),
        kps=kps,
        camera_id=0,
        name=imgf.name,
        points3d_idxs=np.full(len(kps), -1, np.int32)
    )

print("Run match between image.")
pairs = [(i, j) for i, j in itertools.permutations(range(len(feats)), 2)]
for p in tqdm(pairs):
    matches = matcher(feats[p[0]], feats[p[1]])['matches0'].cpu().numpy()
    # matches_pairs = [[prev_idx, curr_idx] for prev_idx, curr_idx in enumerate(matches) if curr_idx >= 0]
    mask = matches > -1
    mask_curr = matches[mask]
    mask_prev = np.argwhere(mask)[:, 0]
    two_view_geometries[p] = TwoViewGeometry(
        id_=p,
        matches=np.vstack([mask_prev, mask_curr]),
        inliers=np.ones_like(matches)
    )

reconstruct = Reconstructor(cameras, images, two_view_geometries)

for idx, tvg in enumerate(reconstruct.two_view_geometries.values()):
    if len(reconstruct.registered) == 0:
        im_id1, im_id2 = tvg.id_
        img1 = reconstruct.images[tvg.id_[0]]
        img2 = reconstruct.images[tvg.id_[1]]
        cam1 = reconstruct.cameras[img1.camera_id]
        cam2 = reconstruct.cameras[img2.camera_id]

        match_prev_kps = img1.kps[tvg.matches[0]]
        match_curr_kps = img2.kps[tvg.matches[1]]

        ret, R, t, mask = recover_pose_from_F(
            np.eye(3),
            cam1.K,
            cam2.K,
            match_prev_kps,
            match_curr_kps
        )
        if not ret:
            exit()

        mask = mask[:, 0].astype(np.bool_)
        tri_prev_kps = match_prev_kps[mask]
        tri_curr_kps = match_curr_kps[mask]
        P1 = np.concatenate([cam1.K, np.zeros((3, 1))], axis=1)
        P2 = cam2.K @ np.concatenate([R, t], axis=1)
        pts3d = linear_triangulation(
            P1,
            P2,
            homo_points(tri_prev_kps),
            homo_points(tri_curr_kps)
        )[:, :3]
        # points4d = cv2.triangulatePoints(
        #     P1,
        #     P2,
        #     tri_prev_kps.T,
        #     tri_curr_kps.T
        # )
        # pts3d = points4d[:3, :].T / points4d[3:4, :].T

        new_pts3d, errors = least_square_opt(
            pts3d,
            tri_prev_kps,
            tri_curr_kps,
            P1,
            P2,
        )
        viz_ = cv2.imread(str(image_dir/img2.name))
        viz = vis_reproj(
            viz_,
            new_pts3d,
            tri_curr_kps,
            cam2.K,
            R, 
            t
        )
        img2.R = R
        img2.t = t
        # img2.points3d_idxs = 
        reconstruct.registered += [im_id1, im_id2]
        reconstruct.points3d = {idx: pt for idx, pt in enumerate(new_pts3d)}
        reconstruct.num_pts = len(new_pts3d)

    
    

    # debug matches
    # viz_match(
    #     viz,
    #     prev_kps[mask_prev],
    #     curr_kps[mask_curr] + images[j].shape[1],
    #     inlier
    # )
    # viz = viz_epipolar_line(
    #     images[j].copy(),
    #     images[i].copy(),
    #     curr_kps[mask_curr],
    #     prev_kps[mask_prev],
    #     ret_val,
    #     mask,
    # )
    cv2.imshow("viz", viz)
    cv2.waitKey(0)

    