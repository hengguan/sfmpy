import numpy as np
from pathlib import Path
import cv2
import itertools

from pyrecon.models import LightGlueInfer, SuperPointInfer
from pyrecon.utils import rbd
from two_view_geometry import estimate_fundamental_matrix


data_dir = Path("/home/gh/workspace/data/P3Data") 
image_dir = data_dir / "images"

img_paths = [i for i in image_dir.glob("*png")]
matches_files = [i for i in data_dir.glob("*txt")]
with open(data_dir/"calibration.txt", "r") as f:
    K = np.array([[float(val) for val in line.strip("\n").split(" ")] for line in f.readlines()])

model_dir = "/home/gh/workspace/model_weights/"
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

print(K)
feats = []
images = []
# extract features
for imgf in img_paths:
    img = cv2.imread(str(imgf))
    feats.append(extractor(img))
    images.append(img)

for i, j in itertools.permutations(range(len(feats)), 2):
    print(i, j)
    matches = matcher(feats[i], feats[j])['matches0'].cpu().numpy()
    matches_pairs = [[prev_idx, curr_idx] for prev_idx, curr_idx in enumerate(matches) if curr_idx >= 0]
    mask = matches > -1
    mask_curr = matches[mask]
    mask_prev = np.argwhere(mask)[:, 0]
    print(mask.shape, mask_curr.shape)

    prev_kps = feats[i]['keypoints'].cpu().numpy()[0]
    curr_kps = feats[j]['keypoints'].cpu().numpy()[0]
    print(prev_kps.shape, curr_kps.shape)

    ret_mat, inlier = estimate_fundamental_matrix(
        prev_kps[mask_prev],
        curr_kps[mask_curr]
    )
    if ret_mat is None:
        continue

    # debug matches
    viz = np.concatenate([images[i], images[j]], axis=1)
    h, w = images[j].shape[:2]
    for idx, (pi, ci) in enumerate(matches_pairs):
        prev_pt = prev_kps[pi].astype(np.int32)
        curr_pt = curr_kps[ci].astype(np.int32)+[w, 0]
        cv2.circle(viz, prev_pt, 3, (0, 0, 255), thickness=-1)
        cv2.circle(viz, curr_pt, 3, (0, 0, 255), thickness=-1)
        if inlier[idx]:
            cv2.line(viz, prev_pt, curr_pt, (0, 255, 0), thickness=1)
        else:
            cv2.line(viz, prev_pt, curr_pt, (255, 0, 0), thickness=1)
    cv2.imshow("viz", viz)
    cv2.waitKey(0)

    