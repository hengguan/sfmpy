import numpy as np
from dataclasses import dataclass


@dataclass
class Camera:
    id_: int
    K: np.ndarray 


@dataclass
class Image:
    id_: int
    R: np.ndarray
    t: np.ndarray
    kps: np.ndarray
    camera_id: int
    name: str
    points3d_idxs: np.ndarray


@dataclass
class TwoViewGeometry:
    id_: tuple 
    matches: np.ndarray
    inliers: np.ndarray


class Reconstructor:
    cameras: dict
    images: dict
    two_view_geometries: dict

    def __init__(self, cameras, images, two_view_geometries):
        self.cameras = cameras
        self.images = images
        self.two_view_geometries = two_view_geometries

        self.registered = set()
        self.points3d = dict()
        self.num_pts: int = 0

    # def 
        