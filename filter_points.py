import numpy as np


def filter_points(pts3d, kps, R, t):
    cam_pts = pts3d @ R.T + t.T

    
def _filter_by_angle():
    pass 
    