from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2




def calc_distance_from_point_to_line(pts, line_pts1, line_pts2):
    line_vec = line_pts2 - line_pts1
    
    pt_vec = pts - line_pts1
    
    pt_vec = pt_vec / np.linalg.norm(pt_vec, axis=1)[:,None]
    
    line_vec = line_vec / np.linalg.norm(line_vec, axis=1)[:,None]
    
    dot = pt_vec @ line_vec.T
    
    angle = np.arccos(dot)
    
    return np.sin(angle) * np.linalg.norm(pts - line_pts1, axis=1)


def opt_3d_points(params, gt_pixels, intrinsics, extrinsics):
    noisy_pts = params.reshape(-1,3)
    
    noisy_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        noisy_pixels.append(project(noisy_pts, intr, extr))
    
    noisy_pixels = np.concatenate(noisy_pixels, axis=0)
    
    error = (noisy_pixels - gt_pixels).ravel()
    
    return error


def project(pts:np.ndarray, intrinsic:np.ndarray, extrinsic:np.ndarray):
    fx,fy,cx,cy,k1,k2 = intrinsic
    rvec,tvec = extrinsic[:3],extrinsic[3:]
    
    Rmat = R.from_rotvec(rvec).as_matrix()
    
    pts = pts @ Rmat.T + tvec
    
    pts = pts[:,:2] / pts[:,2:]
    
    r2 = pts[:,0]**2 + pts[:,1]**2
    
    pts = pts * (1 + k1*r2 + k2*r2**2)[:,None]
    
    pts[:,0] = pts[:,0] * fx + cx
    pts[:,1] = pts[:,1] * fy + cy

    return pts


random_pts = np.random.rand(10,3) * 3

intrinsic1 = np.array([1000.,1000,500,500,0.1,0.1])
extrinsic1 = np.array([0.,0.,0.,0.,0.,0.])

intrinsic2 = np.array([1000.,1000,500,500,0.1,0.1])
extrinsic2 = np.array([0.,0.,0.,0.,0.,0.2])

intrinsic3 = np.array([1000.,1000,500,500,0.1,0.1])
extrinsic3 = np.array([0.,0.,0.,0.,0.,0.4])

intrs = [intrinsic1, intrinsic2, intrinsic3]
extrs = [extrinsic1, extrinsic2, extrinsic3]

gt_pixels_01 = project(random_pts, intrinsic1, extrinsic1)
gt_pixels_02 = project(random_pts, intrinsic2, extrinsic2)
gt_pixels_03 = project(random_pts, intrinsic3, extrinsic3)

gts = np.concatenate([gt_pixels_01, gt_pixels_02, gt_pixels_03], axis=0)

# Noisy points
noisy_pts = random_pts + np.random.rand(10,3) * 0.1

# Optimize 3d points
params = noisy_pts.ravel()

res = least_squares(opt_3d_points, params, args=(gts, intrs, extrs))


print("Ground truth")
print(random_pts)
print("Noisy")
print(noisy_pts)
print("Optimized")
print(res.x.reshape(-1,3))
print(f"Error: {res.cost:.4f}")