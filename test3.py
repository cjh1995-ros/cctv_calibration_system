from autograd import numpy as np
from autograd import jacobian
from scipy.optimize import least_squares
from modules import from_axis_angle_to_matrix, from_matrix_to_axis_angle

import logging


"""
영향을 주는 파라미터

1. initial 값은 잘 줘야 하는가? 맞다면 initial value에 주는 noise도 기록해야함.
2. pixel에 주는 noise는 중요한가? 굉장히 중요함. noise가 작을 수록 GT에 가까이감. => 픽셀을 잘 찾는게 핵심임

# 멀티 카메라 케이스 -> f, cx,cy k1, k2 가 n_camera         --> 11 * n_camera + 3 * n_points (최적화 파라미터 수) / n_camera * point (residual 수)
camera = 5, point = 16 -> 55 + 48 / 5 * 16 -> 103 / 80 = 1.2875
# 단일 카메라 케이스 -> f, cx, cy, k1, k2, theta, pi, h     --> 8 + 2 * n_human (최적화 파라미터 수) / n_human (residual 수)
camera = 1, n_human = 100 -> 8 + 200 / 100 = 2.08


"""


def project(pts:np.ndarray, intrinsic:np.ndarray, extrinsic:np.ndarray):
    f,cx,cy,k1,k2 = intrinsic
    rvec,tvec = extrinsic[:3],extrinsic[3:]
    
    Rmat = from_axis_angle_to_matrix(rvec)
    
    new_pts = pts @ Rmat.T + tvec
    
    new_pts = new_pts / new_pts[:,2:]
    
    r2 = new_pts[:,0]**2 + new_pts[:,1]**2
    
    new_pts = new_pts * (1 + k1*r2 + k2*r2**2)[:,None]
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    new_pts = new_pts @ K.T
    
    return new_pts[:, :2]


def object_function(params, gt_pixels):
    camera_params = params[:44].reshape(-1, 11)
    intrinsics = camera_params[:, :5]
    extrinsics = camera_params[:, 5:]
    
    points = params[44:].reshape(-1, 3)    
        
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error

if __name__ == '__main__':
    logging.basicConfig(filename='log/myapp.log', level=logging.INFO)
    
    use_autograd = False
    pixel_noise_scale = 1
    max_itr = 100
    
    if use_autograd:
        logging.info("Using Autograd")
    else:
        logging.info("Using scipy only")
    
    # Read npy files
    camera_params = np.load('data/cameras.npy') # 5 x (3 + 2 + 6)
    camera_params = camera_params[:-1] # 마지막 제외
    points = np.load('data/points_case1.npy')   # 16 x 3
    
    for i, cam_param in enumerate(camera_params):
        logging.info(f"ground truth camera {i+1} params: {cam_param.tolist()}")
    
    # Generate ground truth pixels
    gt_pixels = []
    
    for cam_param in camera_params:
        intr = cam_param[:5]
        extr = cam_param[5:]
        gt_pixels.append(project(points, intr, extr))
        
    gt_pixels = np.concatenate(gt_pixels, axis=0)
    
    # Add noise to pixels
    noise = np.random.normal(0, pixel_noise_scale, gt_pixels.shape)
    noisy_pixels = gt_pixels + noise
    
    logging.info(f"gt_pixels: {gt_pixels.shape}")
    logging.info(f"gt pixels mean and sigma: {np.mean(gt_pixels):.4f}, {np.std(gt_pixels):.4f}")
    logging.info(f"mean error between gt and noisy: {np.linalg.norm(gt_pixels - noisy_pixels):.4f}")
    
    # Add noise to camera parameters
    noise = np.random.normal(0, 1, camera_params.shape)
    noise[:, :3] *= 200
    noise[:, 3:5] *= 0.01
    noise[:, 5:] *= 0.01
    noisy_camera_params = camera_params + noise

    for i, cam_param in enumerate(noisy_camera_params):
        logging.info(f"noisy truth camera {i+1} params: {cam_param.tolist()}")
    
    # Add noise to points
    noise = np.random.normal(0, 1, points.shape)
    noise *= 0.1
    noisy_points = points + noise
    
    grad_func = jacobian(object_function) if use_autograd else '2-point'
    
    initial_params = np.concatenate([noisy_camera_params.ravel(), noisy_points.ravel()])
    
    res = least_squares(object_function, 
                        initial_params, 
                        args=(noisy_pixels,), 
                        jac=grad_func, 
                        loss='linear',
                        f_scale=1.0,
                        verbose=0, 
                        max_nfev=100)
    
    # Compare results with ground truth
    opted_camera_params = res.x[:44].reshape(-1, 11)
    opted_points = res.x[44:].reshape(-1, 3)

    # logging.info(f"opted points: {opted_points.tolist()}")
    
    opted_pixels = []
    
    for i, cam_param in enumerate(opted_camera_params):
        logging.info(f"opted camera {i+1} params: {cam_param.tolist()}")
        # intr = cam_param[:5]
        # extr = cam_param[5:]
        # opted_pixels.append(project(opted_points, intr, extr))
        
    # opted_pixels = np.concatenate(opted_pixels, axis=0)
    
    # logging.info(f"mean error between gt and opted: {np.linalg.norm(gt_pixels - opted_pixels):.4f}")
    # logging.info(f"Cost function value: {res.cost:.4f}")