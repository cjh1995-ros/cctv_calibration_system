from autograd import numpy as np
from autograd import jacobian
from scipy.optimize import least_squares
from modules import from_axis_angle_to_matrix, from_matrix_to_axis_angle
import cv2
import logging


"""
영향을 주는 파라미터

1. initial 값은 잘 줘야 하는가? 맞다면 initial value에 주는 noise도 기록해야함.
2. pixel에 주는 noise는 중요한가? 굉장히 중요함. noise가 작을 수록 GT에 가까이감. => 픽셀을 잘 찾는게 핵심임

# 멀티 카메라 케이스 -> f, cx,cy k1, k2 가 n_camera         --> 11 * n_camera + 3 * n_points (최적화 파라미터 수) / n_camera * point (residual 수)
camera = 5, point = 16 -> 55 + 48 / 5 * 16 -> 103 / 80 = 1.2875
# 단일 카메라 케이스 -> f, cx, cy, k1, k2, theta, pi, h     --> 8 + 2 * n_human (최적화 파라미터 수) / n_human (residual 수)
camera = 1, n_human = 100 -> 8 + 200 / 100 = 2.08

중요한 점

1. 픽셀을 잘 찝어야 한다. 

- 멀티 카메라라서? 만약에 단일 카메라라면 pixel noise 1 은 괜찮지 않을까?
- 그러면 단일 카메라라고 가정하고 테스트 해보자.

"""


def project(pts:np.ndarray, intrinsic:np.ndarray, extrinsic:np.ndarray):
    f,cx,cy,k1,k2 = intrinsic
    rvec,tvec = extrinsic[:3],extrinsic[3:]
    
    Rmat = from_axis_angle_to_matrix(rvec)
    
    new_pts = pts @ Rmat.T + tvec
    
    new_pts = new_pts[:, :2] / new_pts[:,2:]
    
    r2 = new_pts[:, 0]**2 + new_pts[:, 1]**2
    
    new_pts = new_pts * (1 + k1*r2 + k2*r2**2)[:,None]
    
    new_pts = new_pts * f + np.array([cx, cy])
        
    return new_pts


def object_function(params, gt_pixels, is_mono_camera):
    if is_mono_camera:
        intr = params[:5]
        extrs = params[5: 5+6*4].reshape(-1, 6)
        points = params[11:].reshape(-1, 3)    

    else:
        camera_params = params[:44].reshape(-1, 11)
        points = params[44:].reshape(-1, 3)    
    
        intrinsics = camera_params[:, :5]
        extrinsics = camera_params[:, 5:]
        
    new_pixels = []
    
    if is_mono_camera:
        for extr in extrs:
            new_pixels.append(project(points, intr, extr))
        
    else:
        for intr, extr in zip(intrinsics, extrinsics):
            new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    error = (new_pixels - gt_pixels).ravel()
    
    return error

if __name__ == '__main__':
    logging.basicConfig(filename='log/fk1k2Rt.log', level=logging.INFO)
    
    use_autograd = True
    pixel_noise_scale = 0.01
    intr_noise_scale = 50
    distortion_noise_scale = 0.01
    extrinsic_noise_scale = 0.01
    points_noise_scale = 0.01
    max_itr = 1000
    loss = 'linear'
    f_scale = 1.0
    
    mono_camera = True
    
    if mono_camera:
        logging.info("Mono Camera Calibration")
    else:
        logging.info("Multi Camera Calibration")
    
    if use_autograd:
        logging.info("Using Autograd")
    else:
        logging.info("Using SciPy only")
    
    logging.info(f"Pixel Noise Scale: {pixel_noise_scale}")
    logging.info(f"Camera Matrix Noise Scale: {intr_noise_scale}")
    logging.info(f"Distortion Noise Scale: {distortion_noise_scale}")
    logging.info(f"Extrinsic Noise Scale: {extrinsic_noise_scale}")
    logging.info(f"Points Noise Scale: {points_noise_scale}")
    logging.info(f"Max Iteration: {max_itr}")
    logging.info(f"Loss Function: {loss}")
    logging.info(f"Scale Factor: {f_scale}")
    
    
    # Read npy files
    camera_params = np.load('data/cameras.npy') # 5 x (3 + 2 + 6)
    # camera_params = camera_params 
    camera_params = camera_params[:-1] # 마지막 제외
    points = np.load('data/points_case1.npy')   # 16 x 3
    
    for i, cam_param in enumerate(camera_params):
        logging.info(f"GT Camera {i+1} params: {cam_param.tolist()}")
    
    # Generate ground truth pixels
    gt_pixels = []
    
    for cam_param in camera_params:
        intr = cam_param[:5]
        extr = cam_param[5:]
        gt_pixels.append(project(points, intr, extr))
        
    gt_pixels = np.concatenate(gt_pixels, axis=0)
    
    # cv_K = np.array([[camera_params[0][0], 0, camera_params[0][1]], 
    #                  [0, camera_params[0][0], camera_params[0][2]], 
    #                  [0, 0, 1]])
    # cv_dist = np.array([camera_params[0][3], camera_params[0][4], 0, 0])
    
    # cv_pixels = []
    
    # for extr in camera_params[:, 5:]:
    
    #     cv_pixel, _ = cv2.projectPoints(points, extr[:3], extr[3:], cv_K, cv_dist)
    #     cv_pixels.append(cv_pixel)
    
    # cv_pixels = np.concatenate(cv_pixels, axis=0).reshape(-1, 2)
    
    # Add noise to pixels
    noise = np.random.normal(0, pixel_noise_scale, gt_pixels.shape)
    noisy_pixels = gt_pixels + noise
        
    # Add noise to camera parameters
    noise = np.random.normal(0, 1, camera_params.shape)
    noise[:, :3] *= intr_noise_scale
    noise[:, 3:5] *= distortion_noise_scale
    noise[:, 5:] *= extrinsic_noise_scale
    noisy_camera_params = camera_params + noise

    for i, cam_param in enumerate(noisy_camera_params):
        logging.info(f"Noisy Camera {i+1} params: {cam_param.tolist()}")
    
    # Add noise to points
    noise = np.random.normal(0, points_noise_scale, points.shape)
    noisy_points = points + noise
    
    grad_func = jacobian(object_function) if use_autograd else '2-point'
    
    if mono_camera:
        intr = noisy_camera_params[0][:5]
        extrs = []
        for extr in noisy_camera_params[0][5:]:
            extrs.append(extr)
        noisy_camera_params = np.concatenate([intr, extrs])
        initial_params = np.concatenate([noisy_camera_params.ravel(), noisy_points.ravel()])
    else:
        initial_params = np.concatenate([noisy_camera_params.ravel(), noisy_points.ravel()])
    
    res = least_squares(object_function, 
                        initial_params, 
                        args=(noisy_pixels, mono_camera), 
                        jac=grad_func, 
                        loss=loss,
                        f_scale=f_scale,
                        verbose=0, 
                        max_nfev=max_itr)
    
    # Compare results with ground truth
    if mono_camera:
        opted_intr = res.x[:5]
        extrs = res.x[5:5+6*4].reshape(-1, 6)
        opted_points = res.x[5+6*4:].reshape(-1, 3)
    else:        
        opted_camera_params = res.x[:44].reshape(-1, 11)
        opted_points = res.x[44:].reshape(-1, 3)
    
    opted_pixels = []
    
    if mono_camera:
        logging.info(f"Opted camera intrinsic params: {opted_intr.tolist()}")
        for extr in extrs:
            logging.info(f"Opted camera extrinsic params: {extr.tolist()}")
    else:
        for i, cam_param in enumerate(opted_camera_params):
            logging.info(f"Opted camera {i+1} params: {cam_param.tolist()}")

    logging.info(f"Total Iteration: {res.nfev}")
    logging.info(f"Mean Cost: {res.cost / (len(noisy_pixels) * 2)}")
    logging.info(f"Termination Status: {res.status}")
    logging.info("="*100) # new line