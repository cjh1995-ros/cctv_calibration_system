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



def sphere_projection(pts:np.ndarray, intrinsic:np.ndarray, extrinsic:np.ndarray):
    f, cx, cy, alpha = intrinsic
    rvec, tvec = extrinsic[:3], extrinsic[3:]
    
    Rmat = from_axis_angle_to_matrix(rvec)
    
    new_pts = pts @ Rmat.T + tvec
    
    d = np.sqrt(new_pts[:, 0] ** 2 + new_pts[:, 1] ** 2 + new_pts[:, 2] ** 2)
    
    z = alpha * d + (1 - alpha) * new_pts[:, 2]
    
    new_pts = new_pts[:, :2] / z[:, None]
    
    return new_pts * f + np.array([cx, cy])


def pp_xyz(params, gt_pixels, cxy):
    camera_params = params[:9 * 4].reshape(-1, 9)
    points = params[9 * 4:].reshape(-1, 3)    

    # f, k1, k2, R, t + cxy -> f, cx, cy, k1, k2, R, t
    intrinsics = np.zeros((4, 5), dtype=np.float64)
    extrinsics = np.zeros((4, 6), dtype=np.float64)

    for i, cam_param in enumerate(camera_params):
        f, k1, k2 = cam_param[:3]
        R = cam_param[3:6]
        t = cam_param[6:9]
        intrinsics[i] = [f, cxy[0], cxy[1], k1, k2]
        extrinsics[i] = np.concatenate([R, t])

    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    error = (new_pixels - gt_pixels).ravel()
    
    return error

def pp_z(params, gt_pixels, xy, cxy):
    camera_params = params[:9 * 4].reshape(-1, 9)
    z = params[9 * 4:]
    
    points = np.concatenate([xy, z[:, None]], axis=1)

    # f, k1, k2, R, t + cxy -> f, cx, cy, k1, k2, R, t
    intrinsics = np.zeros((4, 5), dtype=np.float64)
    extrinsics = np.zeros((4, 6), dtype=np.float64)

    for i, cam_param in enumerate(camera_params):
        f, k1, k2 = cam_param[:3]
        R = cam_param[3:6]
        t = cam_param[6:9]
        intrinsics[i] = [f, cxy[0], cxy[1], k1, k2]
        extrinsics[i] = np.concatenate([R, t])

    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    error = (new_pixels - gt_pixels).ravel()
    
    return error

def s_xyz(params, gt_pixels, cxy):
    camera_params = params[:8 * 4].reshape(-1, 8)
    points = params[8 * 4:].reshape(-1, 3)

    # f, alpha, R, t + cxy -> f, cx, cy, alpha, R, t
    intrinsics = np.zeros((4, 4), dtype=np.float64)
    extrinsics = np.zeros((4, 6), dtype=np.float64)

    for i, cam_param in enumerate(camera_params):
        f, alpha = cam_param[:2]
        R = cam_param[2:5]
        t = cam_param[5:8]
        intrinsics[i] = [f, cxy[0], cxy[1], alpha]
        extrinsics[i] = np.concatenate([R, t])

    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(sphere_projection(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    error = (new_pixels - gt_pixels).ravel()
    
    return error

obj_func_dict = {
    'PP-XYZ': pp_xyz,
    'PP-Z': pp_z,
    'S-XYZ': s_xyz
}

if __name__ == '__main__':
    projection_model = 'S-XYZ' # 'PP-XYZ  'PP-Z', 'S-XYZ'    
    
    use_autograd = False
    pixel_noise_scale = 0.1
    intr_noise_scale = 150
    distortion_noise_scale = 0.1
    extrinsic_noise_scale = 0.1
    points_noise_scale = 0.5
    max_itr = 1000
    loss = 'linear'
    f_scale = 1.0
    jac = '2-point'
    
    logging.basicConfig(filename=f'log/{projection_model}.log', level=logging.INFO)
    
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
    logging.info(f"Jacobian: {jac}")
    logging.info(f"Scale Factor: {f_scale}")
    
    
    # Read npy files
    camera_params = np.load('data/cameras.npy') # 5 x (3 + 2 + 6)
    # camera_params = camera_params[:, (0, 3, 4, 5, 6, 7, 8, 9,10)]
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
    
    # Add noise to pixels
    noise = np.random.normal(0, pixel_noise_scale, gt_pixels.shape)
    noisy_pixels = gt_pixels + noise
        
    # Add noise to camera parameters
    if projection_model != 'S-XYZ':
        noise = np.random.normal(0, 1, camera_params.shape)
        noise[:, :3] *= intr_noise_scale
        noise[:, 3:5] *= distortion_noise_scale
        noise[:, 5:] *= extrinsic_noise_scale
        noisy_camera_params = camera_params + noise
        noisy_camera_params = noisy_camera_params[:, (0, 3, 4, 5, 6, 7, 8, 9,10)]

    else :
        shape = (4, 8) # f, alpha, rvec, tvec
        
        noisy_camera_params = np.zeros(shape, dtype=np.float64)
        
        focal = 1000.
        alpha = 0.5
        
        for noisy, gt in zip(noisy_camera_params, camera_params):
            noisy[0] = focal + np.random.normal(0, intr_noise_scale)
            noisy[1] = alpha
            noisy[2:5] = gt[5:8] + np.random.normal(0, extrinsic_noise_scale, 3)
            noisy[5:] = gt[8:] + np.random.normal(0, extrinsic_noise_scale, 3)
        
    for i, cam_param in enumerate(noisy_camera_params):
        logging.info(f"Noisy Camera {i+1} params: {cam_param.tolist()}")
    
    # Add noise to points
    noise = np.random.normal(0, points_noise_scale, points.shape)
    noisy_points = points + noise
    noisy_xy = None
    
    if projection_model == 'PP-Z':
        noisy_points = noisy_points[:, 2]
        noisy_xy = points[:, :2]
    
    # grad_func = jacobian(object_function) if use_autograd else '2-point'
    grad_func = jac
    
    initial_params = np.concatenate([noisy_camera_params.ravel(), noisy_points.ravel()])
    
    cxy = np.array([500., 500.], dtype=np.float64)
    
    if projection_model == 'PP-XYZ':
        args = (noisy_pixels, cxy)
        
    elif projection_model == 'PP-Z':
        args = (noisy_pixels, noisy_xy, cxy)
        
    elif projection_model == 'S-XYZ':
        args = (noisy_pixels, cxy)
    
    res = least_squares(obj_func_dict[projection_model], 
                        initial_params, 
                        args=args, 
                        jac=grad_func, 
                        loss=loss,
                        f_scale=f_scale,
                        verbose=0, 
                        max_nfev=max_itr)
    
    # Compare results with ground truth
    if projection_model == 'PP-XYZ':
        opted_camera_params = res.x[:9 * 4].reshape(-1, 9)
        opted_points = res.x[9 * 4:].reshape(-1, 3)
        absolute_translation_error = np.mean(np.linalg.norm(opted_camera_params[:, 3:] - camera_params[:, 5:], axis=1))
        
    elif projection_model == 'PP-Z':
        opted_camera_params = res.x[:9 * 4].reshape(-1, 9)
        opted_points = np.concatenate([noisy_xy, res.x[9 * 4:][:, None]], axis=1)
        absolute_translation_error = np.mean(np.linalg.norm(opted_camera_params[:, 3:] - camera_params[:, 5:], axis=1))
    
    elif projection_model == 'S-XYZ':
        opted_camera_params = res.x[:8 * 4].reshape(-1, 8)
        opted_points = res.x[8 * 4:].reshape(-1, 3)
        absolute_translation_error = np.mean(np.linalg.norm(opted_camera_params[:, 2:] - camera_params[:, 5:], axis=1))
            
    for i, cam_param in enumerate(opted_camera_params):
        logging.info(f"Opted camera {i+1} params: {cam_param.tolist()}")

    mean_distance_pts = np.mean(np.linalg.norm(opted_points - points, axis=1))
    
    logging.info("="*100) # new line
    logging.info(f"Mean camera parameters : {np.mean(opted_camera_params, axis=0).tolist()}")
    logging.info(f"Absolute Translation Error: {absolute_translation_error}")
    logging.info(f"Mean Distance of Points: {mean_distance_pts}")
    logging.info(f"Total Iteration: {res.nfev}")
    logging.info(f"Mean Cost: {res.cost / (len(noisy_pixels) * 2)}")
    logging.info(f"Termination Status: {res.status}")
    logging.info("="*100) # new line