from autograd import numpy as np
from autograd import jacobian
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from modules import from_axis_angle_to_matrix, from_matrix_to_axis_angle




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
    # Read npy files
    camera_params = np.load('data/cameras.npy') # 5 x (3 + 2 + 6)
    camera_params = camera_params[:-1] # 마지막 제외
    points = np.load('data/points_case1.npy')   # 16 x 3
        
    # Generate ground truth pixels
    gt_pixels = []
    
    for cam_param in camera_params:
        intr = cam_param[:5]
        extr = cam_param[5:]
        gt_pixels.append(project(points, intr, extr))
        
    gt_pixels = np.concatenate(gt_pixels, axis=0)
    
    # Add noise to pixels
    noise = np.random.normal(0, 1, gt_pixels.shape)
    noisy_pixels = gt_pixels + noise
    
    # Add noise to camera parameters
    noise = np.random.normal(0, 1, camera_params.shape)
    noise[:, :3] *= 200
    noise[:, 3:5] *= 0.01
    noise[:, 5:] *= 0.01
    noisy_camera_params = camera_params + noise
        
    # Add noise to points
    noise = np.random.normal(0, 1, points.shape)
    noise *= 0.1
    noisy_points = points + noise
    
    grad_func = jacobian(object_function)
    
    initial_params = np.concatenate([noisy_camera_params.ravel(), noisy_points.ravel()])
    
    res = least_squares(object_function, initial_params, args=(gt_pixels,), jac=grad_func, verbose=0, max_nfev=100)
    # res = least_squares(object_function, initial_params, args=(gt_pixels,), verbose=0, max_nfev=100)
    
    # Compare results with ground truth
    # opted_camera_params = res.x[:55].reshape(-1, 11)
    # opted_points = res.x[55:].reshape(-1, 3)

    opted_camera_params = res.x[:44].reshape(-1, 11)
    opted_points = res.x[44:].reshape(-1, 3)
    
    print("Camera parameters")
    for i, (gt, noisy, opted) in enumerate(zip(camera_params, noisy_camera_params, opted_camera_params)):
        print("Camera {}".format(i))
        print("Ground truth: {}".format(gt))
        print("Noisy: {}".format(noisy))
        print("Optimized: {}".format(opted))
        print()
        
    print("Points Error")
    print(f"Ground truth: {np.linalg.norm(points - opted_points)}")
    # for i, (gt, opted) in enumerate(zip(points, opted_points)):
    #     print("Point {}".format(i))
    #     print("Ground truth: {}".format(gt))
    #     print("Optimized: {}".format(opted))
    #     print()