from typing import List
from scipy.optimize import least_squares
from modules.cameras import Camera
from modules.generator.generator import CameraGenerator, MarkerGenerator
from modules.gui.viz_matplotlib import MatplotVisualizer
from copy import deepcopy
import numpy as np




def calc_z(theta, pts, hinz):
    # To xy plane
    pts_xy = deepcopy(pts)[:, :2]
    hinz_xy = deepcopy(hinz)[:2] # y = 2
    
    # Distance from line to point. line is hinz. which is y=2
    line_pts1 = np.array([2.4, 2.0])
    pt_vec = pts_xy - line_pts1
    
    pt_vec = pt_vec / np.linalg.norm(pt_vec, axis=1)[: ,None]
    line_vec = hinz_xy / np.linalg.norm(hinz_xy)
    
    dot = pt_vec @ line_vec.T
    
    angle = np.arccos(dot)
    
    return np.sin(angle) * np.linalg.norm(pts_xy - line_pts1, axis=1) * np.tan(theta)


def theta_BA(params: np.ndarray, cameras:List[Camera], projections: np.ndarray, xy: np.ndarray, hinz: np.ndarray):
    n_cameras = len(cameras)
    n_params_per_camera = len(cameras[0].params)

    # Set camera params
    camera_params = params[: n_cameras * n_params_per_camera].reshape((n_cameras, n_params_per_camera))
    for i, camera in enumerate(cameras):
        camera.params = camera_params[i]

    theta = params[n_cameras * n_params_per_camera:]

    # Prepare XYZ points
    zeros = np.zeros_like(xy[:, 0]).reshape(-1, 1)
    xy0 = np.hstack([xy, zeros])
        
    zs = calc_z(theta, xy0[8:], hinz)
    
    xy0[8:, 2] += zs

    # Get projections
    proj = []
    
    for cam in cameras:
        proj.append(cam.project(xy0))
        
    proj = np.array(proj).reshape(-1, 2)
    
    # Compute error
    error = (proj - projections).ravel()

    return error


def only_z_BA(params: np.ndarray, cameras: List[Camera], projections: np.ndarray, xy: np.ndarray):
    n_cameras = len(cameras)
    n_params_per_camera = len(cameras[0].params)

    # Set camera params
    camera_params = params[: n_cameras * n_params_per_camera].reshape((n_cameras, n_params_per_camera))
    for i, camera in enumerate(cameras):
        camera.params = camera_params[i]

    # Set point params
    zs = params[n_cameras * n_params_per_camera:].reshape((-1, 1))
    points = np.hstack([xy, zs])
    
    # Get projections
    proj = []
    
    for cam in cameras:
        proj.append(cam.project(points))
        
    proj = np.array(proj).reshape(-1, 2)
    
    # Compute error
    error = (proj - projections).ravel()
    
    return error



def simple_BA(params: np.ndarray, cameras: List[Camera], projections: np.ndarray):
    """Optimize camera params and points"""    
    n_cameras = len(cameras)
    n_params_per_camera = len(cameras[0].params)

    camera_params = params[: n_cameras * n_params_per_camera].reshape((n_cameras, n_params_per_camera))
    points = params[n_cameras * n_params_per_camera:].reshape((-1, 3))
    
    # Set camera params
    for i, camera in enumerate(cameras):
        camera.params = camera_params[i]

    # Get projections
    proj = []
    
    for cam in cameras:
        proj.append(cam.project(points))
        
    proj = np.array(proj).reshape(-1, 2)
    
    # Compute error
    error = (proj - projections).ravel()
    
    return error



def z_from_theta(theta: float, xy: np.ndarray):
    ...

 
if __name__ == '__main__':
    cameras = CameraGenerator().generate_default()
    markers, hinz = MarkerGenerator().generate_default()

    markers = np.array(markers).reshape(-1, 3)

    xy_sigma = 0.05
    z_sigmas = np.linspace(0.1, 1, 10)
    
    PLANE_MODELS = ["THETA", "Z", "XYZ"]
    
    ### Create gt pixels for reference
    gt_pixels = []
    
    for cam in cameras:
        gt_pixels.append(cam.project(markers))
    
    gt_pixels = np.array(gt_pixels).reshape(-1, 2)
    # Save Initial state
    save_cameras = deepcopy(cameras)
    
    for plane_model in PLANE_MODELS:
        for z_sigma in z_sigmas:
            # Reset cameras
            cameras = deepcopy(save_cameras)
            
            ### Create noisy dataset
            for cam in cameras:
                cam.params[0] += np.random.normal(0, 10)
                cam.params[1:3] += np.random.normal(0, 1, 2)
                cam.params[3:] += np.random.normal(0, 0.05, 6)

            # Give noisy xy and z separately. Give bigger noise to z
            noisy_markers = np.copy(markers)
            noisy_markers[:, :2] += np.random.normal(0, xy_sigma, markers[:, :2].shape)
            noisy_markers[:, 2] += np.random.normal(0, z_sigma, markers[:, 2].shape)
            
            # Init params
            if plane_model == "THETA":
                theta = np.pi/4 # gt theta = np.pi / 4
                obj_function = theta_BA
                inits = np.hstack([cam.params for cam in cameras] + [theta])

                u_bounds = []
                l_bounds = []
                
                for cam in cameras:
                    base = np.copy(cam.params)
                    up = np.copy(base)
                    down = np.copy(base)
                    
                    up[0] += 100
                    up[1:3] = np.array([0.0, 0.5])
                    up[3:] += 0.1
                    
                    down[0] -= 100
                    down[1:3] -= np.array([-0.5, 0.0])
                    down[3:] -= 0.1
                    
                    u_bounds.append(up)
                    l_bounds.append(down)
                
                u_bounds = np.hstack(u_bounds + [np.pi / 3])
                l_bounds = np.hstack(l_bounds + [0.0])
                
                bound = (l_bounds, u_bounds)
                
                args = (cameras, gt_pixels, noisy_markers[:, :2], hinz)
                
            elif plane_model == "Z":
                obj_function = only_z_BA
                inits = np.hstack([cam.params for cam in cameras] + [noisy_markers[:, 2]])
                
                u_bounds = []
                l_bounds = []
                
                for cam in cameras:
                    base = np.copy(cam.params)
                    up = np.copy(base)
                    down = np.copy(base)
                    
                    up[0] += 100
                    up[1:3] = np.array([0.0, 0.5])
                    up[3:] += 0.1
                    
                    down[0] -= 100
                    down[1:3] -= np.array([-0.5, 0.0])
                    down[3:] -= 0.1
                    
                    u_bounds.append(up)
                    l_bounds.append(down)


                u_bounds = np.hstack(u_bounds + [np.pi / 3])
                l_bounds = np.hstack(l_bounds + [0.0])
                
                bound = (l_bounds, u_bounds)
                
                
                args = (cameras, gt_pixels, noisy_markers[:, :2])
                
            elif plane_model == "XYZ":
                obj_function = simple_BA
                inits = np.hstack([cam.params for cam in cameras] + [noisy_markers.ravel()])
                args = (cameras, gt_pixels)
            
            # Do simple ba
            res = least_squares(obj_function, inits, args=args)
            
            # Set camera params
            n_cameras = len(cameras)
            n_params_per_camera = len(cameras[0].params)

            camera_params = res.x[: n_cameras * n_params_per_camera].reshape((n_cameras, n_params_per_camera))


            # Review the results
            print("End of optimization")
            print(f"Plane Model - Z Sigma: {plane_model} - {z_sigma:.2f}")
            
            for i, camera in enumerate(cameras):
                camera.params = camera_params[i]
                print(f"Camera {i+1} Parameters")
                print(f"Focal Length: Opted = {camera.params[0]}, GT = {save_cameras[i].params[0]}")
                print(f"Distortion Coeff: Opted = {camera.params[1: 3]}, GT = {save_cameras[i].params[1: 3]}")
            
            # Set point params
            if plane_model == "THETA":
                theta = res.x[n_cameras * n_params_per_camera:]
                print("theta: ", theta)
                points = deepcopy(noisy_markers)
                points[:, 2] = 0.0
                
                zs = calc_z(theta, points[8:], hinz)
                points[8:, 2] += zs
                
            elif plane_model == "Z":
                zs = res.x[n_cameras * n_params_per_camera:].reshape((-1, 1))
                points = np.hstack([noisy_markers[:, :2], zs])
                
            elif plane_model == "XYZ":
                points = res.x[n_cameras * n_params_per_camera:].reshape((-1, 3))
            
            
            # Calculate reprojection error
            
            
            # Error between gt and estimated points
            print("Error between gt and estimated points")
            print("Point Error: ", np.mean(np.abs(points - markers)))
            print("RMSE of Pixels ", res.cost / len(gt_pixels))