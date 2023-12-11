from scipy.optimize import least_squares
import numpy as np
from typing import List, Any
from modules.cameras import Camera
from modules.markers.rectangle import Rect3D
from modules.markers.point import Feature2D, Feature3D



class Optimizer:
    def __init__(self, ba_type: str = "CAM_AND_POINTS"):
        self.ba_type = ba_type
        

    def optimize_cameras(self, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        # Initialize params
        params = np.zeros(0, dtype=np.float32)
        
        # Concatenate camera params
        for camera in cameras:
            params = np.concatenate((params, camera.params))
            
        # Optimize
        res = least_squares(ObjectFunction.simple_BA, params, args=(cameras, points, projections))
        
        return res.x


    def optimize_points(self, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        # Initialize params
        params = np.zeros(0, dtype=np.float32)
                    
        # Concatenate point params
        for point in points:
            params = np.concatenate((params, point.xyz))
            
        # Optimize
        res = least_squares(ObjectFunction.simple_BA, params, args=(cameras, points, projections))
        
        return res.x

    
    
    def optimize_cameras_points(self, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        """
        Args:
            cameras (List[Camera]): List of cameras.
            points (List[Feature3D]): List of 3D points.
            projections (List[Feature2D]): List of projections.
        """
        # Initialize params
        params = np.zeros(0, dtype=np.float32)
        
        # Concatenate camera params
        for camera in cameras:
            params = np.concatenate((params, camera.params))
            
        # Concatenate point params
        for point in points:
            params = np.concatenate((params, point.xyz))
            
        # Optimize
        res = least_squares(ObjectFunction.simple_BA, params, args=(cameras, points, projections))
        
        return res.x
    

class ObjectFunction:
    """Solve a least squares problem using scipy.optimize.least_squares."""
    @staticmethod
    def cam_BA(params: np.ndarray, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        camera_params = params
        
        # Set camera params
        for i, camera in enumerate(cameras):
            camera.params = camera_params[i * camera.n_total:(i + 1) * camera.n_total]
        
        ### Compute error
        # Initialize params        
        projected = np.zeros((len(projections), 2), dtype=np.float32)
        n_pts_slicer = 0
        
        # Matching camera and 3d point
        for i, camera in enumerate(cameras):
            camera_id = camera.id
            
            matched_points = np.array([point.to_npy() for point in points if point.camera_id == camera_id])
            
            # Project points
            projected[n_pts_slicer: n_pts_slicer + len(matched_points)] = camera.project(matched_points)
            
            n_pts_slicer += len(matched_points)
            
        # Compute error
        error = (projected - projections).ravel()
        
        return error



    @staticmethod
    def point_BA(params: np.ndarray, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        # Extract point params
        point_params = params.reshape(-1, 3)
                    
        # Set point params
        for i, point in enumerate(points):
            point.xyz = point_params[i]
        
        ### Compute error
        # Initialize params
        projected = np.zeros((len(projections), 2), dtype=np.float32)
        n_pts_slicer = 0
        
        # Matching camera and 3d point
        for i, camera in enumerate(cameras):            
            matched_points = np.array([point.to_npy() for point in points if camera.id in point.camera_id])
            
            # Project points
            projected[n_pts_slicer: n_pts_slicer + len(matched_points)] = camera.project(matched_points)
            
            n_pts_slicer += len(matched_points)
            
        # Compute error
        error = (projected - projections).ravel()
        
        return error
    
    
    @staticmethod
    def simple_BA(params: np.ndarray, cameras: List[Camera], points: List[Feature3D], projections: List[Feature2D]):
        """
        Compute the sum of squared differences between projected and observed 2D points.
        Points are Feature3Ds.
        projections are np.ndarray of shape (n_points, 2).
        """
        n_camera_params = 0
        for camera in cameras:
            n_camera_params += camera.n_total
        
        # Extract camera params
        camera_params = params[:n_camera_params]
        
        # Extract point params
        point_params = params[n_camera_params:].reshape(-1, 3)
        
        # Set camera params
        for i, camera in enumerate(cameras):
            camera.params = camera_params[i * camera.n_total:(i + 1) * camera.n_total]
            
        # Set point params
        for i, point in enumerate(points):
            point.xyz = point_params[i]
        
        ### Compute error
        # Initialize params
        projected = np.zeros((len(projections), 2), dtype=np.float32)
        n_pts_slicer = 0
        
        # Matching camera and 3d point
        for i, camera in enumerate(cameras):
            camera_id = camera.id
            
            matched_points = np.array([point.to_npy() for point in points if point.camera_id == camera_id])
            
            # Project points
            projected[n_pts_slicer: n_pts_slicer + len(matched_points)] = camera.project(matched_points)
            
            n_pts_slicer += len(matched_points)
            
        # Compute error
        error = (projected - projections).ravel()
        
        return error
        
