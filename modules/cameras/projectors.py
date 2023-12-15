"""
Projection will be done in the following steps:

"""
from autograd import numpy as np
from scipy.spatial.transform import Rotation



"""
Projection utils
Below summarizes are for explaining the projection process

1. World coordinate to camera coordinate
world2cam:      transform points from world to camera

2. Camera coordinate to normalized coordinate
Two cases from cam to norm plane

---------- Case 1 ----------
2-1. To normalized plane (x, y, 1)
cam2norm:       camera coordinate to normalized coordinate
    theta_u:    calculate theta from normalized coordinate

perspective:    perspective projection
equidistant:    equidistant projection

---------- Case 2 ----------
2-2. To unit sphere plane (x', y', z')
cam2sphere:     camera coordinate to sphere coordinate

single_sphere:      single sphere projection -> unified camera model
double_sphere:      double sphere camera model
triple_sphere:      double sphere camera model
--------------------

3. distorting points (still in norm plane)
polynomial:     polynomial distortion
fov:            fov distortion
equidistant:    equidistant distortion

4. normalized coordinate to pixel coordinate
norm2pixel:     normalized coordinate to pixel coordinate
"""

class BasicConvertor:
    """
    Basic convertor is for converting points and some utils.
    Converting:
        1. Convert from world coord to camera coord
        2. Convert from camera coord to normalized coord
        3. Convert from normalized coord to pixel coord
        4. Convert from pixel coord to normalized coord
    
    Utils:
        1. Rescale points
        2. Homogeneous points
        3. Dehomogeneous points
    """
    @staticmethod
    def world_to_camera(pts, R, t):
        Rmat = Rotation.from_rotvec(R).as_matrix()
        return pts @ Rmat.T + t

    @staticmethod
    def camera_to_world(pts, R, t):
        Rmat = Rotation.from_rotvec(R).as_matrix()
        inv_R = np.linalg.inv(Rmat)
        return pts @ inv_R.T - inv_R @ t

    @staticmethod
    def rescale(r, pts):
        return r[:, None] * pts
    
    @staticmethod
    def homogeneous(pts):
        return np.hstack(
            (pts, np.ones((len(pts), 1)))
            ) # 이거 확인 필요
    
    @staticmethod
    def dehomogeneous(pts):
        return pts[:, :2]
    
    @staticmethod
    def normalized_to_pixel(pts, K):
        return pts @ K.T

    @staticmethod
    def pixel_to_normalized(pts, K):
        return pts @ np.linalg.inv(K).T



class Projector:
    """
    Collection of projection functions.

    Returns:
        np.ndarray: ru values
    """
    @staticmethod
    def camera_to_normalized(pts):
        return pts / pts[:, 2][:, None]

    @staticmethod
    def theta_u(pts):
        return np.arctan(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    
    @staticmethod
    def perspective(pts, params):
        new_pts = Projector.camera_to_normalized(pts)
        return new_pts[:, :2], np.tan(Projector.theta_u(new_pts))
    
    @staticmethod
    def equidistant(pts, params):
        new_pts = Projector.camera_to_normalized(pts)
        
        ru = Projector.theta_u(new_pts)
        rp = np.tan(ru)
        
        return (ru / rp)[:, None] * new_pts[:, :2], ru

    @staticmethod
    def single_sphere(pts, params):
        """_summary_

        Args:
            pts (_type_): 3d point cloud
            params (_type_): params for UCM

        Returns:
            np.ndarray vector: ru
        """
        alpha   = params[0]
        d = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
        
        z = alpha * d + (1 - alpha) * pts[:, 2]
        
        new_pts = pts[:, :2] / z[:, None]
        
        return new_pts, np.sqrt(new_pts[:, 0] ** 2 + new_pts[:, 1] ** 2)
        
        
    @staticmethod
    def double_sphere(pts, params):
        alpha   = params[0]
        xi      = params[1]
        
        d1 = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
        d2 = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + (xi * d1 + pts[:, 2]) ** 2)
        
        z = alpha * d2 + (1 - alpha) * (xi * d1 + pts[:, 2])
        
        new_pts = pts[:, :2] / z[:, None]
        
        return new_pts, np.sqrt(new_pts[:, 0] ** 2 + new_pts[:, 1] ** 2)
        
    @staticmethod
    def triple_sphere(pts, params):
        alpha   = params[0]
        xi      = params[1]
        gamma   = params[2]

        d1 = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
        d2 = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + (xi * d1 + pts[:, 2]) ** 2)
        # d3 = 
        
        # return new_pts, np.sqrt(new_pts[:, 0] ** 2 + new_pts[:, 1] ** 2)

class Distorter:
    """
    Calculate rd from ru with distortion parameters.
    """
    @staticmethod
    def none(ru):
        return ru
        
    @staticmethod
    def polynomial(ru, k):
        return ru * (1 + k[0] * ru ** 2 + k[1] * ru ** 4)

    @staticmethod
    def fov(ru, w):
        return np.arctan(2 * ru * np.tan(w / 2)) / w
    
    @staticmethod
    def equidistant(ru):
        return np.arctan(ru)
    
    
#########################################################
"""
Unproject utils

In unproject case, we need to convert point from pixel plane to normalized plane or unit sphere.
So it will be different by projection type.

Unproject process is the reverse process of projection process.

1. pixel coordinate to normalized coordinate(distorted or not)  - Solve distortion equation
2. normalized coordinate to camera coordinate                   - Solve projection equation

For polynomial distortion case, we need to solve polynomial equation with gauss-newton method.
For fov and equidistant distortion, they are closed form. So we can solve them directly.
"""    
class UnProjector:    
    @staticmethod
    def perspective(ru):
        return np.arctan(ru)
    
    @staticmethod
    def equidistant(ru):
        return ru    





class UnDistorter:
    @staticmethod
    def polynomial(rd, k):
        assert len(k) == 2, "Polynomial distortion should have only 2 parameters"

        ru = rd
        init_diff = np.inf
        step = 0.1
        
        for i in range(100):
            diff = rd - ru * (1 + k[0] * ru ** 2 + k[1] * ru ** 4)
            
            if init_diff > np.abs(diff): step *= 1.2
            else: step *= -0.5
            
            if (np.abs(diff) < 1e-3): 
                break
            
            init_diff = np.abs(diff)
            ru -= step * ru * (1 + k[0] * ru ** 2 + k[1] * ru ** 4) / (1 + 3 * k[0] * ru ** 2 + 5 * k[1] * ru ** 4)
        
        return ru
        
        
    @staticmethod
    def fov(rd, w):
        return np.tan(rd * w) / (2 * np.tan(w / 2))
    
    @staticmethod
    def equidistant(rd):
        return np.tan(rd)
    