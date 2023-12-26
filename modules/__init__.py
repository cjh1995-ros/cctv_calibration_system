from scipy.spatial.transform import Rotation as R
# import numpy as np
from autograd import numpy as np




def from_axis_angle_to_matrix(rvec: np.ndarray):
    theta = np.linalg.norm(rvec)
    r = rvec / theta
    
    r_x, r_y, r_z = r
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    R = np.array([
        [c + r_x**2*(1-c), r_x*r_y*(1-c) - r_z*s, r_x*r_z*(1-c) + r_y*s],
        [r_y*r_x*(1-c) + r_z*s, c + r_y**2*(1-c), r_y*r_z*(1-c) - r_x*s],
        [r_z*r_x*(1-c) - r_y*s, r_z*r_y*(1-c) + r_x*s, c + r_z**2*(1-c)]
    ], dtype=np.float64)
    
    return R

def from_matrix_to_axis_angle(R: np.ndarray):
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    r_x = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    r_y = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    r_z = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    
    rvec = np.array([r_x, r_y, r_z]) * theta
    
    return rvec

def conv_transform2pose(transform: np.ndarray) -> np.ndarray:
    """Convert transform matrix to pose vector.
    
    Args:
        transform (np.ndarray): Transform matrix of shape (4, 4).
    
    Returns:
        np.ndarray: Pose vector of shape (6,).
    """
    assert transform.shape == (6, )
    
    rvec = transform[:3]
    tvec = transform[3:]
    
    Rmat = from_axis_angle_to_matrix(rvec)
    
    ori = from_matrix_to_axis_angle(Rmat.T)
    pos = -Rmat @ tvec
    
    return np.concatenate([ori, pos])

def conv_pose2transform(pose: np.ndarray) -> np.ndarray:
    """Convert pose vector to transform matrix.
    
    Args:
        pose (np.ndarray): Pose vector of shape (6,).
    
    Returns:
        np.ndarray: Transform matrix of shape (4, 4).
    """
    assert pose.shape == (6, )
    
    ori = pose[:3]
    pos = pose[3:] # -Rmat.T @ tvec -> tvec = -Rmat @ pos
    
    ori_mat = from_axis_angle_to_matrix(ori)
    rvec = from_matrix_to_axis_angle(ori_mat.T)
    
    tvec = -ori_mat.T @ pos
    
    return np.concatenate([rvec, tvec])

