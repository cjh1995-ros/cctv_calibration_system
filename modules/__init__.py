from scipy.spatial.transform import Rotation as R
import numpy as np


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
    
    Rmat = R.from_rotvec(rvec).as_matrix()
    
    ori = R.from_matrix(Rmat.T).as_rotvec()
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
    
    ori_mat = R.from_rotvec(ori).as_matrix() # R.T 
    rvec = R.from_matrix(ori_mat.T).as_rotvec()
    
    tvec = -ori_mat.T @ pos
    
    return np.concatenate([rvec, tvec])

