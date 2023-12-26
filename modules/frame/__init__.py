from modules.cameras import Camera
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np





# Frames with camera information.
type MonoCameraFrames = List[Camera, List[BasicFrame]]
type MultiCameraFrames = List[MonoCameraFrames]

@dataclass
class BasicFrame:
    path:           str
    idx:            int
    is_fisheye:     bool = field(init=False)
    shape:          tuple = field(init=False)
    data:           np.ndarray = field(init=False)
    corners:        np.ndarray = field(init=False)
    transform:      np.ndarray = field(init=False)
    
    
    def __post_init__(self):
        self.data = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.shape = self.data.shape

    def gen_bounds(self):
        """Generate bounds for optimization."""
        bounds = []
        
        center = self.transform
        
        diff = 1.0
        
        upper = center + diff
        lower = center - diff
        
        bounds.append((lower[0], upper[0]))
        bounds.append((lower[1], upper[1]))
        bounds.append((lower[2], upper[2]))

        bounds.append((lower[3], upper[3]))
        bounds.append((lower[4], upper[4]))
        bounds.append((lower[5], upper[5]))        
        
        return bounds




def read_mono_frames(intrinsic_type: str, project_type: str, distortion_type: str, data_path: str) -> MonoCameraFrames:
    camera = Camera(intrinsic_type, project_type, distortion_type)
    data_path = Path(data_path)
    
    image_paths = sorted(data_path.glob("*.png"))
    
    return [camera, [ BasicFrame(path=str(image_path), idx=idx) for idx, image_path in enumerate(image_paths) ]]



def is_good_frame(image: BasicFrame, min_ratio: float = 0.02, max_ratio: float = 0.3) -> bool:
    """
    Add the corners information into image if it is good.
    
    Standard of good image:
        - The board is in the image.
        - The convex hull of corners is not too small or big. (at least 10% and 90% of the image)
        - Corners are well distributed.
    """
    # Find the convex hull of the corners and check the area.
    hull = cv2.convexHull(image.corners)#.reshape(-1, 2)
    
    hull_area = cv2.contourArea(hull)
    
    # Check the ratio of area. if its too small or too large, return False.
    r = np.pi / 4 if image.is_fisheye else 1
        
    image_area = r * image.data.shape[0] * image.data.shape[1]
    
    ratio = hull_area / image_area
    
    if ratio < min_ratio or ratio > max_ratio: return False
    
    return True






def distribution_of_corners(images: List[BasicFrame]) -> np.ndarray:
    """_summary_
    Calculate the distribution of corners in the images.
    If the image is rectangle, then we will divide the image into grid x grid.
    
    Default:
        - grid shape = (10, 10)
        
    Args:
        images (List[BasicFrame]): _description_

    Returns:
        np.ndarray: _description_
    """
    
    width, height = images[0].shape
    
    grid_pattern = (10, 10)
    
    grid_size = (width // grid_pattern[0], height // grid_pattern[1])
    
    grid = np.zeros(grid_pattern, dtype=np.int32)
    
    for image in images:
        for corner in image.corners:
            grid_idx = (corner[0] // grid_size[0], corner[1] // grid_size[1])
            
            grid[grid_idx] += 1
    
    
def distribution_of_corners_in_fisheye(images: List[BasicFrame]) -> np.ndarray:
    """_summary_
    Divide the image with theta and radius.
    
    Default:
        - theta = 30
        - normed_radius = 0.1
        
    Args:
        images (List[BasicFrame]): _description_

    Returns:
        np.ndarray: _description_
    """

    
