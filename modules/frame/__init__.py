from modules.cameras import Camera
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np





# Frames with camera information.
type Frames = Dict[Camera, List[BasicFrame]]

@dataclass
class BasicFrame:
    path:       str
    idx:        int
    is_fisheye: bool = field(default=False)
    shape:      tuple = field(init=False)
    data:       np.ndarray = field(init=False)
    corners:    np.ndarray = field(init=False)
    pose:       np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.data = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.shape = self.data.shape




def read_frames(camera_type: Camera, data_path: str) -> Frames:
    data_path = Path(data_path)
    
    image_paths = sorted(data_path.glob("*.png"))
    
    return [ BasicFrame(path=str(image_path), idx=idx) for idx, image_path in enumerate(image_paths) ]



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

    


if __name__ == "__main__":
    db_path = "data/"
    
    frames = read_frames(db_path)
    
    print(len(frames))
    print(frames[0].shape)