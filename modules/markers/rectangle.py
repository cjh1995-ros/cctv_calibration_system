from dataclasses import dataclass, field
from typing import List, ClassVar
from .point import Feature3D
import numpy as np






@dataclass
class Rect3D:
    """
    Represents a 3D rectangle.
        0 ------ 3 --------> y_axis
        |        |
        |        |
        1 ------ 2
        |
        |
        v
        x_axis
    Args:
        _id (int): The ID of the rectangle in set.
        _camera_id (List[int]): The list of camera IDs associated with the rectangle.
        _corners (List[Feature3D]): The list of 3D feature points representing the corners of the rectangle.
    """
    _rect_id: int = -1
    _camera_id: List[int] = field(init=False)
    _corners: List[Feature3D] = field(init=False)

    def noisy(self, sigma: float):
        """
        Add noise to the rectangle.

        Args:
            sigma (float): The standard deviation of the noise.
        """
        for corner in self.corners:
            corner.noisy(sigma)

    @property
    def rect_id(self):
        """
        Get the ID of the rectangle.

        Returns:
            int: The ID of the rectangle.
        """
        return self._rect_id
    
    @property
    def camera_id(self):
        """
        Get the list of camera IDs associated with the rectangle.

        Returns:
            List[int]: The list of camera IDs.
        """
        return self._camera_id
    
    @property
    def corners(self):
        """
        Get the list of 3D feature points representing the corners of the rectangle.

        Returns:
            List[Feature3D]: The list of corners.
        """
        return self._corners
    
    @rect_id.setter
    def rect_id(self, id: int):
        """
        Set the ID of the rectangle.

        Args:
            id (int): The ID to set.
        """
        self._rect_id = id
        
    @camera_id.setter
    def camera_id(self, camera_id: List[int]):
        """
        Set the list of camera IDs associated with the rectangle.

        Args:
            camera_id (List[int]): The list of camera IDs to set.
        """
        self._camera_id = camera_id
        
    @corners.setter
    def corners(self, corners: List[Feature3D]):
        """
        Set the list of 3D feature points representing the corners of the rectangle.

        Args:
            corners (List[Feature3D]): The list of corners to set.
        """
        self._corners = corners

    def is_plane(self):
        """
        Check if the rectangle is a plane.
        Simple way to check if the rectangle is a plane is to check if the 4 corners are coplanar.
        Returns:
            bool: True if the rectangle is a plane, False otherwise.
        """
        vec1 = self.corners[1] - self.corners[0]
        vec2 = self.corners[2] - self.corners[0]
        vec3 = self.corners[3] - self.corners[0]
        return np.isclose(vec1 @ (vec2.cross(vec3)), 0.0, atol=1e-6)

    def is_rect(self) -> bool:
        """
        Check if the rectangle is rect.
        Returns:
            bool: True if the rectangle is a plane. False otherwise.
        """
        # Check the corners are in same plane
        if not self.is_plane(): return False
        
        # Check the vectors are orthogonal
        vec10 = self.corners[1] - self.corners[0]
        vec30 = self.corners[3] - self.corners[0]
        
        val0 = vec30 @ vec10
        
        vec12 = self.corners[1] - self.corners[2]
        vec32 = self.corners[3] - self.corners[2]

        val1 = vec32 @ vec12
        
        if np.isclose(val0, 0, atol=1e-6) and np.isclose(val1, 0, atol=1e-6):
            return True

        return False
    
    def to_npy(self):
        """
        Get the numpy array representation of the rectangle.

        Returns:
            np.ndarray: The numpy array representation of the rectangle.
        """
        return np.array([corner.to_npy() for corner in self.corners], dtype=np.float32)


"""
CROSS_CHECK_DICT =
{
    cross_id: {train_center: [train_idx1, train_idx2, train_idx3]}
}
Train idx list will be used in setting corners again 
"""
CROSS_CHECK_CENTER = {
    0: 2,
    1: 3,
    2: 0, 
    3: 1
}
CROSS_CHECK_TRAIN_IDX = {
    0: [1, 2, 3],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [0, 1, 2]
}


@dataclass
class RoadMarker(Rect3D):
    """
    Represents a road marker that is a cross-check rectangle.

    Args:
        _is_opt (bool): Indicates if the road marker is optimized.
        _cross_check (bool): Indicates if the road marker is a cross-check rectangle.
        _valid_id (int): The ID of the valid marker.
        _train_markers (List[Feature3D]): The list of 3D feature points representing the train markers.
        _valid_marker (Feature3D): The 3D feature point representing the valid marker.
    """
    _is_opt:                bool = False
    _cross_check:           bool = False # if 
    _valid_id:              int = -1
    _train_markers:         List[Feature3D] = field(init=False)
    CROSS_CHECK_CENTER:     ClassVar = CROSS_CHECK_CENTER
    CROSS_CHECK_TRAIN_IDX:  ClassVar = CROSS_CHECK_TRAIN_IDX
    
    
    @property
    def is_opt(self):
        """
        Get the optimization status of the road marker.

        Returns:
            bool: True if the road marker is optimized, False otherwise.
        """
        return self._is_opt
    
    @property
    def cross_check(self):
        """
        Get the cross-check status of the road marker.

        Returns:
            bool: True if the road marker is a cross-check rectangle, False otherwise.
        """
        return self._cross_check
    
    @property
    def valid_id(self):
        """
        Get the ID of the valid marker.

        Returns:
            int: The ID of the valid marker.
        """
        return self._valid_id
    
    @property
    def train_markers(self):
        """
        Get the list of 3D feature points representing the train markers.

        Returns:
            List[Feature3D]: The list of train markers.
        """        
        return self._train_markers
        
    @is_opt.setter
    def is_opt(self, is_opt: bool):
        """
        Set the optimization status of the road marker.

        Args:
            is_opt (bool): The optimization status to set.
        """
        self._is_opt = is_opt
        
    @cross_check.setter
    def cross_check(self, cross_check: bool):
        """
        Set the cross-check status of the road marker.

        Args:
            cross_check (bool): The cross-check status to set.
        """
        self._cross_check = cross_check
    
    @valid_id.setter
    def valid_id(self, valid_id: int):
        """
        Set 3 things:
            ID of the valid marker
            Train markers
        Args:
            valid_id (int): The ID to set.
        """
        assert valid_id in range(4), "This is not good validation ID"
        self._valid_id = valid_id
        
        # init train marker in train markers
        self._train_markers = [corner for idx, corner in enumerate(self.corners) \
                                if idx != self._valid_id]
        
    @train_markers.setter
    def train_markers(self, train_markers: List[Feature3D]):
        """
        Set the list of 3D feature points representing the train markers.

        Args:
            train_markers (List[Feature3D]): The list of train markers to set.
        """
        # set train markers
        self._train_markers = train_markers
        
        # set corners
        train_idx = RoadMarker.CROSS_CHECK_TRAIN_IDX[self._valid_id]
        
        for idx, train_idx in enumerate(train_idx):
            self._corners[train_idx] = self._train_markers[idx]
        
        
    def estimate_valid_marker(self) -> Feature3D:
        """
        Get the 3D feature point representing the valid marker.
        Returns:
            Feature3D: The valid marker.
        """
        cross_id = RoadMarker.CROSS_CHECK_CENTER[self._valid_id]

        vecs = [self.corners[idx] - self.corners[cross_id] for idx in range(4)
                    if idx != cross_id and idx != self._valid_id]
        
        self.valid_marker = self.corners[cross_id] + vecs[0] + vecs[1]

        return self._valid_marker
    
    
    def to_optimize(self):
        """
        Only train markers will be optimized. return (3, 3)
        """
        return np.array([corner.to_npy() for corner in self.train_markers], dtype=np.float32)