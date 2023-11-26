from dataclasses import dataclass, field
from typing import List
from point import Feature2D, Feature3D





@dataclass
class Rect3D:
    """
    Args:
        Vector3D (_type_): _description_
    """
    _id:            int = -1
    _camera_id:     List[int] = field(init=False)
    _corners:       List[Feature3D] = field(init=False)

    @property
    def id(self):
        return self._id
    
    @property
    def camera_id(self):
        return self._camera_id
    
    @property
    def corners(self):
        return self._corners
    
    @id.setter
    def id(self, id: int):
        self._id = id
        
    @camera_id.setter
    def camera_id(self, camera_id: List[int]):
        self._camera_id = camera_id
        
    @corners.setter
    def corners(self, corners: List[Feature3D]):
        self._corners = corners
        
@dataclass
class RoadMarker(Rect3D):
    """
        This class is for cross check rectangle marker.
        It has 4 markers and 1 id
        0 ------ 3
        |        |
        |        |
        1 ------ 2    
    Args:
        Vector3D (_type_): _description_
    """
    _is_opt:            bool = False
    _cross_check:       bool = False
    _valid_id:          int = -1
    
    