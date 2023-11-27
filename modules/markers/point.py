from dataclasses import field
from typing import List
from modules.markers import Vector2D, Vector3D





class Feature2D(Vector2D):
    """
    Args:
        Vector2D (_type_): _description_
    """
    _id:            int = -1 # feature id in camera
    _camera_id:     int = -1 # camera id

    def __post_init__(self):
        super().__post_init__()
        self._id = -1
        self._camera_id = -1
        
    @property
    def id(self):
        return self._id
    
    @property
    def camera_id(self):
        return self._camera_id
    
    @id.setter
    def id(self, id: int):
        self._id = id
        
    @camera_id.setter
    def camera_id(self, camera_id: int):
        self._camera_id = camera_id
    
    def __str__(self):
        return f"Feature2D: id: {self.id}, camera_id: {self.camera_id}, x: {self.x}, y: {self.y}"
    
    def __repr__(self):
        return f"Feature2D: id: {self.id}, camera_id: {self.camera_id}, x: {self.x}, y: {self.y}"
    
    def to_npy(self):
        return self._xy



class Feature3D(Vector3D):
    """
    Args:
        Vector3D (_type_): _description_
    """
    _id:            int                        # feature id in world
    _camera_id:     List[int] = field(init=False)   # camera id
    
    def __post_init__(self):
        super().__post_init__()
        self._id = -1
        
    @property
    def id(self):
        return self._id
    
    @property
    def camera_id(self):
        return self._camera_id
    
    @id.setter
    def id(self, id: int):
        self._id = id
        
    def to_npy(self):
        return self._xyz