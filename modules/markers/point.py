from dataclasses import field
from typing import List
from modules.markers import Vector2D, Vector3D
import numpy as np




class Feature2D(Vector2D):
    """
    Args:
        Vector2D (_type_): _description_
    """
    _id:            int # feature id in camera
    _camera_id:     int = -1 # camera id

    def __post_init__(self):
        super().__post_init__()
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

    def from_npy(self, id, camera_id, data: np.ndarray):
        self.id = id
        self.camera_id = camera_id
        self._xy = data

class Feature3D(Vector3D):
    """
    Args:
        Vector3D (_type_): _description_
    """
    _id:            int                        # feature id in world
    _camera_id:     List[int] = field(init=False)   # camera id
    
    def __post_init__(self):
        super().__post_init__()
        self._camera_id = []
        
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id: int):
        self._id = id
    
    @property
    def camera_id(self):
        return self._camera_id

    @camera_id.setter
    def camera_id(self, camera_id: List[int]):
        self._camera_id = camera_id    
        
    def to_npy(self):
        return self._xyz
    
    def from_npy(self, id:int, camera_id: List[int], data: np.ndarray):
        self.id = id
        self.camera_id = camera_id
        self._xyz = data



class Feature3Ds:
    """
    Args:
        _type_ (_type_): _description_
    """
    _features: List[Feature3D] = field(init=False)
    
    def __post_init__(self):
        self._features = []
        
    def __getitem__(self, index):
        return self._features[index]
    
    def __setitem__(self, index, value):
        self._features[index] = value
        
    def __len__(self):
        return len(self._features)
    
    def __str__(self):
        return f"Feature3Ds: {self._features}"
    
    def __repr__(self):
        return f"Feature3Ds: {self._features}"
    
    def append(self, feature: Feature3D):
        self._features.append(feature)
        
    def extend(self, features: List[Feature3D]):
        self._features.extend(features)
        
    def to_npy(self):
        return np.array([feature.to_npy() for feature in self._features])
        
    def from_npy(self, data: np.ndarray):
        self._features = [Feature3D.from_npy(feature) for feature in data]
                
    def get_noisy(self, sigma: float):
        from copy import deepcopy
        
        copied = deepcopy(self)
        
        for feature in copied._features:
            feature.noisy(sigma)
            
        return copied
            
    def get_camera_ids(self):
        camera_ids = []
        
        for feature in self._features:
            camera_ids.extend(feature.camera_id)
            
        return list(set(camera_ids))

    def set_camera_ids(self, camera_ids: List[List[int]]):
        for i, feature in enumerate(self._features):
            feature.camera_id = camera_ids[i]
    
    def get_feature_ids(self):
        feature_ids = []
        
        for feature in self._features:
            feature_ids.append(feature.id)
            
        return list(set(feature_ids))
    
    def get_features_by_camera_id(self, camera_id: int):
        from copy import deepcopy
        
        features = []
        
        for feature in self._features:
            if camera_id in feature.camera_id:
                features.append(deepcopy(feature))
                
        return features
    