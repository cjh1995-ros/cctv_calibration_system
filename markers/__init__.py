from dataclasses import dataclass, field
import numpy as np


@dataclass
class Vector2D:
    _x: float = 0.0
    _y: float = 0.0
    _xy: np.ndarray = field(init=False) 
    
    def __post_init__(self):
        self._xy = np.array([self._x, self._y])
    
    def norm(self):
        return np.linalg.norm(self.xy)
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def xy(self):
        return self._xy
    
    @x.setter
    def x(self, x: float):
        self._x = x
        self._xy[0] = x
        
    @y.setter
    def y(self, y: float):
        self._y = y
        self._xy[1] = y
        
    @xy.setter
    def xy(self, xy: np.ndarray):
        self._xy = xy
        self._x = xy[0]
        self._y = xy[1]
        
    def __add__(self, other):
        ret = Vector2D()
        ret.x = self.x + other.x
        ret.y = self.y + other.y
        return ret
    
    def __sub__(self, other):
        ret = Vector2D()
        ret.x = self.x - other.x
        ret.y = self.y - other.y
        return ret
    
    def __mul__(self, f):
        ret = Vector2D()
        ret.xy = self.xy * f
        return ret
        
    def __matmul__(self, other):
        ret = 0.0
        ret += self.x * other.x
        ret += self.y * other.y
        return ret
    
    def __truediv__(self, f):
        ret = Vector2D()
        ret.xy = self.xy / f
        return ret
    
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
    


@dataclass
class Vector3D:
    _x: float = 0.0
    _y: float = 0.0
    _z: float = 0.0
    _xyz: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self._xyz = np.array([self._x, self._y, self._z])
    
    def norm(self):
        return np.linalg.norm(self.xyz)
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z
    
    @property
    def xyz(self):
        return self._xyz
    
    @x.setter
    def x(self, x: float):
        self._x = x
        self._xyz[0] = x
        
    @y.setter
    def y(self, y: float):
        self._y = y
        self._xyz[1] = y
        
    @z.setter
    def z(self, z: float):
        self._z = z
        self._xyz[2] = z
        
    @xyz.setter
    def xyz(self, xyz: np.ndarray):
        self._xyz = xyz
        self._x = xyz[0]
        self._y = xyz[1]
        self._z = xyz[2]
        
    def __add__(self, other):
        ret = Vector3D()
        ret.x = self.x + other.x
        ret.y = self.y + other.y
        ret.z = self.z + other.z
        return ret
    
    def __sub__(self, other):
        ret = Vector3D()
        ret.x = self.x - other.x
        ret.y = self.y - other.y
        ret.z = self.z - other.z
        return ret
    
    def __mul__(self, f):
        ret = Vector3D()
        ret.xyz = self.xyz * f
        return ret
        
    def __matmul__(self, other):
        ret = 0.0
        ret += self.x * other.x
        ret += self.y * other.y
        ret += self.z * other.z
        return ret
    
    def __truediv__(self, f):
        ret = Vector3D()
        ret.xyz = self.xyz / f
        return ret
        
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
        


# @dataclass
# class Feature3d(BasePoint3d):
#     """
#     id:         marker id
#     cam_id:     camera id for visible graph
#     color:      color for visualization
#     is_opt:     is optimized?
#     opt_list:   bool list for optimization. It will be set from outside
#     """
#     _id:            List[int] = field(default_factory=list)
#     _cam_id:        List[int] = field(default_factory=list)
#     _color:         np.ndarray = field(default_factory=np.ndarray)
#     _is_opt:        bool = False
#     _opt_list:      List[bool] = field(default_factory=list)

#     def __post_init__(self):
#         self.color = np.random.randint(0, 255, (3, ), dtype=np.uint8)

#     @property
#     def id(self):
#         return self._id
    
#     @property
#     def cam_id(self):
#         return self._cam_id
    
#     @property
#     def color(self):
#         return self._color
    
#     @property
#     def is_opt(self):
#         return self._is_opt
    
#     @property
#     def opt_list(self):
#         return self._opt_list
    
#     @property
#     def opt_params(self):
#         return self._xyz[self._opt_list]

#     @id.setter
#     def id(self, id: List[int]):
#         self._id = id

#     @cam_id.setter
#     def cam_id(self, cam_id: List[int]):
#         self._cam_id = cam_id

#     @color.setter
#     def color(self, color: np.ndarray):
#         self._color = color

#     @is_opt.setter
#     def is_opt(self, is_opt: bool):
#         self._is_opt = is_opt

#     @opt_list.setter
#     def opt_list(self, opt_list: List[bool]):
#         self._opt_list = opt_list    

#     def size(self):
#         return np.linalg.norm(self.xyz)

#     def viz_graph(self):
#         """
#         It will return visible graph
#         """
#         tmp = {}
#         for cam_id in self.cam_id:
#             tmp[cam_id] = self._id
#         return tmp
    
#     def __sub__(self, other):
#         ret = Feature3d()
#         ret.x = self.x - other.x
#         ret.y = self.y - other.y
#         ret.z = self.z - other.z
#         return ret

#     def __add__(self, other):
#         ret = Feature3d()
#         ret.x = self.x + other.x
#         ret.y = self.y + other.y
#         ret.z = self.z + other.z
#         return ret
    
#     def __mul__(self, f):
#         self.xyz = self.xyz * f


#     def __matmul__(self, other):
#         ret = 0.0
#         ret += self.x * other.x
#         ret += self.y * other.y
#         ret += self.z * other.z
#         return ret
    
#     def angle(self, other):
#         ret = 0.0
#         ret += self @ other
#         return ret / (self.size() * other.size())

# @dataclass
# class BaseRect():
#     """
#         This class is for rectangle marker.
#         It has 4 markers and 1 id
#         0 ------ 3
#         |        |
#         |        |
#         1 ------ 2
#     """
#     _rect_id:       int = -1
#     _markers:       List[Feature3d] = field(default_factory=list) # should be 4 length

#     @property
#     def rect_id(self):
#         return self._rect_id
    
#     @property
#     def markers(self):
#         return self._markers

#     @rect_id.setter
#     def rect_id(self, rect_id: int):
#         self._rect_id = rect_id

#     @markers.setter
#     def markers(self, markers: List[Feature3d]):
#         self._markers = markers



# @dataclass
# class RoadMarker(BaseRect):
#     """
#         This class is for cross check rectangle marker.
#         It has 4 markers and 1 id
#         0 ------ 3
#         |        |
#         |        |
#         1 ------ 2
#     """
#     _cross_check:       bool = False
#     _valid_id:          int = -1 # should be one of 0 ~ 3
#     _train_markers:     List[Feature3d] = field(default_factory=list) # should be 3 length
#     _valid_marker:      Feature3d = Feature3d()

#     @property
#     def cross_check(self):
#         return self._cross_check
    
#     @property
#     def valid_id(self):
#         return self._valid_id
    
#     @cross_check.setter
#     def cross_check(self, cross_check: bool):
#         self._cross_check = cross_check

#     @valid_id.setter
#     def valid_id(self, valid_id: int):
#         self._valid_id = valid_id

#     @property
#     def train_markers(self):
#         """
#             It will return train markers except valid marker.
#         """
#         train_markers = []
#         for idx in range(4):
#             if idx != self._valid_id:
#                 train_markers.append(self.markers[idx])
        
#         return train_markers
    
#     @train_markers.setter
#     def train_markers(self, train_markers: List[Feature3d]):
#         self._train_markers = train_markers

#     @property
#     def valid_marker(self):
#         """
#             It will return valid marker.
#             Normally, it will be returned after optimized.
#             So we should estimate valid marker through train markers.
#             As they must be on same plane, we can estimate valid marker 
#             with vector addition.
#         """
#         cross_id_dict = {0: 2, 1: 3, 2: 0, 3: 1}
#         cross_id = cross_id_dict[self._valid_id]

#         vecs = []

#         for idx in range(4):
#             if idx != cross_id and idx != self._valid_id:
#                 vecs.append(self.markers[idx] - self.markers[cross_id])

#         return self.markers[cross_id] + vecs[0] + vecs[1]
    
#     def is_rect(self):
#         """
#             Checking the markers construct rect.
#             It will be decided by cos
#         """
#         ...

#     def rect_err(self):
#         """

#         """