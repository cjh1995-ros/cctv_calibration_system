from dataclasses import dataclass, field
import numpy as np


@dataclass
class Vector2D:
    _x: float = 0.0
    _y: float = 0.0
    _xy: np.ndarray = field(init=False) 
    
    def __post_init__(self):
        self._xy = np.array([self._x, self._y], dtype=np.float32)
    
    def norm(self):
        return np.linalg.norm(self.xy)
    
    
    def noisy(self, sigma: float):
        self.xy += np.random.normal(0, sigma, 2)
        
    
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
        
        
    @staticmethod
    def random():
        return Vector2D(np.random.rand(2))
    
        
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
        self._xyz = np.array([self._x, self._y, self._z], dtype=np.float32)
    
    def norm(self):
        return np.linalg.norm(self.xyz)
    
    def noisy(self, sigma: float):
        self.xyz += np.random.normal(0, sigma, 3)
    
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
        
    def cross(self, other):
        ret = Vector3D()
        ret.x = self.y * other.z - self.z * other.y
        ret.y = self.z * other.x - self.x * other.z
        ret.z = self.x * other.y - self.y * other.x
        return ret