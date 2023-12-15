from dataclasses import dataclass, field
from typing import Any, Dict, Callable
# import numpy as np
from autograd import numpy as np
from modules.cameras.projectors import Projector, Distorter, BasicConvertor




""""""
INTR_OPT_PARAM_NUM : Dict[str, int]= \
{
    "CONST":            0,
    "FOCAL":            1,
    "FCXCY":            3,
    "FXYCXY":           4,
}

PROJECTION_PARAM_NUM : Dict[str, int] = \
{
    "PERSPECTIVE":      0,
    "EQUIDISTANT":      0,
    "SINGLE_SPHERE":    1,
    "DOUBLE_SPHERE":    2,
    "TRIPLE_SPHERE":    3,
}

PROJECTOIN_FUNC : Dict[str, Callable] = \
{
    "PERSPECTIVE":      Projector.perspective,
    "EQUIDISTANT":      Projector.equidistant,
    "SINGLE_SPHERE":    Projector.single_sphere,
    "DOUBLE_SPHERE":    Projector.double_sphere,
    "TRIPLE_SPHERE":    Projector.triple_sphere,
}

DISTORTION_PARAM_NUM : Dict[str, int] = \
{
    "NONE":             0,
    "POLYNOMIAL":       2,
    "FOV":              1,
    "EQUIDISTANT":      0,
}

DISTORTION_FUNC : Dict[str, Callable] = \
{
    # "NONE":             Distorter.none,
    "POLYNOMIAL":       Distorter.polynomial,
    "FOV":              Distorter.fov,
    "EQUIDISTANT":      Distorter.equidistant,
}


@dataclass
class Camera:
    _id:                int
    _intr_opt_type:     str
    _is_opt_extr:       bool
    _proj_func_type:    str
    _dist_type:         str
    _params:            np.ndarray = field(init=False)
    _initial_params:    np.ndarray = field(init=False)
    K:                  np.ndarray = field(init=False)
    R:                  np.ndarray = field(init=False)
    t:                  np.ndarray = field(init=False)
    dtype:              Any = np.float32
    
    def __post_init__(self):
        self.n_opt_intr    = INTR_OPT_PARAM_NUM[self._intr_opt_type]
        self.n_opt_extr    = 6 if self._is_opt_extr else 0
        self.n_proj_func   = PROJECTION_PARAM_NUM[self._proj_func_type]
        self.n_dist        = DISTORTION_PARAM_NUM[self._dist_type]
        
        self.n_total = self.n_opt_intr + self.n_proj_func + self.n_dist + self.n_opt_extr
        
        # Init K and dist
        self.K = np.zeros((3, 3), dtype=self.dtype)
        
        # Init params for optimization
        self.params = np.zeros(self.n_total, dtype=self.dtype)
    
    def project(self, pts: np.ndarray) -> np.ndarray:
        """
        Project world coordinate points to image plane
        Args:
            pts (np.ndarray): points in world coordinate, 3d

        Returns:
            np.ndarray: projected point, 2d
        """
        # Conver points from world to camera coordinate
        pts = BasicConvertor.world_to_camera(pts, self.R, self.t)
        
        # Project points into normalized plane
        pts, r_us = PROJECTOIN_FUNC[self._proj_func_type](pts, self._params[self.n_opt_intr:self.n_opt_intr + self.n_proj_func])              

        # Distort points
        if self._dist_type != "NONE":
            # print(self._params[self.n_opt_intr + self.n_proj_func:self.n_opt_intr + self.n_proj_func + self.n_dist])
            r_ds = DISTORTION_FUNC[self._dist_type](r_us, self._params[self.n_opt_intr + self.n_proj_func:self.n_opt_intr + self.n_proj_func + self.n_dist])
            
            # Rescale the normalized pixels with rd / ru
            pts = BasicConvertor.rescale(r_ds / r_us, pts)
        
        
        # Convert normalized coordinate to pixel coordinate
        pts = BasicConvertor.homogeneous(pts)
        pts = BasicConvertor.normalized_to_pixel(pts, self.K)
        
        return BasicConvertor.dehomogeneous(pts)
    
    @property
    def initial_params(self):
        return self._initial_params
    
    @initial_params.setter
    def initial_params(self, initial_params: np.ndarray):
        """_summary_
        Args:
            params (np.ndarray): 
                It should be full parameters for initializing params.
                Initialize K, dist, extrinsic, sphere params
                params should be like this: [intr, projection(sphere), dist, extrinsic]
                ex) [f, cx, cy, alpha, gamma, ..., k1, k2, r, t]
        """
        self._initial_params = initial_params
        m = 0
        n = 3
        self.init_K(self._initial_params[m:n])
        
        m = n
        n += self.n_proj_func
        self.init_sphere(self._initial_params[m:n])
        
        m = n
        n += self.n_dist
        self.init_dist(self._initial_params[m:n])
        
        self.init_transform(self._initial_params[-6:])

    
    def init_K(self, intr: np.ndarray):
        # Init K
        self.K[0, 0] = intr[0]
        self.K[1, 1] = intr[0]
        self.K[0, 2] = intr[1]
        self.K[1, 2] = intr[2]
        
        # Init params
        if self._intr_opt_type == "CONST":
            pass
        elif self._intr_opt_type == "FOCAL":
            self._params[:self.n_opt_intr] = intr[0]
        elif self._intr_opt_type == "FCXCY":
            self._params[:self.n_opt_intr] = intr
        elif self._intr_opt_type == "FXYCXY":
            self._params[0] = intr[0]
            self._params[1:self.n_opt_intr] = intr

    def init_sphere(self, params: np.ndarray):
        """무조건 최적화"""
        self._params[self.n_opt_intr:self.n_opt_intr + self.n_proj_func] = params
        
    def init_transform(self, params: np.ndarray):
        self.R = params[:3]
        self.t = params[3:]
        
        if self._is_opt_extr:
            self._params[-self.n_opt_extr:] = params

    def init_dist(self, params):
        """무조건 최적화"""
        self._params[self.n_opt_intr + self.n_proj_func:self.n_opt_intr + self.n_proj_func + self.n_dist] = params
    
    ######## Start of Setter and Getter ########
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id: int):
        self._id = id
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params: np.ndarray):
        self._params = params
        
        # Change K, R, t
        # 1. K
        if self._intr_opt_type == "CONST":
            pass
        elif self._intr_opt_type == "FOCAL":
            self.K[0, 0] = self._params[0]
            self.K[1, 1] = self._params[0]
        elif self._intr_opt_type == "FCXCY":
            self.K[0, 0] = self._params[0]
            self.K[1, 1] = self._params[0]
            self.K[0, 2] = self._params[1]
            self.K[1, 2] = self._params[2]
        elif self._intr_opt_type == "FXYCXY":
            self.K[0, 0] = self._params[0]
            self.K[1, 1] = self._params[1]
            self.K[0, 2] = self._params[2]
            self.K[1, 2] = self._params[3]
        
        # 2. R, t
        if self._is_opt_extr:
            self.R = self._params[-6:-3]
            self.t = self._params[-3:]
        