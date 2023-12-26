from dataclasses import dataclass, field
from typing import Any, Dict, Callable
# import numpy as np
from autograd import numpy as np
from modules.cameras.projectors import Projector, Distorter, BasicConvertor





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

PROJECTION_PARAM_INIT : Dict[str, Any] = \
{
    "PERSPECTIVE":      np.array([], dtype=np.float32),
    "EQUIDISTANT":      np.array([], dtype=np.float32),
    "SINGLE_SPHERE":    np.array([0.5], dtype=np.float32),
    "DOUBLE_SPHERE":    np.array([0.5, 0.0], dtype=np.float32),
    "TRIPLE_SPHERE":    np.array([0.5, 0.0, 0.0], dtype=np.float32),    
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
    "NONE":             Distorter.none,
    "POLYNOMIAL":       Distorter.polynomial,
    "FOV":              Distorter.fov,
    "EQUIDISTANT":      Distorter.equidistant,
}

DISTORTION_PARAM_INIT : Dict[str, Any] = \
{
    "NONE":             None,
    "POLYNOMIAL":       np.array([-0.2, 0.0], dtype=np.float32),
    "FOV":              np.array([0.5], dtype=np.float32),
    "EQUIDISTANT":      None
}

@dataclass
class Camera:
    """
        [f, cx, cy, alpha, gamma, ..., k1, k2]
    """
    _intr_opt_type:     str
    _proj_func_type:    str
    _dist_type:         str
    _params:            np.ndarray = field(init=False)
    _initial_params:    np.ndarray = field(init=False)
    K:                  np.ndarray = field(init=False)
    dtype:              Any = np.float32
    
    def __post_init__(self):
        self.n_intr     = INTR_OPT_PARAM_NUM[self._intr_opt_type]
        self.n_proj     = PROJECTION_PARAM_NUM[self._proj_func_type]
        self.n_dist     = DISTORTION_PARAM_NUM[self._dist_type]
        
        self.n_total = self.n_intr + self.n_proj + self.n_dist
        
        # Init K and dist
        self.K = np.zeros((3, 3), dtype=self.dtype)
        
        # Init params for optimization
        self._params = np.zeros(self.n_total, dtype=self.dtype)
        
        # Init projection params
        self._params[self.n_intr:self.n_intr + self.n_proj] = PROJECTION_PARAM_INIT[self._proj_func_type]
        
        # Init distortion params
        self._params[self.n_intr + self.n_proj:self.n_intr + self.n_proj + self.n_dist] = DISTORTION_PARAM_INIT[self._dist_type]
    
    def project(self, pts: np.ndarray) -> np.ndarray:
        """
        Project camera coordinate points to image plane
        Args:
            pts (np.ndarray): points in camera coordinate, 3d

        Returns:
            np.ndarray: projected point, 2d
        """
        # Project points into normalized plane, pt_u and ru
        pts, r_us = PROJECTOIN_FUNC[self._proj_func_type](pts, self._params[self.n_intr:self.n_intr + self.n_proj])              

        # Distort points
        r_ds = DISTORTION_FUNC[self._dist_type](r_us, self._params[self.n_intr + self.n_proj:self.n_intr + self.n_proj + self.n_dist])
        
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
        self.init_intrinsic(self._initial_params[m:n])
        
        m = n
        n += self.n_proj
        self.init_sphere(self._initial_params[m:n])
        
        m = n
        n += self.n_dist
        self.init_dist(self._initial_params[m:n])

    
    def init_intrinsic(self, intr: np.ndarray):
        assert len(intr) == 3, "Initial intrinsics shapes should be f, cx, cy"
        
        # Init K
        self.K[0, 0] = intr[0]
        self.K[1, 1] = intr[0]
        self.K[0, 2] = intr[1]
        self.K[1, 2] = intr[2]
        
        # Init params
        if self._intr_opt_type == "CONST":
            pass
        elif self._intr_opt_type == "FOCAL":
            self._params[:self.n_intr] = intr[0]
        elif self._intr_opt_type == "FCXCY":
            self._params[:self.n_intr] = intr
        elif self._intr_opt_type == "FXYCXY":
            self._params[0] = intr[0]
            self._params[1:self.n_intr] = intr
            

    def init_sphere(self, params: np.ndarray):
        """무조건 최적화"""
        self._params[self.n_intr:self.n_intr + self.n_proj] = params

    def init_dist(self, params):
        """무조건 최적화"""
        self._params[self.n_intr + self.n_proj:self.n_intr + self.n_proj + self.n_dist] = params
    
    def for_optimize(self):
        """Return the parameters for optimization."""
        
        return self._params, np.array([self.K[1, 1], self.K[2, 1]], dtype=self.dtype)
    
    
    ######## Setter and Getter ########    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params: np.ndarray):
        self._params = params
        
        # Change K
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
            
    def gen_bounds(self):
        """Generate bounds for optimization."""
        bounds = []
        
        f = self.K[0, 0]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        f_min = f * 0.5
        f_max = f * 1.5
        
        cx_min = 0.5 * cx
        cx_max = 1.5 * cx
        
        cy_min = 0.5 * cy
        cy_max = 1.5 * cy
        
        # Intrinsics
        if self._intr_opt_type == "CONST":
            pass
        elif self._intr_opt_type == "FOCAL":
            bounds.append((f_min, f_max))
        elif self._intr_opt_type == "FCXCY":
            bounds.append((f_min, f_max))
            bounds.append((cx_min, cx_max))
            bounds.append((cy_min, cy_max))
        elif self._intr_opt_type == "FXYCXY":
            bounds.append((f_min, f_max))
            bounds.append((f_min, f_max))
            bounds.append((cx_min, cx_max))
            bounds.append((cy_min, cy_max))
            
        # Projection
        if self._proj_func_type == "SINGLE_SPHERE":
            bounds.append((0, 1))
        elif self._proj_func_type == "DOUBLE_SPHERE":
            bounds.append((0, 1))
            bounds.append((0, 1))
        elif self._proj_func_type == "TRIPLE_SPHERE":
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 1))
        
        # Distortion
        if self._dist_type == "NONE":
            pass
        elif self._dist_type == "POLYNOMIAL":
            bounds.append((-0.5, 0.5))
            bounds.append((-0.5, 0.5))
        elif self._dist_type == "FOV":
            bounds.append((0, np.pi))
        elif self._dist_type == "EQUIDISTANT":
            pass
        
        return bounds