from modules.generator import BasicGenerator
from modules.cameras import Camera
from typing import Dict, List
import json
import numpy as np




class CameraGenerator(BasicGenerator):
    def generate(self, datas: List[Dict]):
        cameras = []
        
        for i, data in enumerate(datas):
            tmp = Camera(data['id'],
                         data['intr_opt_type'],
                         data['is_extr_opt'],
                         data['proj_func_type'],
                         data['dist_type'])
            
            tmp.init_params(data['init_params'])
            
            cameras.append(tmp)
            
        return cameras                
    
    def generate_from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            datas = json.load(f)
            
        return self.generate(datas)
    

    def generate_default(self):
        from scipy.spatial.transform import Rotation as R
        datas = []
        
        Rmat1 = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, -1, 0]], dtype=np.float32)
        
        Rmat2 = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, -1, 0]], dtype=np.float32)

        Rmat3 = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, -1, 0]], dtype=np.float32)

        Rmat4 = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, -1, 0]], dtype=np.float32)

        
        rvec1 = R.from_matrix(Rmat1).as_rotvec()
        rvec2 = R.from_matrix(Rmat2).as_rotvec()
        rvec3 = R.from_matrix(Rmat3).as_rotvec()
        rvec4 = R.from_matrix(Rmat4).as_rotvec()
        
        datas.append({"id": 0, 
                      "intr_opt_type": "FXYCXY", 
                      "is_extr_opt": True, 
                      "proj_func_type": "PERSPECTIVE", 
                      "dist_type": "NONE",
                      "init_params": [1000.0, 500.0, 500.0, rvec1[0], rvec1[1], rvec1[2], 1.0, -3.0, 3.0]})

        datas.append({"id": 1, 
                      "intr_opt_type": "FXYCXY", 
                      "is_extr_opt": True, 
                      "proj_func_type": "PERSPECTIVE", 
                      "dist_type": "NONE",
                      "init_params": [1000.0, 500.0, 500.0, rvec2[0], rvec2[1], rvec2[2], 2.0, -1.0, 2.0]})

        datas.append({"id": 2, 
                      "intr_opt_type": "FXYCXY", 
                      "is_extr_opt": True, 
                      "proj_func_type": "PERSPECTIVE", 
                      "dist_type": "NONE",
                      "init_params": [1000.0, 500.0, 500.0, rvec3[0], rvec3[1], rvec3[2], 3.0, -2.0, 1.0]})

        datas.append({"id": 3, 
                      "intr_opt_type": "FXYCXY", 
                      "is_extr_opt": True, 
                      "proj_func_type": "PERSPECTIVE", 
                      "dist_type": "NONE",
                      "init_params": [1000.0, 500.0, 500.0, rvec4[0], rvec4[1], rvec4[2], 1.5, -2.5, 3.0]})
                
        return self.generate(datas)
    
    
from modules.markers.point import Feature3D
from modules.markers.rectangle import Rect3D

class MarkerGenerator(BasicGenerator):
    def generate(self, datas: List[Dict]):
        """"""
            
    def generate_from_file(self, file_path: str):
        """"""    

    def generate_point_default(self):
        """"""
        
        
        
class RectangleGenerator(BasicGenerator):
    def generate(self, datas: List[Dict]):
        """"""