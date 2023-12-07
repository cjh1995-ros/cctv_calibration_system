from modules.cameras.camera import Camera

from typing import Dict, List
import json
import numpy as np




class Generator:
    def __init__(self, config):
        self.config = config
    
    def generate_camera(self, datas: List[Dict]):
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
            
        return self.generate_camera(datas)
    

    def generate_default(self):
        from scipy.spatial.transform import Rotation as R
        datas = []
        
        Rmat1 = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, 1, 0]], dtype=np.float32)
        

        
        rvec1 = R.from_matrix(Rmat1).as_rotvec()
        
        datas.append({"id": 0, 
                      "intr_opt_type": "FCXCY", 
                      "is_extr_opt": True, 
                      "proj_func_type": "PERSPECTIVE", 
                      "dist_type": "POLYNOMIAL",
                      "init_params": [1000.0, 500.0, 500.0, 0.0, 0.0, rvec1[0], rvec1[1], rvec1[2], 0.0, 0.0, 3.0]})
        
        # datas.append({"id": 1, 
        #               "intr_opt_type": "FCXCY", 
        #               "is_extr_opt": True, 
        #               "proj_func_type": "PERSPECTIVE", 
        #               "dist_type": "POLYNOMIAL",
        #               "init_params": [1000.0, 500.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
        
        return self.generate_camera(datas)