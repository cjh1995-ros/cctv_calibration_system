from typing import List, Dict
from modules.gui.viz_matplotlib import MatplotVisualizer
from modules.cameras import Camera
from modules import conv_pose2transform, conv_transform2pose
from autograd import numpy as np

    
def generate_cameras(datas: List[Dict]):
    cameras = []
    
    for i, data in enumerate(datas):
        tmp = Camera(data['id'],
                        data['intr_opt_type'],
                        data['is_extr_opt'],
                        data['proj_func_type'],
                        data['dist_type'])
        
        tmp.initial_params = data['init_params']
        
        cameras.append(tmp)
        
    return cameras                


def test_cameras():
    from scipy.spatial.transform import Rotation as R
    datas = []
    
    ori1 = np.array([[1, 0, 0], 
                        [0, 0, 1], 
                        [0, -1, 0]], dtype=np.float64)
    
    rot1 = np.array([-np.pi/6, np.pi/4, 0], dtype=np.float64)
    ori1 = ori1 @ R.from_rotvec(rot1).as_matrix()
    
    ori_rvec1 = R.from_matrix(ori1).as_rotvec()
    pos1 = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    pose1 = np.concatenate([ori_rvec1, pos1])
    transform1 = conv_pose2transform(pose1)
    
    ori2 = np.array([[1, 0, 0], 
                        [0, 0, 1], 
                        [0, -1, 0]], dtype=np.float64)

    rot2 = np.array([-np.pi/6, -np.pi/4, 0], dtype=np.float64)
    ori2 = ori2 @ R.from_rotvec(rot2).as_matrix()

    ori_rvec2 = R.from_matrix(ori2).as_rotvec()
    pos2 = np.array([4.0, 0.0, 3.0], dtype=np.float64)
    pose2 = np.concatenate([ori_rvec2, pos2])
    transform2 = conv_pose2transform(pose2)
    
    ori3 = np.array([[1, 0, 0], 
                        [0, 0, 1], 
                        [0, -1, 0]], dtype=np.float64)

    rot3 = np.array([0, -np.pi * 2/3, -np.pi/3], dtype=np.float64)
    ori3 = ori3 @ R.from_rotvec(rot3).as_matrix()

    ori_rvec3 = R.from_matrix(ori3).as_rotvec()
    pos3 = np.array([4.0, 4.0, 3.0], dtype=np.float64)
    pose3 = np.concatenate([ori_rvec3, pos3])
    transform3 = conv_pose2transform(pose3)

    ori4 = np.array([[1, 0, 0], 
                        [0, 0, 1], 
                        [0, -1, 0]], dtype=np.float64)
    pos4 = np.array([0.0, 4.0, 3.0], dtype=np.float64) # 

    rot4 = np.array([0, np.pi * 2/3, np.pi/3], dtype=np.float64)
    ori4 = ori4 @ R.from_rotvec(rot4).as_matrix()

    ori_rvec4 = R.from_matrix(ori4).as_rotvec()
    pose4 = np.concatenate([ori_rvec4, pos4])
    transform4 = conv_pose2transform(pose4)

    ori5 = np.array([[0, +1, 0], 
                        [-1, 0, 0], 
                        [0, 0, -1]], dtype=np.float64)
    pos5 = np.array([0.0, 0.0, 5.0], dtype=np.float64) # 

    # rot5 = np.array([0, np.pi * 2/3, np.pi/3], dtype=np.float64)
    # ori5 = ori5 @ R.from_rotvec(rot5).as_matrix()

    ori_rvec5 = R.from_matrix(ori5).as_rotvec()
    pose5 = np.concatenate([ori_rvec5, pos5])
    transform5 = conv_pose2transform(pose5)

    
    datas.append({"id": 0, 
                    "intr_opt_type": "FOCAL", 
                    "is_extr_opt": True, 
                    "proj_func_type": "PERSPECTIVE", 
                    "dist_type": "POLYNOMIAL",
                    "init_params": [1000.0, 500.0, 500.0, -0.2, 0.1, transform1[0], transform1[1], transform1[2], transform1[3], transform1[4], transform1[5]]})

    datas.append({"id": 1, 
                    "intr_opt_type": "FOCAL", 
                    "is_extr_opt": True, 
                    "proj_func_type": "PERSPECTIVE", 
                    "dist_type": "POLYNOMIAL",
                    "init_params": [1000.0, 500.0, 500.0, -0.2, 0.1, transform2[0], transform2[1], transform2[2], transform2[3], transform2[4], transform2[5]]})

    datas.append({"id": 2, 
                    "intr_opt_type": "FOCAL", 
                    "is_extr_opt": True, 
                    "proj_func_type": "PERSPECTIVE", 
                    "dist_type": "POLYNOMIAL",
                    "init_params": [1000.0, 500.0, 500.0, -0.2, 0.1, transform3[0], transform3[1], transform3[2], transform3[3], transform3[4], transform3[5]]})

    datas.append({"id": 3, 
                    "intr_opt_type": "FOCAL", 
                    "is_extr_opt": True, 
                    "proj_func_type": "PERSPECTIVE", 
                    "dist_type": "POLYNOMIAL",
                    "init_params": [1000.0, 500.0, 500.0, -0.2, 0.1, transform4[0], transform4[1], transform4[2], transform4[3], transform4[4], transform4[5]]})

    # datas.append({"id": 4, 
    #                 "intr_opt_type": "FOCAL", 
    #                 "is_extr_opt": True, 
    #                 "proj_func_type": "PERSPECTIVE", 
    #                 "dist_type": "POLYNOMIAL",
    #                 "init_params": [1000.0, 500.0, 500.0, -0.2, 0.1, transform5[0], transform5[1], transform5[2], transform5[3], transform5[4], transform5[5]]})
            
    return generate_cameras(datas)    





def test_pcl_01():
    # Test point cloud 1. Z = 0, number of points = 16. Shape like grid
    first_rect  = np.array([[0.5, 0.5, 0.0],
                           [1.5, 0.5, 0.0],
                           [1.5, 1.5, 0.0],
                           [0.5, 1.5, 0.0]], dtype=np.float64)
    second_rect = np.array([[2.0, 0.5, 0.0],
                            [3.0, 0.5, 0.0],
                           [3.0, 1.5, 0.0],
                           [2.0, 1.5, 0.0]], dtype=np.float64)
    third_rect  = np.array([[0.5, 2.0, 0.0],
                           [1.5, 2.0, 0.0],
                           [1.5, 3.0, 0.0],
                           [0.5, 3.0, 0.0]], dtype=np.float64)
    fourth_rect = np.array([[2.0, 2.0, 0.0],
                            [3.0, 2.0, 0.0],
                           [3.0, 3.0, 0.0],
                           [2.0, 3.0, 0.0]], dtype=np.float64)
    points = np.concatenate([first_rect, second_rect, third_rect, fourth_rect], axis=0).reshape(-1, 3)
    return points

def test_pcl_02():
    # Test point cloud 2. Z = 0, number of points = 16. Shape like messy
    first_rect  = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)
    second_rect = np.array([[0.5, 0.5, 0.0],
                            [1.5, 0.5, 0.0],
                           [1.5, 1.5, 0.0],
                           [0.5, 1.5, 0.0]], dtype=np.float64)
    third_rect  = np.array([[-0.5, 0.75, 0.0],
                           [0.5, 0.75, 0.0],
                           [0.5, 1.75, 0.0],
                           [-0.5, 1.75, 0.0]], dtype=np.float64)
    fourth_rect = np.array([[-0.75, -0.75, 0.0],
                            [0.25, -0.75, 0.0],
                            [0.25, 0.25, 0.0],
                            [-0.75, 0.25, 0.0]], dtype=np.float64)
    points = np.concatenate([first_rect, second_rect, third_rect, fourth_rect], axis=0).reshape(-1, 3)
    
    return points    


def test_pcl_03():
    # Test point cloud 1. Z = 0, number of points = 16. Shape like grid
    first_rect  = np.array([[0.5, 0.5, 0.0],
                           [1.5, 0.5, 0.0],
                           [1.5, 1.5, 0.0],
                           [0.5, 1.5, 0.0]], dtype=np.float64)
    second_rect = np.array([[2.0, 0.5, 0.0],
                            [3.0, 0.5, 0.0],
                           [3.0, 1.5, 0.0],
                           [2.0, 1.5, 0.0]], dtype=np.float64)
    third_rect  = np.array([[0.5, 2.0, 0.5],
                           [1.5, 2.0, 0.5],
                           [1.5, 3.0, 1.5],
                           [0.5, 3.0, 1.5]], dtype=np.float64)
    fourth_rect = np.array([[2.0, 2.0, 0.5],
                            [3.0, 2.0, 0.5],
                           [3.0, 3.0, 1.5],
                           [2.0, 3.0, 1.5]], dtype=np.float64)
    points = np.concatenate([first_rect, second_rect, third_rect, fourth_rect], axis=0).reshape(-1, 3)
    return points


POINT = {
    "case1": test_pcl_01,
    "case2": test_pcl_02,
    "case3": test_pcl_03,
}


if __name__ == '__main__': 
    cameras = test_cameras()    
    
    # Save datas as npy
    cam = np.array([cam.initial_params for cam in cameras])
    # np.save("./data/cameras.npy", cam)

    points = POINT["case3"]()
    
    vz = MatplotVisualizer(is_inv=True)
    vz.vis_3d(None, cameras, points)
    vz.vis_satellite(None, cameras, points, scale=80)
    
    # for case in ["case1", "case2", "case3"]:
    #     points = POINT[case]()

    #     vz = MatplotVisualizer(is_inv=True)
    #     # vz.vis_3d(None, cameras, points)
    #     vz.vis_satellite(None, cameras, points, scale=80)

    #     # np.save(f"./data/points_{case}.npy", points)
    #     # print(f"Save points_{case}.npy")

        
    # # Visualize
