import unittest
from modules.markers.point import Feature3D
from modules.markers.rectangle import Rect3D
from typing import List

import numpy as np

def basic_case_0() -> Rect3D:
    f0 = Feature3D(0, 0, 0)
    f1 = Feature3D(0, 1, 0)
    f2 = Feature3D(1, 1, 0)
    f3 = Feature3D(1, 0, 0)
    corners = [f0, f1, f2, f3]
    
    rect = Rect3D()
    
    rect.corners = corners

    return rect

def basic_case_1() -> Rect3D:
    f0 = Feature3D(0, 0, 1)
    f1 = Feature3D(0, 1, 0)
    f2 = Feature3D(1, 1, 0)
    f3 = Feature3D(1, 0, 0)
    corners = [f0, f1, f2, f3]
    
    rect = Rect3D()
    
    rect.corners = corners

    return rect

def two_rect_case_0() -> List[Rect3D]:
    rect = basic_case_0()
    
    

class TestRect3D(unittest.TestCase):
    def test_init(self):
        f0 = Feature3D(0, 0, 0)
        f1 = Feature3D(0, 1, 0)
        f2 = Feature3D(1, 1, 0)
        f3 = Feature3D(1, 0, 0)
        corners = [f0, f1, f2, f3]
        
        camera_ids = [0, 1, 2] # 0~2 카메라까지 보인다는 뜻
        
        rect = Rect3D()
        
        rect.id = 0
        rect.corners = corners
        rect.camera_id = camera_ids
        
        self.assertEqual(rect.corners, corners, "Corners are not same")
        self.assertEqual(rect.camera_id, camera_ids, "Camera IDX are not same")
    
    
    def test_coplanarity_v0(self):
        rect = basic_case_0()
        
        self.assertEqual(rect.is_plane(), True, "This is not coplanar")
    
    
    def test_coplanarity_v1(self):
        rect = basic_case_1()
        
        self.assertEqual(rect.is_plane(), False, "This is not coplanar")

    
    def test_is_rect(self):
        rect = basic_case_0()
        
        self.assertEqual(rect.is_rect(), True, "This is not rectangular")
        
    
    def test_to_npy(self):
        rect = basic_case_0()
        
        npy_0 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        
        npy_rect = rect.to_npy()
        
        self.assertEqual(npy_0.shape, npy_rect.shape, "Numpy array shape is not same")
        self.assertIsInstance(npy_rect, np.ndarray, "Numpy array is not instance of np.ndarray")
        self.assertAlmostEqual(np.linalg.norm(npy_0 - npy_rect), 0.0, delta=1e-6, msg="Numpy array is not same")
    
    def test_noisy(self):
        rect = basic_case_0()
        
        rect.noisy(0.1)
        
        self.assertNotEqual(rect.corners[0].x, 0.0, msg="Noisy is not working")
        self.assertNotEqual(rect.corners[0].y, 0.0, msg="Noisy is not working")
        self.assertNotEqual(rect.corners[0].z, 0.0, msg="Noisy is not working")
    
    
if __name__ == '__main__':
    unittest.main()
