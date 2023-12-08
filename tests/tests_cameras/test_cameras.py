import unittest
from modules.cameras import Camera
from modules.generator.generator import CameraGenerator
from modules.cameras.projectors import Projector, Distorter, BasicConvertor

import numpy as np
import cv2

def TestCase1():
    id              = 0
    intr_opt_type   = "FCXCY"
    is_extr_opt   = True
    proj_func_type   = "PERSPECTIVE"
    dist_type       = "POLYNOMIAL"
    # init_params = np.array([1000, 500, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # init_params = np.array([1000.0, 500.0, 500.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # init_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init_params = np.array([1.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # init_params = np.array([1000, 500, 500, 0, 0, 0])
    
    return {"id": id, 
            "intr_opt_type": intr_opt_type, 
            "is_extr_opt": is_extr_opt, 
            "proj_func_type": proj_func_type, 
            "dist_type": dist_type,
            "init_params": init_params}
    
class TestCamera(unittest.TestCase):
    def test_cam_init(self):
        """Check Initialization."""
        tmp = TestCase1()
        cam = Camera(tmp["id"], 
                     tmp["intr_opt_type"], 
                     tmp["is_extr_opt"], 
                     tmp["proj_func_type"], 
                     tmp["dist_type"])
        
        cam.init_params(tmp["init_params"])
        
        # print(f"length of param: {len(cam.params)}")
        # print(f"Initial params: {cam.params}")
        # print(f"K: {cam.K}")
        self.assertEqual(cam._id, tmp["id"], "ID is not same")
        self.assertEqual(cam._intr_opt_type, tmp["intr_opt_type"], "Intr opt type is not same")
        self.assertEqual(cam._is_opt_extr, tmp["is_extr_opt"], "Extr opt type is not same")
        self.assertEqual(cam._proj_func_type, tmp["proj_func_type"], "Proj opt type is not same")
        self.assertEqual(cam._dist_type, tmp["dist_type"], "Dist type is not same")
        
    def test_convertor(self):
        """Check convertor functions."""
        test_pts = np.random.rand(100, 3)
        
        rvec = np.array([1, 0, 0])
        tvec = np.array([0, 0, 0])
        
        # Check if the conversion is same
        cam_pts = BasicConvertor.world_to_camera(test_pts, rvec, tvec)
        world_pts = BasicConvertor.camera_to_world(cam_pts, rvec, tvec)

        self.assertTrue(np.allclose(test_pts, world_pts), "World and camera conversion is not same")

        # Check homogeneous and dehomogeneous
        test2_pts = np.random.rand(100, 2)
        h_pts   = BasicConvertor.homogeneous(test2_pts)
        dh_pts  = BasicConvertor.dehomogeneous(h_pts)

        self.assertEqual(h_pts.shape[1], 3, "Homogeneous conversion is not same")
        self.assertTrue(np.allclose(test2_pts, dh_pts), "Homogeneous and dehomogeneous conversion is not same")

    def test_projector(self):
        """Check projector functions."""
        random_pts = np.random.rand(100, 3)
        
        # Test Perspective Projection
        new_pts, ru = Projector.perspective(random_pts, None)
        self.assertEqual(new_pts.shape[1], 2, "Homogeneous conversion is not same")
        
        r_ds = Distorter.polynomial(ru, np.array([0.1, 0.1]))
        new_pts = BasicConvertor.rescale(r_ds / ru, new_pts)
        self.assertEqual(new_pts.shape[1], 2, "Homogeneous conversion is not same")
        
        new_pts = BasicConvertor.homogeneous(new_pts)
        self.assertEqual(new_pts.shape[1], 3, "Homogeneous conversion is not same")
        
        new_pts = BasicConvertor.normalized_to_pixel(new_pts, np.eye(3))
        self.assertEqual(new_pts.shape[1], 3, "Homogeneous conversion is not same")
        
        new_pts = BasicConvertor.dehomogeneous(new_pts)        
        self.assertEqual(new_pts.shape[1], 2, "Homogeneous conversion is not same")
        
    def test_projector_v2(self):
        """Comparing the results of OpenCV and my implementation"""
        random_pts = np.random.rand(100, 3)
        
        tmp = TestCase1()
        cam = Camera(tmp["id"], 
                     tmp["intr_opt_type"], 
                     tmp["is_extr_opt"], 
                     tmp["proj_func_type"], 
                     tmp["dist_type"])
        cam.init_params(tmp["init_params"])
        
        cv_K = cam.K
        cv_R = cam.R
        cv_t = cam.t
        # cv_dist = np.array([[0.1, 0.1, 0., 0.]])
        cv_dist = np.array([[0.1, 0.1, 0., 0.]])
                
        # cv project points
        cv_pts, _ = cv2.projectPoints(random_pts, cv_R, cv_t, cv_K, cv_dist)
        
        # my project points
        my_pts = cam.project(random_pts)
        
        # for i in range(100):
        #     print(cv_pts[i], my_pts[i])
        
        self.assertAlmostEqual(np.sum(cv_pts - my_pts), 0, 4, "OpenCV and my implementation is not same")
        
    def test_generator(self):
        g = CameraGenerator()
        
        default_cameras = g.generate_default()

        for cam in default_cameras:
            print(cam.params)
        
        self.assertEqual(len(default_cameras), 4, "Default cameras are not same")