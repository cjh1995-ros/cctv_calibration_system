import unittest

from modules.markers.point import Feature2D, Feature3D
import numpy as np





class TestFeature2D(unittest.TestCase):
    def test_init(self):
        vec1 = Feature2D(1, 2)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)

    def test_init_v1(self):
        vec1 = Feature2D(1, 2)
        
        vec1.x = 3
        self.assertEqual(vec1.x, 3)
        self.assertEqual(vec1.y, 2)
        self.assertEqual(vec1.xy[0], 3)
        
        vec1.y = 4
        self.assertEqual(vec1.x, 3)
        self.assertEqual(vec1.y, 4)
        self.assertEqual(vec1.xy[1], 4)
        
        vec1.xy = np.array([5, 6], dtype=np.float32)
        self.assertEqual(vec1.x, 5)
        self.assertEqual(vec1.y, 6)
        self.assertEqual(vec1.xy[0], 5)
        self.assertEqual(vec1.xy[1], 6)

    def test_add(self):
        vec1 = Feature2D(1, 2)
        vec2 = Feature2D(3, 4)
        vec3 = vec1 + vec2
        self.assertEqual(vec3.x, 4)
        self.assertEqual(vec3.y, 6)
        
    def test_sub(self):
        vec1 = Feature2D(1, 2)
        vec2 = Feature2D(3, 4)
        vec3 = vec1 - vec2
        self.assertEqual(vec3.x, -2)
        self.assertEqual(vec3.y, -2)
        
    def test_mul(self):
        vec1 = Feature2D(1, 2)
        val = 3
        vec3 = vec1 * val
        self.assertEqual(vec3.x, 3)
        self.assertEqual(vec3.y, 6)
        
    def test_div(self):
        vec1 = Feature2D(1, 2)
        val = 3
        vec3 = vec1 / val
        self.assertAlmostEqual(vec3.x, 1/3)
        self.assertAlmostEqual(vec3.y, 2/3)

    def test_id(self):
        vec1 = Feature2D(1, 2)
        vec1.id = 1
        vec1.camera_id = 2
        self.assertEqual(vec1.id, 1)
        self.assertEqual(vec1.camera_id, 2)

class TestFeature3D(unittest.TestCase):
    def test_init(self):
        vec1 = Feature3D(1, 2, 3)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)
        self.assertEqual(vec1.z, 3)

    def test_add(self):
        vec1 = Feature3D(1, 2, 3)
        vec2 = Feature3D(3, 4, 5)
        vec3 = vec1 + vec2
        self.assertEqual(vec3.x, 4)
        self.assertEqual(vec3.y, 6)
        self.assertEqual(vec3.z, 8)
        
    def test_sub(self):
        vec1 = Feature3D(1, 2, 3)
        vec2 = Feature3D(3, 4, 5)
        vec3 = vec1 - vec2
        self.assertEqual(vec3.x, -2)
        self.assertEqual(vec3.y, -2)
        self.assertEqual(vec3.z, -2)
        
    def test_mul(self):
        vec1 = Feature3D(1, 2, 3)
        val = 3
        vec3 = vec1 * val
        self.assertEqual(vec3.x, 3)
        self.assertEqual(vec3.y, 6)
        self.assertEqual(vec3.z, 9)
        
    def test_div(self):
        vec1 = Feature3D(3, 3, 3)
        val = 3
        vec3 = vec1 / val
        self.assertAlmostEqual(vec3.x, 1)
        self.assertAlmostEqual(vec3.y, 1)
        self.assertAlmostEqual(vec3.y, 1)

if __name__ == '__main__':
    unittest.main()
