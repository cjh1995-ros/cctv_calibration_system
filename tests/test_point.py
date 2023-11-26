import unittest
from markers.point import Feature2D, Feature3D

class TestFeature2D(unittest.TestCase):
    def test_init(self):
        vec1 = Feature2D(1, 2)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)

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
        self.assertEqual(vec3.x, 1/3)
        self.assertEqual(vec3.y, 2/3)

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
        self.assertEqual(vec3.x, 1)
        self.assertEqual(vec3.y, 1)
        self.assertEqual(vec3.y, 1)

if __name__ == '__main__':
    unittest.main()
