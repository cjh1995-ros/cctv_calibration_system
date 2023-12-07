import unittest
from modules.markers import Vector2D, Vector3D

class TestVector2D(unittest.TestCase):
    def test_init(self):
        vec1 = Vector2D(1, 2)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)

    def test_add(self):
        vec1 = Vector2D(1, 2)
        vec2 = Vector2D(3, 4)
        vec3 = vec1 + vec2
        self.assertEqual(vec3.x, 4)
        self.assertEqual(vec3.y, 6)
        
    def test_sub(self):
        vec1 = Vector2D(1, 2)
        vec2 = Vector2D(3, 4)
        vec3 = vec1 - vec2
        self.assertEqual(vec3.x, -2)
        self.assertEqual(vec3.y, -2)
        
    def test_mul(self):
        vec1 = Vector2D(1, 2)
        val = 3
        vec3 = vec1 * val
        self.assertEqual(vec3.x, 3)
        self.assertEqual(vec3.y, 6)
        
    def test_div(self):
        vec1 = Vector2D(1, 2)
        val = 3
        vec3 = vec1 / val
        self.assertEqual(vec3.x, 1/3)
        self.assertEqual(vec3.y, 2/3)

class TestVector3D(unittest.TestCase):
    def test_init(self):
        vec1 = Vector3D(1, 2, 3)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)
        self.assertEqual(vec1.z, 3)

    def test_add(self):
        vec1 = Vector3D(1, 2, 3)
        vec2 = Vector3D(3, 4, 5)
        vec3 = vec1 + vec2
        self.assertEqual(vec3.x, 4)
        self.assertEqual(vec3.y, 6)
        self.assertEqual(vec3.z, 8)
        
    def test_sub(self):
        vec1 = Vector3D(1, 2, 3)
        vec2 = Vector3D(3, 4, 5)
        vec3 = vec1 - vec2
        self.assertEqual(vec3.x, -2)
        self.assertEqual(vec3.y, -2)
        self.assertEqual(vec3.z, -2)
        
    def test_mul(self):
        vec1 = Vector3D(1, 2, 3)
        val = 3
        vec3 = vec1 * val
        self.assertEqual(vec3.x, 3)
        self.assertEqual(vec3.y, 6)
        self.assertEqual(vec3.z, 9)
        
    def test_div(self):
        vec1 = Vector3D(3, 3, 3)
        val = 3
        vec3 = vec1 / val
        self.assertEqual(vec3.x, 1)
        self.assertEqual(vec3.y, 1)
        self.assertEqual(vec3.y, 1)

    def test_noisy(self):
        vec1 = Vector3D(1, 2, 3)
        sigma = 0.1
        
        vec1.noisy(sigma)
        self.assertEqual(vec1.x, 1)
        self.assertEqual(vec1.y, 2)
        self.assertEqual(vec1.z, 3)

if __name__ == '__main__':
    unittest.main()
