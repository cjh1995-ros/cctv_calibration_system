from contextlib import AbstractContextManager
from typing import Any
import unittest
from modules.board import Chessboard
from modules.frame import BasicFrame

import numpy as np
import cv2

    
class TestCamera(unittest.TestCase):
    def test_image_init(self):
        # Test 
        image = BasicFrame()
        
        image.path = "data/frame0000.png"
        image.data = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
        
        # self.assertIsNone(image.corners)
        self.assertIsNotNone(image.data)
        
    def test_board_pattern(self):
        pattern = (6, 9)
        size = (0.05, 0.05)
        
        board = Chessboard()
        board.pattern = pattern
        board.size = size
        
        self.assertEqual(board.pattern, pattern)
        self.assertEqual(board.size, size)
        self.assertEqual(len(board.points), np.prod(pattern), "Length of points are different from pattern.")