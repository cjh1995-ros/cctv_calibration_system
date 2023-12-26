from modules.frame import BasicFrame
import logging
import numpy as np
import cv2











class BaseBoard:
    """
    Purpose: 
        - Find the corners of the board in the image.
        - Calculate the location of 3d points in the board.
        - Initial guess of the camera extrinsic parameters.
    """    
    def __init__(self):
        self._pattern = None
        self._size = None
        self._points = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configure_logger()

    def configure_logger(self):
        # 로거의 기본 설정: 로그 레벨, 포맷, 핸들러 등을 설정
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(f'.log/{self.__class__.__name__.lower()}.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    
    @property
    def pattern(self):
        """pattern of the board."""
        return self._pattern
    
    @property
    def size(self):
        """size of the board. should be (height, width)"""
        return self._size
    
    @property
    def points(self):
        return self._points    

    
    @pattern.setter
    def pattern(self, pattern: tuple):
        """Reserve points container"""
        assert len(pattern) == 2 and pattern[0] > 0 and pattern[1] > 0, "Invalid pattern. Must be (rows, cols)."
        assert isinstance(pattern, tuple) or isinstance(pattern, list), "Invalid type. Must be tuple or list."

        self._pattern = pattern
        self._size = np.prod(pattern)
        self._points = np.zeros((self._size, 3), dtype=np.float32)
        
        self.logger.debug(f"Pattern is set to {self._pattern}")
        self.logger.debug(f"Size is set to {self._size}")
        self.logger.debug(f"Number of points is set to {len(self._points)}")
        

    @size.setter
    def size(self, size: tuple):
        """Get width and height of board and set 3d points"""
        assert len(size) == 2 and size[0] > 0 and size[1] > 0, "Invalid size. Must be (rows, cols)."
        assert isinstance(size, tuple) or isinstance(size, list), "Invalid type. Must be tuple or list."
        
        self._size = size
        
        self.logger.debug(f"Size is set to {self._size}")
        
        x_pattern = self._pattern[1]
        y_pattern = self._pattern[0]
        
        x = np.linspace(0, self._size[1] * (x_pattern - 1), x_pattern, dtype=np.float32)
        y = np.linspace(0, self._size[0] * (y_pattern - 1), y_pattern, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        self._points[:, :2] = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        self.logger.debug("Points are set through size.")

        
    def find_corners(self, image: BasicFrame) -> (bool, np.ndarray):
        """Find the corners of the board in the image."""
        ...




class Chessboard(BaseBoard):
    def __init__(self):
        super().__init__()
        
    def find_corners(self, image: BasicFrame) -> (bool, np.ndarray):
        """Find the corners of the board in the image."""
        assert self._pattern is not None, "Pattern is not set."
        assert self._size is not None, "Size is not set."
        
        found, corners = cv2.findChessboardCorners(image.data, self._pattern)
        
        if not found:
            self.logger.error("Chessboard corners are not found.")
            return False, None
        
        return found, corners.reshape(-1, 2)