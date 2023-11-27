import numpy as np
from dataclasses import dataclass, field
from gen_roadmarkers import gen_gps_data, gen_satellite_image_data, gen_lidar_data
import json
from copy import deepcopy



@dataclass
class BaseMarkers():
    # The variables you need to add
    gen_from:           str = 'GE' # image or gps
    gps_file:           str = 'data/markers/ETRI_44markers_ICTWAY+MMS.csv'
    markers_file:       str = 'data/markers/ETRI_44markers_satellite.json'
    slope_mask_file:    str = 'data/markers/slope_mask.json'
    origin_idx:         int = 23
    meter_per_pixel:    float = 0.07352941176
    
    # The variables you don't need to add
    gt_markers_dict:    dict = field(init=False)
    markers:            np.ndarray = field(init=False)
    train_idxs:         np.ndarray = field(init=False)
    valid_idxs:         np.ndarray = field(init=False)
    train_markers:      np.ndarray = field(init=False)
    valid_markers:      np.ndarray = field(init=False)

    def __post_init__(self):
        """
        if gen_from is image -> use markers from image
        if gen_from is gps   -> use markers from gps   
        """
        # init mask
        with open(self.slope_mask_file, 'r') as f:
            slope_tmp_dict = json.load(f)
        
        self.slope_mask = np.array(slope_tmp_dict['slope_mask'])
                
        # Gen dataset. -> init markers
        self.gen_dataset()

        self.gt_markers_dict = {}
        for i, marker in enumerate(self.markers):
            self.gt_markers_dict[i] = marker

        # init train markers
        self.train_markers = None


    def gen_dataset(self):
        """해당 모듈은 SfMA나 SfMP 모듈에서 수정할 수 있도록 __post_init__으로 부터 분리되었다"""
        # gen image dataset
        if self.gen_from == 'MP':
            self.markers = gen_satellite_image_data(marker_file=self.markers_file,
                                                    origin_idx=self.origin_idx,
                                                    meter_per_pixel=self.meter_per_pixel)

        # gen google earth dataset
        elif self.gen_from == 'GE':
            self.markers = gen_gps_data(file=self.gps_file,
                                        origin_idx=self.origin_idx)

        else:
            raise Exception("Not proper mode")

        # init meter per pixel
        self.gen_meter_per_pixel()


    def gen_meter_per_pixel(self):
        """
        만든 이유: 이미지를 3차원 점으로 바꿀 때는 미리 계산한 결과가 존재함. 그런데
        다른 데이터셋은 적용이 안될 수 있음. 3차원 점의 위치들과 픽셀의 위치
        를 알고 있음으로 이를 이용해서 
        ** 각 데이터셋의 meter per pixel을 만드는 것이 목표 **
        
        필요한 것
            - 이미지 상 픽셀
            - 3차원 상의 점
        """
        # get satellite pixels
        with open(self.markers_file, 'r') as f:
            tmp_dict = json.load(f)
        
        self.satellite_marker_pixels = np.array(tmp_dict['pts'], dtype=np.float32)
        
        # to standard coord
        pixels = deepcopy(self.satellite_marker_pixels)
        pixels -= pixels[self.origin_idx]

        if self.gen_from == 'MP':
            self.meter_per_pixel = 0.06172839506172839
        
        elif self.gen_from == 'GE' or self.gen_from == 'lidar':
            pts_x, pts_y = self.markers[:, 0], self.markers[:, 1]
            pixels_x, pixels_y = pixels[:, 0], pixels[:, 1]

            self.meter_per_pixel = np.zeros((2, ), dtype=np.float32)

            self.meter_per_pixel[0] = np.sum(np.abs(pts_x)) / np.sum(np.abs(pixels_x))
            self.meter_per_pixel[1] = np.sum(np.abs(pts_y)) / np.sum(np.abs(pixels_y))


    def prepare_params4opt(self):
        if self.train_markers is None:
            return self.markers.flatten()
        else :
            return self.train_markers.flatten()

    def set_params4opt(self, params: np.ndarray):
        if self.train_markers is None:
            self.markers = params.reshape(self.markers.shape)
        else :
            self.train_markers = params.reshape(self.train_markers.shape)

    def gen_bounds(self, xlim: float = 2., ylim: float = 2., zlim: float = 20.):
        """최대 2m 정도만 움직이는 것을 허용함"""
        tmp_standard = deepcopy(self.prepare_params4opt()).reshape(-1, 3)

        up_bounds   = tmp_standard + np.array([xlim, ylim, zlim])
        down_bounds = tmp_standard - np.array([xlim, ylim, zlim])

        return np.vstack((down_bounds.flatten(), up_bounds.flatten())).T

    def to_standard_coord(self):
        self.markers -= self.markers[self.origin_idx]

    def calc_rmse_with_lidar(self):
        # load lidar data
        lidar_data = gen_lidar_data()

        # calc rmse
        mse = (lidar_data - self.markers) ** 2
        return np.sqrt(mse.mean())

        
    def update_valid_roadmarkers(self, mode: int = -1):
        if mode == -1:
            raise Exception("Not proper mode")

        for valid_idx, train_idx in zip(self.valid_idxs, self.train_idxs):
            contours = self.markers[train_idx]
            new_marker = self.markers[valid_idx]

            if mode == 0:
                new_marker[0] = contours[0][0] - (contours[2][0] - contours[1][0])
                new_marker[1] = contours[1][1] + (contours[0][1] - contours[2][1])
                new_marker[2] = np.mean(contours[:, 2])

            elif mode == 1:
                new_marker[0] = contours[0][0] + (contours[2][0] - contours[1][0])
                new_marker[1] = contours[2][1] + (contours[0][1] - contours[1][1])
                new_marker[2] = np.mean(contours[:, 2])

            elif mode == 2:
                new_marker[0] = contours[2][0] - (contours[1][0] - contours[0][0])
                new_marker[1] = contours[0][1] - (contours[1][1] - contours[2][1])
                new_marker[2] = np.mean(contours[:, 2])


    def update_marker_after_opt(self):
        """
            만든 이유!
            최적화 끝난 후 파라미터들을 원래의 형태로 돌려놓기 위해 만들어 놓은 것.
            하지만 최적화를 끝낸 것은 train_markers이다. 그렇기에 markers에 
            최적화된 결과를 삽입하고 이를 바탕으로 valid markers를 생성한다.
            이후 markers를 return
        """
        # update markers with trained data
        for t_val, t_idx in zip(self.train_markers, self.train_idxs.flatten()):
            self.markers[t_idx] = t_val



@dataclass
class NoOpt(BaseMarkers):
    """Not optimizing road markers"""
    def prepare_params4opt(self):
        return None


    def set_params4opt(self, params: np.ndarray):
        pass


    def gen_bounds(self):
        return None

@dataclass
class SfMA(BaseMarkers):
    pass



@dataclass
class SfMP(BaseMarkers):
    theta: float = 0.0

    def prepare_params4opt(self):
        return self.theta

    def set_params4opt(self, params: np.ndarray):
        # update theta
        self.theta = deepcopy(params[-1])

        if self.train_markers is None:
            # get base points from markers
            base_pts = self.markers[np.where(self.slope_mask == 1)]
            pts = self.markers[np.where(self.slope_mask == 2)]

            # calc distances between base points and points
            distances = self.calc_distance(base_pts, pts)
            zs = np.tan(self.theta) * distances

            # update z values
            self.markers[:, 2][np.where(self.slope_mask == 2)] = deepcopy(zs)

        else:
            # get base points from markers
            base_pts = self.train_markers[np.where(self.slope_mask == 1)]
            pts = self.train_markers[np.where(self.slope_mask == 2)]

            # calc distances between base points and points
            distances = self.calc_distance(base_pts, pts)
            zs = np.tan(self.theta) * distances

            # update z values
            self.train_markers[:, 2][np.where(self.slope_mask == 2)] = deepcopy(zs)


    def gen_bounds(self, tlim: float = np.pi / 4.):
        tmp_standard = deepcopy(self.prepare_params4opt())

        up_bounds   = tmp_standard + tlim
        down_bounds = tmp_standard - tlim

        return np.vstack((down_bounds, up_bounds)).T
        

    def calc_distance(self, base: np.ndarray, pts: np.ndarray, data_type = np.float64) -> np.ndarray:
        # if base pts is one -> do something!
        if len(base) == 2:            
            xz_0 = np.array([base[0][0], base[0][1], 1], dtype=data_type)
            xz_1 = np.array([base[1][0], base[1][1], 1], dtype=data_type)
            line_coeff = np.cross(xz_0, xz_1)

        elif len(base) == 1:
            line_coeff = np.array([1, 0, -1], dtype=np.float32)

        ab_square = np.linalg.norm(line_coeff[:2])
        distances = 1. / ab_square * np.array([np.abs(line_coeff[0] * pt[0] + line_coeff[1] * pt[2] + line_coeff[2]) for pt in pts])
        return distances

@dataclass
class SfMZ(BaseMarkers):
    def prepare_params4opt(self):
        if self.train_markers is None:
            return self.markers[:, 2].flatten()
        else:
            return self.train_markers[:, 2].flatten()


    def set_params4opt(self, params: np.ndarray):
        if self.train_markers is None:
            self.markers[:, 2] = deepcopy(params)
        else:
            self.train_markers[:, 2] = deepcopy(params)
        

    def gen_bounds(self, zlim: float = 2.):
        tmp_standard = deepcopy(self.prepare_params4opt())

        up_bounds   = tmp_standard + zlim
        down_bounds = tmp_standard - zlim

        return np.vstack((down_bounds, up_bounds)).T



if __name__ == '__main__':
    # 1. ETRI case
    meter_per_pixel = 0.06172839506172839

    # 2. SeoulTech case
    meter_per_pixel = 0.07352941176

    sfmz = SfMZ(
        gen_from='MP',
        markers_file='data/SeoulTech/markers/SeoulTech_satellite.json',
        slope_mask_file='data/SeoulTech/markers/SeoulTech_mask.json',
        origin_idx=16,
        # theta=1e-2,
        meter_per_pixel=meter_per_pixel,
    )
    zbounds = sfmz.gen_bounds()
    print(zbounds)
