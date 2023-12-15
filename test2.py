from modules.gui.viz_matplotlib import MatplotVisualizer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from copy import deepcopy
# from autograd import numpy as np
import numpy as np



history = []


def project(pts:np.ndarray, intrinsic:np.ndarray, extrinsic:np.ndarray):
    fx,fy,cx,cy,k1,k2 = intrinsic
    rvec,tvec = extrinsic[:3],extrinsic[3:]
    
    Rmat = R.from_rotvec(rvec).as_matrix()
    
    new_pts = pts @ Rmat.T + tvec
    
    new_pts = new_pts[:,:2] / new_pts[:,2:]
    
    r2 = new_pts[:,0]**2 + new_pts[:,1]**2
    
    new_pts = new_pts * (1 + k1*r2 + k2*r2**2)[:,None]
    
    new_pts[:,0] = new_pts[:,0] * fx + cx
    new_pts[:,1] = new_pts[:,1] * fy + cy

    return new_pts

def calc_z(theta, pts, line_pts1, line_pts2):
    line_vec = line_pts2 - line_pts1
    pt_vec = pts - line_pts1
    
    pt_vec = pt_vec / np.linalg.norm(pt_vec, axis=2).reshape(3, 2, 1)
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    dot = pt_vec @ line_vec.T
    
    angle = np.arccos(dot)
    
    return np.sin(angle) * np.linalg.norm(pts - line_pts1, axis=2) * np.tan(theta)


def object_function_01(params, gt_pixels, intrinsics, extrinsics, xy):
    global history
    
    history.append(params)
    
    theta = params[0]
    
    ### Make xyz points from theta and xy
    # Init xyz
    ones = np.ones_like(xy[:, :, 0])
    xy1 = np.stack([xy[:, :, 0], xy[:, :, 1], ones], axis=-1)
    
    # Calc z
    line_pts1 = np.array([2.4, 2.0, 1.0])
    line_pts2 = np.array([2.4, 2.2, 1.0])
    
    zs = calc_z(theta, xy1[:, 3:, :], line_pts1, line_pts2)
    
    # Make new xyz with theta
    new_xyz = deepcopy(xy1)
    new_xyz[:, 3:, 2] += zs
    
    
    ### Project new xyz
    new_xyz = new_xyz.reshape(-1, 3)
    
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(new_xyz, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error



def object_function_02(params, gt_pixels, extrinsics, xy):
    global history
    
    history.append(params)
    
    
    intrinsics = params[:-1].reshape(-1, 6)
    theta = params[-1]
    
    ### Make xyz points from theta and xy
    # Init xyz
    ones = np.ones_like(xy[:, :, 0])
    xy1 = np.stack([xy[:, :, 0], xy[:, :, 1], ones], axis=-1)
    
    # Calc z
    line_pts1 = np.array([2.4, 2.0, 1.0])
    line_pts2 = np.array([2.4, 2.2, 1.0])
    
    zs = calc_z(theta, xy1[:, 3:, :], line_pts1, line_pts2)
    
    # Make new xyz with theta
    new_xyz = deepcopy(xy1)
    new_xyz[:, 3:, 2] += zs
    
    
    ### Project new xyz
    new_xyz = new_xyz.reshape(-1, 3)
    
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(new_xyz, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error

def object_function_025(params, gt_pixels, xy):
    global history
    
    history.append(params)
    
    camera_params = params[:48].reshape(-1, 12)
    intrinsics = camera_params[:, :6]
    extrinsics = camera_params[:, 6:]
    theta = params[-1]
    
    ### Make xyz points from theta and xy
    # Init xyz
    ones = np.ones_like(xy[:, :, 0])
    xy1 = np.stack([xy[:, :, 0], xy[:, :, 1], ones], axis=-1)
    
    # Calc z
    line_pts1 = np.array([2.4, 2.0, 1.0])
    line_pts2 = np.array([2.4, 2.2, 1.0])
    
    zs = calc_z(theta, xy1[:, 3:, :], line_pts1, line_pts2)
    
    # Make new xyz with theta
    new_xyz = deepcopy(xy1)
    new_xyz[:, 3:, 2] += zs
    
    
    ### Project new xyz
    new_xyz = new_xyz.reshape(-1, 3)
    
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(new_xyz, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error

def object_function_03(params, gt_pixels, extrinsics):
    global history
    
    history.append(params)


    intrinsics = params[:24].reshape(-1, 6)
    points = params[24:].reshape(-1, 3)    
        
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error


def object_function_04(params, gt_pixels):
    global history
    
    history.append(params)
    
    
    camera_params = params[:48].reshape(-1, 12)
    intrinsics = camera_params[:, :6]
    extrinsics = camera_params[:, 6:]
    
    points = params[48:].reshape(-1, 3)    
        
    new_pixels = []
    
    for intr, extr in zip(intrinsics, extrinsics):
        new_pixels.append(project(points, intr, extr))
        
    new_pixels = np.concatenate(new_pixels, axis=0)
    
    ### Calculate error
    error = (new_pixels - gt_pixels).ravel()
    
    return error

if __name__ == '__main__':
    # Parameters
    rows = 3
    cols = 5
    distance = 0.2

    # Create 3d points
    x = np.linspace(0, (cols-1)*distance, cols) + 2.
    y = np.linspace(0, (rows-1)*distance, rows) + 2.

    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)

    xy = np.stack([xx, yy], axis=-1).reshape(-1, 2).reshape(rows, cols, 2)

    xyz0 = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).reshape(rows, cols, 3)

    new_zs = np.array([0.0, 0.2, 0.4])

    xyz_theta = deepcopy(xyz0)
    xyz_theta[:, 2, 2] += new_zs[0]
    xyz_theta[:, 3, 2] += new_zs[1]
    xyz_theta[:, 4, 2] += new_zs[2]


    # Create synthetic camera parameters
    intrinsic1 = np.array([1000., 1000., 500., 500., 0.1, 0.1])
    extrinsic1 = np.array([0., 0., 0., -1., -1., 1.0])

    intrinsic2 = np.array([1000., 1000., 500., 500., 0.1, 0.1])
    extrinsic2 = np.array([0., 0., 0., -1., -1., 2.0])

    intrinsic3 = np.array([1000., 1000., 500., 500., 0.1, 0.1])
    extrinsic3 = np.array([0., 0., 0., -1., -1., 3.0])

    intrinsic4 = np.array([1000., 1000., 500., 500., 0.1, 0.1])
    extrinsic4 = np.array([0., 0., 0., -1., -1., 4.0])

    intrs = [intrinsic1, intrinsic2, intrinsic3, intrinsic4]
    extrs = [extrinsic1, extrinsic2, extrinsic3, extrinsic3]

    # Create ground truth pixels
    gt_pixels_01 = project(xyz_theta.reshape(-1, 3), intrinsic1, extrinsic1)
    gt_pixels_02 = project(xyz_theta.reshape(-1, 3), intrinsic2, extrinsic2)
    gt_pixels_03 = project(xyz_theta.reshape(-1, 3), intrinsic3, extrinsic3)
    gt_pixels_04 = project(xyz_theta.reshape(-1, 3), intrinsic4, extrinsic4)

    gts = np.concatenate([gt_pixels_01, gt_pixels_02, gt_pixels_03, gt_pixels_04], axis=0)

    # Add noise to camera parameters
    init_intrs = []

    for intr in intrs:
        noise = np.random.rand(6)
        noise[:4] *= 100
        noise[4:] *= 0.1
        init_intrs.extend(intr + noise)

    init_intrs = np.array(init_intrs)

    init_extrs = []

    for extr in extrs:
        noise = np.random.rand(6) * 0.1
        init_extrs.extend(extr + noise)

    init_extrs = np.array(init_extrs)

    ### Prepare initial parameters
    init_intrs_extrs = np.zeros((4, 12), dtype=np.float32)

    for i, (intr, extr) in enumerate(zip(init_intrs.reshape(-1, 6), init_extrs.reshape(-1, 6))):
        init_intrs_extrs[i] = np.concatenate([intr, extr], axis=0)

    noisy_xyz = xyz_theta + np.random.rand(*xyz_theta.shape) * 0.1

    init_theta = 0.0

    # Case 1: Just theta
    # inits = init_theta
    # res = least_squares(object_function_01, inits, verbose=2, args=(gts, intrs, extrs, xy))
    
    # Case 2: Intrinsic and theta
    # inits = np.concatenate([init_intrs, [init_theta]], axis=0)
    # res = least_squares(object_function_02, inits, verbose=2, args=(gts, extrs, xy))

    # Case 2.5: Intrinsic + Extrinsic + theta
    # noisy_xy = xy + np.random.rand(*xy.shape) * 0.0005
    # inits = np.concatenate([init_intrs_extrs.ravel(), [init_theta]], axis=0)
    # res = least_squares(object_function_025, inits, verbose=2, args=(gts, xy))
    
    # Case 3: Intrinsic + 3D points
    # inits = np.concatenate([init_intrs, noisy_xyz.ravel()], axis=0)
    # res = least_squares(object_function_03, inits, verbose=2, args=(gts, extrs))
    
    # Case 4: Intrinsic + Extrinsic + 3D Points
    inits = np.concatenate([init_intrs_extrs.ravel(), noisy_xyz.ravel()], axis=0)
    res = least_squares(object_function_04, inits, verbose=2, args=(gts,))
    
    camera_params = res.x[:48].reshape(-1, 12)
    intrinsics = camera_params[:, :6]
    extrinsics = camera_params[:, 6:]
    
    mean_intr = np.mean(intrinsics, axis=0)
    mean_extr = np.mean(extrinsics, axis=0)
    
    print("Mean intrinsics")
    print(mean_intr)
    
    for i, (intr, extr) in enumerate(zip(intrinsics, extrinsics)):
        print(f"Camera {i}")
        print(f"Intrinsics: {intr}")
        print(f"Extrinsics: {extr}")
        print()
        
    # Residual between points and gt markers
    # theta = res.x[-1]
    
    # print(theta)
    
    # ### Make xyz points from theta and xy
    # ones = np.ones_like(xy[:, :, 0])
    # xy1 = np.stack([xy[:, :, 0], xy[:, :, 1], ones], axis=-1)
    
    # # Calc z
    # line_pts1 = np.array([2.4, 2.0, 1.0])
    # line_pts2 = np.array([2.4, 2.2, 1.0])
    
    # zs = calc_z(theta, xy1[:, 3:, :], line_pts1, line_pts2)
    
    # # Make new xyz with theta
    # new_xyz = deepcopy(xy1)
    # new_xyz[:, 3:, 2] += zs
    
    # new_xyz = new_xyz.reshape(-1, 3)
    
    new_xyz = res.x[48:].reshape(-1, 3)
    print("Residual between points and gt markers")
    print(np.linalg.norm(new_xyz - xyz_theta.reshape(-1, 3)))
    