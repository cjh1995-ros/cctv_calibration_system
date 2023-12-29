from test_data_generate_bed import test_cameras, POINT
import json
import numpy as np


if __name__ == '__main__':
    numbering = 3
    
    cameras = test_cameras()
    markers = POINT[f"case{numbering}"]()
    
    # Save camera params
    camera_params = []
    for cam in cameras:
        camera_params.append(cam.params)
        
    camera_params = np.array(camera_params).reshape(-1, 9)
    np.savetxt(f"camera_params_{numbering}.txt", camera_params, delimiter=", ")
    
    # Add noise to camera params
    noisy_params = np.copy(camera_params)
    for cam_param in noisy_params:
        cam_param[0]    += np.random.normal(0, 50)
        cam_param[1:3]  += np.random.normal(0, 0.1, 2)
        cam_param[3:]   += np.random.normal(0, 0.05, 6)
    np.savetxt(f"noisy_camera_params_{numbering}.txt", noisy_params, delimiter=", ")
    
    # Save GT markers
    # markers = np.array(markers).reshape(-1, 3)
    np.savetxt(f"gt_markers_{numbering}.txt", markers, delimiter=", ")
    
    # Save Noisy markers. 
    # For xy-axis, Add small noise
    # For z-axis, Add noise to only slope
    noisy_markers = np.copy(markers)
    noisy_markers[:, :2] += np.random.normal(0, 0.005, markers[:, :2].shape)
    noisy_markers[8:, 2] += np.random.normal(0, 0.05, markers[8:, 2].shape)
    np.savetxt(f"noisy_markers_{numbering}.txt", noisy_markers, delimiter=", ")
    
    # Save GT pixels as json
    gt_pixels_per_camera = []
    
    for i, cam in enumerate(cameras):
        idx_pixels = {"file": f"synthetic_camera_{i}.txt",
                      "idx_pixels": []}
        
        pixels = cam.project(markers)

        for i, pixel in enumerate(pixels):
            idx_pixels["idx_pixels"].append([i, pixel[0], pixel[1]])        
        
        gt_pixels_per_camera.append(idx_pixels)
    
    with open(f"markers_images_{numbering}.json", "w") as f:
        json.dump(gt_pixels_per_camera, f, indent=4)
        
    