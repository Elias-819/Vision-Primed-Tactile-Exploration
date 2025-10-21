# Vision-Primed Tactile Exploration

## Project Overview

This project is based on **AcTExplore** and aims to improve 3D object exploration by combining vision and tactile feedback. The core idea of the project is to simulate the process of exploring an unknown object using **vision-guided tactile exploration**, which efficiently predicts the next touch points and refines the 3D object model by acquiring tactile data through force-controlled sliding and pressing.

The algorithm code is located in the `my_tools` folder, where key functionalities, including point cloud processing, exploration strategies, and coverage evaluation, are implemented.

---

## Vision-Primed Tactile Exploration: Algorithm Overview

This research, presented in the dissertation *"Vision-Primed Tactile Exploration: Projection-Guided Touch for 3D Reconstruction"* by Dongze Li, focuses on the challenges of **tactile exploration** in robotics, particularly in environments with occlusions or poor lighting. The key motivation is to bridge the gap between vision and tactile sensing by using **vision-guided tactile exploration** to enhance 3D object reconstruction.

### Key Objectives:
1. **Multimodal Exploration**: The integration of visual data with tactile feedback helps to improve exploration efficiency, particularly in poorly lit or occluded environments.
2. **Tactile Coverage Maximization**: The system aims to maximize the tactile sensor’s contact with the object's surface to gather sufficient data for 3D reconstruction.
3. **Next-Touch Prediction**: The algorithm predicts the next optimal touch point using multi-view point clouds, ensuring efficient exploration of unexplored regions.
4. **Force-Controlled Exploration**: The exploration is guided by force feedback, using **sliding** and **pressing** techniques to interact with the surface safely and accurately.

### Algorithm and Techniques:
- **Point Cloud Extraction**: The algorithm starts by extracting a sparse camera-view point cloud using **depth buffering** to retain only the closest point at each pixel. This helps to reduce occlusion issues in the environment.
  
- **Next-Touch Prediction**: By leveraging **multi-view convex-hull projections**, the system predicts the next touch points. This approach is based on detecting frontiers in the object’s surface and extrapolating boundary normals to suggest safe touch points.
  
- **Force-Safe Sliding and Pressing**: The sliding and pressing actions are force-controlled to ensure the robot maintains safe contact with the surface while acquiring tactile measurements. **Sliding** provides efficient early coverage, while **pressing** focuses on finer details.
  
- **Exploration Policy**: A **reinforcement learning (RL)** model, particularly using **PPO** (Proximal Policy Optimization), is used to decide the exploration strategy. The model is trained using tactile data combined with vision-based action guidance to explore unknown objects efficiently.

---

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Elias-819/Vision-Primed-Tactile-Exploration.git
cd Vision-Primed-Tactile-Exploration
pip install -r requirements.txt
```

## my_tools introduction

### cam_view_scale.py:
The cam_view_scale.py file simulates the camera view for point cloud data. It applies the concept of camera intrinsic and extrinsic parameters to project 3D points onto a 2D image plane. The script filters out points based on the camera's field of view (FOV) and region of interest (ROI), and performs depth buffering to select the closest point for each pixel. It is designed to simulate the process of a robot or camera system observing an object, extracting visible points, and saving the filtered data to a new point cloud file.
Key Functions

1.**load_xyz(path)**:
Loads point cloud data from a .txt file at the specified path.
Returns the points as a numpy array.

2.**save_pcd(pts, path)**:
Saves the given point cloud data (pts) to a .pcd file at the specified path.
The .pcd file format includes headers for point cloud data like size, type, and point count.

3.**simulate_camera_view(points, R, t, width, height, fx, fy, cx, cy, roi_px=None, roi_rel=None)**:
Simulates the camera view by projecting 3D points onto a 2D plane based on the given camera parameters (rotation R, translation t, focal lengths fx, fy, and center cx, cy).
It includes the option to apply a region of interest (ROI) to filter out points outside a specific area in the camera image. You can define the ROI either by pixel coordinates (roi_px) or relative proportions (roi_rel).
The function applies depth buffering to retain only the closest point for each pixel.
Returns the indices of the visible points after filtering.

4.**Main Execution**:
Loads the point cloud data from a .pcd file using load_xyz.
Defines the camera parameters, including position and orientation.
Simulates the camera view and filters the visible points using the simulate_camera_view function.
Displays the visible points in 3D and saves them to a new .pcd file.

### pcd_to_npy.py:
The pcd_to_npy.py script is designed to convert a PCD (Point Cloud Data) file into a NumPy array format. Point cloud data, often used in 3D vision tasks, is typically stored in .pcd files. This script reads the .pcd file, extracts the point cloud, converts it into a NumPy array, and saves the result in .npy format. This conversion is useful for processing and manipulating point cloud data in a more flexible format, especially for machine learning or other computational tasks.

### predict_Multi_projection.py:
The predict_Multi_projection.py file performs multi-view edge extrapolation and point cloud densification for improving the resolution and prediction of next touch points in the exploration process. This script is designed to generate predicted touch points for object exploration by considering multiple projections, estimating surface normals, and applying geometric extrapolation techniques.

#### Key Features
1.**Point Cloud Densification**:
The script uses a K-Nearest Neighbor (KNN) approach for local linear interpolation to densify the point cloud.
Additional points are inserted between neighboring points based on the KNN distance, improving the resolution and density of the point cloud.
The densification process is configurable with parameters such as the number of interpolation points (num_interp), the maximum distance for neighbors (max_dist), and the number of nearest neighbors (knn).

2.**Multi-Projection Edge Extrapolation**:
The script applies multi-view extrapolation using convex hull projections of the point cloud to predict future touch points.
The edge points of the point cloud are identified, and for each boundary point, the script generates a predicted point by extrapolating along the surface normal direction.
The surface normals are estimated using multi-scale nearest neighbor methods to improve robustness.

3.**Visualization**:
The script generates visualizations of the original points, newly added dense points, boundary points, predicted points, and normals.
The predicted points and their directions are visualized with arrows to illustrate the extrapolated paths.

4.**Prediction Output**:
The resulting predicted points are saved to a .pcd file.
Surface normals of the predicted points are saved to a .txt file.

### test_sliding.py:
The tset_sliding_first_edition.py script is designed to drive the tactile sensor movements in a robotic environment using multi-point predictions based on vision-guided exploration. The script integrates force control, reinforcement learning, and point cloud prediction to perform tactile exploration. It simulates the robot's interaction with an object, adjusting its movements to collect relevant data points for 3D object reconstruction.

#### Key Features
1.**Force Control with Pressure Feedback**:
The script uses a simple feedback loop to adjust the robot's movements based on force readings from the tactile sensor.
A force target is defined, and small adjustments are made to the depth of the sensor based on the difference between the measured force and the target force.

2.**Multi-Point Prediction**:
The script utilizes the multi_projection function from predict_Multi_projection.py to predict the next set of touch points. These predictions guide the robot's exploration path, determining where and how to move next.

3.**Path Planning and Sliding**:
The script defines a sequence of points for the robot to move through, based on the predicted touch points. The robot slides between these points, adjusting its depth and orientation based on force feedback.

4.**Environment Simulation with PyBullet**:
The environment is simulated using PyBullet, where the robot’s movements are executed. The tactile sensor's feedback is also recorded through this simulation, allowing for real-time adjustments and exploration.

5.**Logger for Tracking Progress**:
The script logs the robot's progress, including coverage, path length, and sensor feedback, saving the results to a CSV file for later analysis.

6.**Visualizations**:
The script optionally visualizes the exploration path in real-time using PyBullet’s visualization tools and matplotlib. It can visualize the sensor's path and depth map, helping to analyze the exploration process.
