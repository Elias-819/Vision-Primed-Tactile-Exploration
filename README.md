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

---

### my_tools introduction


