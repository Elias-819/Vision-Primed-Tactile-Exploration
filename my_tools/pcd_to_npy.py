# 将当前的部分3D点云从 PCD 文件转换为 NumPy 数组并保存
# scripts/pcd2npy.py
import open3d as o3d
import numpy as np

# 1. 读取 PCD
pcd = o3d.io.read_point_cloud("H:\projects\AcTExplore\output\cam/paitical_visible_from_camera.pcd")

# 3. 转为 numpy array
pts = np.asarray(pcd.points).astype(np.float32)  # shape: (1024, 3)

# 4. 保存
np.save("H:\projects\AcTExplore\output\cam/paitical_visible_from_camera.npy", pts)
print("Saved partial cloud: paitical_visible_from_camera.npy")
