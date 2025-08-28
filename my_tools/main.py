import numpy as np
import open3d as o3d
import os
import omegaconf
import matplotlib
# 强制使用支持交互的后端，避免 Agg 无窗口模式
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Activate 3D projection

import pybullet as p
import pybulletX as px
from scipy.spatial.transform import Rotation as R

# 初始化点云可视化窗口
plt.ion()
fig = plt.figure('PointCloud Viewer')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('灰色：历史点云   蓝色：新更新点云')
# 初始化空数据
old_pts = np.empty((0,3))
new_pts = np.empty((0,3))
sc_old = ax.scatter(old_pts[:,0], old_pts[:,1], old_pts[:,2], c='gray', s=1, depthshade=False)
sc_new = ax.scatter(new_pts[:,0], new_pts[:,1], new_pts[:,2], c='blue', s=5, depthshade=False)
# 非阻塞显示窗口
plt.show(block=False)
if p.getConnectionInfo()['isConnected'] == 0:
    px.init(mode=p.GUI)

from env import TactoEnv
from stable_baselines3 import PPO
from infer_actexplore import probe_until_touch

# 读取点云
def load_pointcloud(path):
    if path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    else:
        raise ValueError('只支持 .pcd 格式！')

# 保存点云
def save_pointcloud(points, path):
    if path.endswith('.pcd'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path, pcd)
    else:
        raise ValueError('只支持 .pcd 格式！')

# 合并点云
def merge_pointcloud(points1, points2):
    return np.vstack([points1, points2])

# 计算覆盖率
def calc_coverage(points, workspace, voxel_size=0.005):
    min_bound, max_bound = workspace
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    inside_points = points[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inside_points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    occupied_voxels = len(voxel_grid.get_voxels())
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    total_voxels = np.prod(grid_shape)
    return occupied_voxels / total_voxels

# 调用预测脚本
def run_predict_script(input_path, output_path):
    import subprocess
    subprocess.run([
        'python', 'my_tools/predict_Multi_projection.py',
        '--input', input_path,
        '--output', output_path
    ], check=True)
    real_output = 'output/predict.pcd'
    if not os.path.exists(output_path) and os.path.exists(real_output):
        import shutil
        shutil.copy(real_output, output_path)
    return load_pointcloud(output_path)

# 选择下一个触碰点
def choose_next_point(current_path, predict_path, output_path, blacklist_path):
    import subprocess
    subprocess.run([
        'python', 'my_tools/choose_next_touch_point.py',
        '--current_pointcloud', current_path,
        '--predict_points', predict_path,
        '--output_path', output_path,
        '--blacklist', blacklist_path,
        '--blacklist_dist', '0.003'
    ], check=True)
    pcd = o3d.io.read_point_cloud(output_path)
    arr = np.asarray(pcd.points)
    if arr.shape[0] == 0:
        raise RuntimeError(f"选点脚本没有生成点，输出文件为空：{output_path}")
    idx_path = output_path.replace('.pcd', '_idx.txt')
    if os.path.exists(idx_path):
        idx = int(np.loadtxt(idx_path))
    else:
        idx = 0
    next_point = arr[0]
    return next_point, idx

# 模拟运动
def move_to_point(point):
    print(f"模拟运动到：{point}")

# 法线转四元数
def normal_to_quat(normal):
    r = R.align_vectors([[0,0,1]], [normal])[0]
    return r.as_quat()

# 采样触碰点
def sample_touch_points_with_probe(model, env, target_point, quat, num_steps=20, log_func=print):
    obs = env.reset_to_point(target_point, quat)
    touch_ok = probe_until_touch(env, target_point, normal=None)
    if not touch_ok:
        log_func(f"目标点 {target_point.tolist()} 未碰到物体，自动放弃。")
        return np.zeros((0,3))
    log_func(f"目标点 {target_point.tolist()} 试探成功，开始RL探索。")
    collected_points = []
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if isinstance(action, np.ndarray) else action
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, done, _, info = step_result
        else:
            obs, reward, done, info = step_result
        pts_this_step = info.get("new_touch_points")
        if pts_this_step is None and hasattr(env, "pointcloud"):
            pts_this_step = np.array(env.pointcloud)
        if isinstance(pts_this_step, np.ndarray) and pts_this_step.size>0:
            pts_this_step = pts_this_step.reshape(-1,3)
            collected_points.append(pts_this_step)
        if done:
            break
    return np.vstack(collected_points) if collected_points else np.zeros((0,3))

# 主函数
def main():
    workspace = np.array([[0.0,0.0,0.0],[0.2,0.2,0.2]])
    coverage_threshold = 0.95
    model_path = 'Training/Logs/PPO_Contact_AMB_3/model_100000_steps.zip'
    cfg = omegaconf.OmegaConf.load("conf/RL.yaml")
    init_pointcloud_path = 'output/visible_from_camera.pcd'
    if not os.path.exists(init_pointcloud_path):
        raise FileNotFoundError(f"初始点云文件不存在: {init_pointcloud_path}")
    current_points = load_pointcloud(init_pointcloud_path)
    iteration = 0
    env = TactoEnv(cfg)
    model = PPO.load(model_path, env)
    used_target_points = []
    blacklist_path = 'output/blacklist.txt'
    if not os.path.exists(blacklist_path):
        np.savetxt(blacklist_path, np.zeros((0,3)), fmt="%.6f")
    Z_OFFSET = 0.05
    predict_normals_path = 'output/predict_normals.txt'
    while True:
        iteration += 1
        print(f"\n======= 第 {iteration} 轮探索 =======")
        current_path = f'output/current.pcd'
        save_pointcloud(current_points, current_path)
        predict_points = run_predict_script(current_path, f'output/predict.pcd')
        predict_normals = np.loadtxt(predict_normals_path)
        np.savetxt(blacklist_path, np.array(used_target_points), fmt="%.6f")
        next_point, next_point_idx = choose_next_point(current_path, 'output/predict.pcd', f'output/next_point.pcd', blacklist_path)
        move_to_point(next_point)
        quat = normal_to_quat(predict_normals[next_point_idx])
        world_next_point = next_point + np.array([0,0,Z_OFFSET])
        new_points = sample_touch_points_with_probe(model, env, world_next_point, quat, num_steps=20)
        print(f"本轮采集新点数: {new_points.shape}")
        if new_points.shape[0] < 3:
            used_target_points.append(next_point)
            continue
        # 更新可视化
        old_pts = current_points.copy()
        new_pts = new_points.copy()
        sc_old._offsets3d = (old_pts[:,0], old_pts[:,1], old_pts[:,2])
        sc_new._offsets3d = (new_pts[:,0], new_pts[:,1], new_pts[:,2])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)  # 短暂暂停以处理 GUI 事件
        current_points = merge_pointcloud(current_points, new_points)
        print(f"当前点云总点数: {current_points.shape}")
        coverage = calc_coverage(current_points, workspace)
        print(f"当前点云覆盖率：{coverage:.3f}")
        save_pointcloud(current_points, f'output/step_{iteration}_all.pcd')
        if coverage > coverage_threshold:
            break
    save_pointcloud(current_points, 'output/final_pointcloud.pcd')
    print("流程结束，最终点云已保存。")

if __name__ == '__main__':
    main()
