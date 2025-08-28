# my_tools/active_refinement.py
# -*- coding: utf-8 -*-
import reconstruction
reconstruction.Recons.visualize_result = lambda self, mesh_gen=False: None
import os
os.environ["PYOPENGL_PLATFORM"] = "windows"
# ───── 在导入任何与 OpenGL、pyrender、tacto 相关模块之前，先做劫持 ─────
import numpy as np

# 定义一个“虚拟”Renderer，接口与 pyrender.OffscreenRenderer 保持一致
class DummyRenderer:
    def __init__(self, width, height, *args, **kwargs):
        self.width = width
        self.height = height
    def render(self, scene, flags=None):
        # 返回一个空的 RGB 数组和深度数组
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        return color, depth
    def delete(self):
        pass

# 劫持 pyrender.OffscreenRenderer
import pyrender
pyrender.OffscreenRenderer = DummyRenderer

import sys
import numpy as np
import open3d as o3d
import pybullet as p
import pybulletX as px
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

# ─────── 确保项目根目录在 sys.path ───────────────────────────
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import TactoEnv
from my_tools.predict_Multi_projection import multi_projection, visualize_all

# 用 GUI 模式启动仿真，并 monkey-patch px.init
_client_id = p.connect(p.GUI)
px.init = lambda *args, **kwargs: _client_id

def align_sensor_orientation(curr_ori: np.ndarray, desired_dir: np.ndarray) -> np.ndarray:
    """
    将 curr_ori 表示的本地 +Z 轴对准 desired_dir，
    返回新的四元数 new_ori (x,y,z,w)。
    """
    curr_rot = R.from_quat(curr_ori)
    curr_z   = curr_rot.apply([0,0,-1])
    axis = np.cross(curr_z, desired_dir)
    if np.linalg.norm(axis) < 1e-6:
        return curr_ori
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(curr_z, desired_dir), -1.0, 1.0))
    delta = R.from_rotvec(axis * angle)
    return (delta * curr_rot).as_quat()

def choose_translation_action(direction: np.ndarray, env: TactoEnv) -> int:
    """
    在 env.action_to_direction 的平移动作（编号 0-5）中，
    找到与给定 direction 最接近的动作 ID。
    """
    best_aid, best_dot = 0, -np.inf
    for aid, delta in env.action_to_direction.items():
        if aid > 5:  # 只看 ±X,±Y,±Z
            continue
        vec = delta[:3]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        dot = float(np.dot(vec, direction))
        if dot > best_dot:
            best_dot, best_aid = dot, aid
    return best_aid

def choose_rotation_action(axis: np.ndarray, env: TactoEnv) -> int:
    """
    在 env.action_to_direction 的旋转动作（aid>=6）中，
    挑一个最贴合给定旋转轴 axis 的动作 ID。
    """
    best_aid, best_dot = None, -np.inf
    for aid, delta in env.action_to_direction.items():
        if aid < 6:  # 0-5 是平移
            continue
        rotv = delta[3:]  # axis-angle 向量
        if np.linalg.norm(rotv) < 1e-6:
            continue
        rv = rotv / np.linalg.norm(rotv)
        dot = float(np.dot(rv, axis))
        if dot > best_dot:
            best_dot, best_aid = dot, aid
    return best_aid

def attempt_physical_touch(env: TactoEnv,
                           origin_pos: np.ndarray,
                           origin_ori: np.ndarray,
                           desired_dir: np.ndarray,
                           max_rot_steps: int,
                           max_trans_steps: int,
                           rot_step_angle: float,
                           trans_step_size: float) -> (np.ndarray or None, np.ndarray):
    """
    1) 复位到 origin_pos/origin_ori；
    2) 连续+离散旋转对齐，使本地感知面(-Z)朝向 desired_dir；
    3) 射线检测 + 基于行进距离阈值的离散平移，直至接触。
    返回 (touch_point, 最终 ori)，失败时复位并返回 (None, origin_ori)。
    """
    # 1) 复位
    env.digit_body.set_base_pose(origin_pos, origin_ori)
    for _ in range(5): p.stepSimulation()

    # 2) 连续对齐
    ori_cont = align_sensor_orientation(origin_ori, desired_dir)
    env.digit_body.set_base_pose(origin_pos, ori_cont)
    for _ in range(5): p.stepSimulation()
    # 同步当前位姿
    curr_pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
    curr_pos = np.array(curr_pos)
    ori = np.array(ori)

    # 离散旋转微调
    for _ in range(max_rot_steps):
        curr_z = R.from_quat(ori).apply([0,0,-1])
        axis = np.cross(curr_z, desired_dir)
        if np.linalg.norm(axis) < 1e-3:
            break
        axis /= np.linalg.norm(axis)

        rot_aid = choose_rotation_action(axis, env)
        _, _, _, _ = env.step(rot_aid)
        for _ in range(2): p.stepSimulation()
        _, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        ori = np.array(ori)

        if np.dot(R.from_quat(ori).apply([0,0,-1]), desired_dir) > 1 - 1e-3:
            break

    # 3) 预计算最大行进距离
    max_dist = max_trans_steps * trans_step_size
    # 记录当前起始点
    start_pos, _ = p.getBasePositionAndOrientation(env.digit_body.id)
    start_pos = np.array(start_pos)

    # 离散平移推进
    for _ in range(max_trans_steps):
        trans_aid = choose_translation_action(desired_dir, env)
        _, _, _, _ = env.step(trans_aid)
        for _ in range(2): p.stepSimulation()

        # 检测接触
        if getattr(env, "is_touching", False):
            info = env._get_observation()
            touches = info.get("new_touch_points", [])
            if touches:
                return np.array(touches[0]), ori

        # 基于行进距离阈值提前中断
        curr_pos, _ = p.getBasePositionAndOrientation(env.digit_body.id)
        curr_pos = np.array(curr_pos)
        if np.linalg.norm(curr_pos - start_pos) > max_dist:
            break

    # 失败：飞出或遍历完毕，复位到 origin
    env.digit_body.set_base_pose(origin_pos, origin_ori)
    for _ in range(5): p.stepSimulation()
    return None, origin_ori




def visualize_with_normal(raw_pts, pred_pts, normals, hit_idx, length):
    """
    只画法线：原始点云灰，预测点云红，被测试点黄，法线箭头黄。
    """
    geoms = []
    pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_pts))
    pcd_raw.paint_uniform_color([0.8,0.8,0.8])
    geoms.append(pcd_raw)

    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts))
    cols = np.tile([1,0,0], (len(pred_pts),1))
    cols[hit_idx] = [1,1,0]
    pcd_pred.colors = o3d.utility.Vector3dVector(cols)
    geoms.append(pcd_pred)

    ori = pred_pts[hit_idx]
    n0  = normals[hit_idx] / np.linalg.norm(normals[hit_idx])
    pts = np.vstack([ori, ori + n0 * length])
    ls  = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0,1]])
    )
    ls.colors = o3d.utility.Vector3dVector([[1,1,0]])
    geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)

@hydra.main(version_base="1.1", config_path="../conf", config_name="test")
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())

    # ─── 初始化环境并完成首触 ───────────────────────────────────────
    env = TactoEnv(cfg)
    obs = env.reset()  # 内部会执行 _go_first_touch()
    origin_pos = np.array(env.curr_pos)
    origin_ori = np.array(env.curr_ori)

    # ─── 读取初始点云 & 边界提取 ────────────────────────────────────
    init_pcd = o3d.io.read_point_cloud(cfg.test.input_pcd)
    init_pts = np.asarray(init_pcd.points)
    bidx = set()
    for axes in [(0,1),(1,2),(0,2)]:
        proj = init_pts[:,axes]
        if len(proj)>=3:
            bidx.update(ConvexHull(proj).vertices.tolist())
    edge_pts = init_pts[sorted(bidx)]

    # ─── 设置累积保存目录 ───────────────────────────────────────────
    refined_dir = os.path.join(get_original_cwd(), "output", "refined")
    os.makedirs(refined_dir, exist_ok=True)
    accum_pts = init_pts.copy()
    accurate, inaccurate = set(), set()
    center = init_pts.mean(axis=0)

    # ─── Active Refinement 主循环 ───────────────────────────────────
    for it in range(cfg.test.active_iter+1):
        print(f"=== Iteration {it} ===")
        pts_for_pred = init_pts if it==0 else accum_pts

        preds, pred_norms, edge_points, arrow_lines = multi_projection(
            pts_for_pred,
            num_interp=cfg.test.num_interp,
            step=cfg.test.step_size,
            knn_normals=cfg.test.knn_normals
        )

        if it == 0:
            # 首轮：可视化 mesh+点云+箭头
            visualize_all(init_pts, edge_pts, preds, arrow_lines)
            continue

        # 物理接触验证
        hit_pt, hit_idx = None, None
        #max_steps = int(np.ceil(cfg.test.max_depth / cfg.test.step_size))
        for idx, (p0, n0) in enumerate(zip(preds, pred_norms)):
            if idx in accurate or idx in inaccurate:
                continue

            # 确定朝向：使传感面贴向物体
            v = p0 - center
            desired = (-n0 if np.dot(v,n0)>0 else n0)
            desired /= np.linalg.norm(desired)

            rot_step    = cfg.test.rotate_step   # 角度单位（rad）
            trans_step  = cfg.test.step_size     # 平移单位（m）
            max_rot     = int(np.ceil(np.pi / rot_step))
            max_trans   = int(np.ceil(cfg.test.max_depth / trans_step))

            contact, final_ori = attempt_physical_touch(
                env,
                origin_pos,
                origin_ori,
                desired,
                max_rot,
                max_trans,
                rot_step,
                trans_step
            )

            if contact is not None:
                hit_pt, hit_idx = contact, idx
                accurate.add(idx)
                origin_pos = hit_pt.copy()
                origin_ori = final_ori.copy()
                print(f"[✅] 点{idx}命中 at {hit_pt}")
                break
            else:
                inaccurate.add(idx)
                print(f"[❌] 点{idx}未命中")

        if hit_pt is None:
            print("无预测点命中，终止。")
            break

        # 累加 & 写文件
        accum_pts = np.vstack([accum_pts, hit_pt])
        out_file = os.path.join(refined_dir, f"iter_{it}.pcd")
        o3d.io.write_point_cloud(
            out_file,
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(accum_pts))
        )

        #visualize_with_normal(init_pts, preds, pred_norms, hit_idx, cfg.test.step_size)

    print("Active refinement 完成。")

if __name__ == "__main__":
    main()
