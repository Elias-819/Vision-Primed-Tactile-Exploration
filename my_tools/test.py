"""
move_to_pred.py

脚本说明：在 AcTExplore 环境中，使用多视角点云外推算法（predict_Multi_projection.multi_projection）
基于当前累计触觉点云：
  - 第一次从相机可见点云文件加载；
  - 之后使用仿真中实时收集到的触觉点云；
对预测出的每个触点：
  1. 计算 Base Link 到 Gel 面的偏移；
  2. 将基座移动到目标，再可选对齐表面法线；
  3. 停留若干秒、采样、更新点云，循环多点执行。

用法：
    python move_to_pred.py --config conf/RL.yaml \
                         [--num_interp 10] [--step 0.005] [--knn_normals 30] [--pause 2.0]
"""
import argparse
import yaml
import time
import os
import numpy as np
import pybullet as p
import pybulletX as px
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from env import TactoEnv
from predict_Multi_projection import multi_projection
import open3d as o3d

import csv
from pathlib import Path
from datetime import datetime
# Monkey-patch px.init to support string modes
_orig_px_init = px.init
class PathViz:
    def __init__(self, enabled=False, stride=1, width=2.0, lifetime=0.0, color=(0,1,0)):
        self.enabled = bool(enabled)
        self.stride = max(1, int(stride))
        self.width = float(width)
        self.lifetime = float(lifetime)
        self.color = list(map(float, color))
        self.prev = None
        self.count = 0

    def reset(self):
        self.prev = None
        self.count = 0

    def _cur_base_pos(self, env):
        pos, _ = p.getBasePositionAndOrientation(env.digit_body.id)
        return np.asarray(pos, dtype=float)

    def draw(self, env):
        if not self.enabled:
            return
        self.count += 1
        if (self.count % self.stride) != 0:
            return
        cur = self._cur_base_pos(env)
        if self.prev is not None and np.linalg.norm(cur - self.prev) > 1e-6:
            p.addUserDebugLine(self.prev.tolist(), cur.tolist(),
                               lineColorRGB=self.color, lineWidth=self.width,
                               lifeTime=self.lifetime)
        self.prev = cur
def _px_init(mode, *args, **kwargs):
    mode_val = getattr(p, mode) if isinstance(mode, str) and hasattr(p, mode) else mode
    return _orig_px_init({}, mode_val)
px.init = _px_init

def parse_args():
    parser = argparse.ArgumentParser(
        description="在 AcTExplore 中执行多点预测驱动的传感器移动示例"
    )
    parser.add_argument('--config',      type=str,   required=True, help='RL 配置文件路径 (YAML)')
    parser.add_argument('--num_interp',   type=int,   default=5,     help='边界插值数量')
    parser.add_argument('--step',         type=float, default=0.0025, help='外推步长 (米)')
    parser.add_argument('--knn_normals',  type=int,   default=30,    help='法线估计 KNN 数量')
    parser.add_argument('--pause',        type=float, default=0,   help='停留时间 (秒)')

    parser.add_argument('--log_dir',  type=str, default='outputs/press_logs/logs_5', help='日志目录')
    parser.add_argument('--run_name', type=str, default=None,          help='运行名(可选)')

    parser.add_argument('--viz_path', action='store_true', help='在仿真中绘制轨迹线')
    parser.add_argument('--viz_stride', type=int, default=1, help='每多少个微步画一段线')
    parser.add_argument('--viz_width', type=float, default=2.0, help='轨迹线宽度')
    parser.add_argument('--viz_lifetime', type=float, default=0.0, help='线段生存期，0=永久')
    parser.add_argument('--viz_color', type=str, default='0,1,0', help='RGB 颜色，例如 0,1,0 表示绿色')

    return parser.parse_args()

def recompute_coverage(env, pcd_np=None) -> float:
    """
    用 accumulate=False 重算覆盖率，保持与对比实验一致。
    """
    if pcd_np is not None:
        env.recon.pcd = np.asarray(pcd_np, dtype=np.float32)
    try:
        env.recon.compute_coverage(accumulate=False, visualize=False)
    except TypeError:
        env.recon.compute_coverage(accumulate=False)
    return float(getattr(env.recon, "coverage", float("nan")))

def _get_pose(env):
    # 取当前传感器基座位姿（作为轨迹点）
    return p.getBasePositionAndOrientation(env.digit_body.id)

class RLLogger:
    """
    两个 CSV：
      1) steps：每次“触摸采样完成（一次 env.step）”记一行（即一个 RL step）
         列：time_iso, rl_step, touch_idx, coverage(%), pcd_points, reward, done,
             pos_x,pos_y,pos_z, ori_x,ori_y,ori_z,ori_w, path_len_m
      2) traj ：每个触摸采样作为一个路径点
         列：rl_step, touch_idx, pos_x,pos_y,pos_z, ori_x,ori_y,ori_z,ori_w, path_len_m
    """
    def __init__(self, log_dir="output/logs", run_name=None):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tag = run_name if run_name else time.strftime("%Y%m%d_%H%M%S")
        self.path_steps = os.path.join(log_dir, f"steps_{tag}.csv")
        self.path_traj  = os.path.join(log_dir, f"traj_{tag}.csv")

        self.f_steps = open(self.path_steps, "w", newline="", encoding="utf-8")
        self.f_traj  = open(self.path_traj,  "w", newline="", encoding="utf-8")
        self.w_steps = csv.writer(self.f_steps)
        self.w_traj  = csv.writer(self.f_traj)

        self.w_steps.writerow(["time_iso","elapsed_s","rl_step","touch_idx","coverage(%)","pcd_points",
                       "reward","done","pos_x","pos_y","pos_z",
                       "ori_x","ori_y","ori_z","ori_w","path_len_m"])
        self.w_traj.writerow(["time_iso","elapsed_s","rl_step","touch_idx","pos_x","pos_y","pos_z",
                      "ori_x","ori_y","ori_z","ori_w","path_len_m"])

        self.rl_step = 0
        self.prev_pos = None
        self.path_len = 0.0  # 累计路径长度（只按采样点计算）
        self.t0 = None


    def _update_path(self, pos):
        if self.prev_pos is not None:
            dp = np.linalg.norm(np.asarray(pos) - np.asarray(self.prev_pos))
            self.path_len += float(dp)
        self.prev_pos = np.asarray(pos)

    def _pc_stats(self, env):
        try:
            pc = np.asarray(env.recon.pcd)
            n  = pc.shape[0] if pc.ndim == 2 else 0
        except Exception:
            n = 0
        try:
            cov = float(env.recon.coverage)
        except Exception:
            cov = float("nan")
        return n, cov

    def log_start(self, env):
        # 可选：写一条基线（rl_step=0，不计入轨迹）
        n, cov = self._pc_stats(env)
        t = datetime.now().isoformat(timespec="seconds")
        pos, ori = _get_pose(env)
        self.t0 = time.perf_counter()
        self.w_steps.writerow([t, f"{0.000:.3f}", 0, -1,
                       f"{cov:.6f}" if np.isfinite(cov) else "", int(n),
                       "", "", pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3],
                       f"{self.path_len:.6f}"])
        self.f_steps.flush()

    def log_touch_step(self, env, touch_idx, reward=None, done=None):
        """
        在一次 env.step(...) 后调用：这就等于“完成一次触摸采样 = 1 个 step”
        同时把当前位置作为路径点写入 traj.csv，并累计路径长度。
        """
        self.rl_step += 1
        pos, ori = _get_pose(env)
        self._update_path(pos)

        n, cov = self._pc_stats(env)
        t = datetime.now().isoformat(timespec="seconds")
        elapsed = 0.0 if self.t0 is None else (time.perf_counter() - self.t0)
        # steps.csv
        self.w_steps.writerow([t, f"{elapsed:.3f}", self.rl_step, int(touch_idx),
                       f"{cov:.6f}" if np.isfinite(cov) else "", int(n),
                       "" if reward is None else f"{float(reward):.6f}",
                       "" if done   is None else int(bool(done)),
                       pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3],
                       f"{self.path_len:.6f}"])
        self.f_steps.flush()

        # traj.csv
        self.w_traj.writerow([t, f"{elapsed:.3f}", self.rl_step, int(touch_idx),
                      pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3],
                      f"{self.path_len:.6f}"])
        self.f_traj.flush()

    def close(self):
        for f in (self.f_steps, self.f_traj):
            try: f.close()
            except: pass

def load_point_cloud(npy_path: str, env, cfg):
    """
    加载或初始化点云：
      - 若存在可见点云文件，则对齐到世界坐标系；
      - 否则使用 env.recon.pcd
    返回：初始点云 (N,3)
    """
    # 对象底部 Z
    obj_pos, _ = p.getBasePositionAndOrientation(env.obj.id)
    base_z = obj_pos[2]

    if os.path.isfile(npy_path):
        pc = np.load(npy_path)
        zmin = pc[:,2].min()
        pc[:,2] += base_z - zmin
        print(f"[Load] 相机可见点云，Z 范围 {pc[:,2].min():.4f} ~ {pc[:,2].max():.4f}")
    else:
        pc = np.asarray(env.recon.pcd)
        print(f"[Load] 使用触觉采集点云，Z 范围 {pc[:,2].min():.4f} ~ {pc[:,2].max():.4f}")
    return pc.copy()

def subsample_representative_points(preds, normals, eps=0.005, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(preds)
    labels = clustering.labels_
    reps, rep_norms = [], []
    for lbl in set(labels):
        idxs = np.where(labels == lbl)[0]
        pts = preds[idxs]
        nms = normals[idxs]
        center = pts.mean(axis=0)
        avg_n = nms.mean(axis=0)
        avg_n /= np.linalg.norm(avg_n) + 1e-9
        reps.append(center)
        rep_norms.append(avg_n)
    # 噪声点
    noise = np.where(labels == -1)[0]
    for i in noise:
        reps.append(preds[i])
        rep_norms.append(normals[i] / (np.linalg.norm(normals[i]) + 1e-9))
    return np.vstack(reps), np.vstack(rep_norms)

def compute_gel_offset():
    # Gel 面中心相对于 base_link 原点的本地偏移
    half_length_x = 0.0323276 / 2.0
    half_thickness = 0.0340165 / 2.0
    return np.array([half_length_x, 0.0, half_thickness])

def predict_touch_points(partial_pcd, args):
    preds, _, surf_normals, _, _ = multi_projection(
        partial_pcd,
        num_interp=args.num_interp,
        step=args.step,
        knn_normals=args.knn_normals
    )
    # return subsample_representative_points(preds, surf_normals) # 聚类
    return preds, surf_normals

def create_visual_markers(preds):
    sphere_vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.002,
        rgbaColor=[1, 0, 0, 1]
    )
    for pt in preds:
        p.createMultiBody(0, -1, sphere_vis, pt.tolist())

def visualize_pressure_map(depth_map: np.ndarray, idx: int):
    """
    单独显示某次触点的压力热力图。

    参数:
        depth_map: H×W 压力（变形深度）图
        idx:       触点编号 (从0开始计)
    """
    plt.figure("Pressure Map")
    plt.clf()
    im = plt.imshow(depth_map, cmap='hot', vmin=0, vmax=depth_map.max())
    plt.title(f"Pressure Map at touch #{idx+1}")
    plt.colorbar(im, label='Deformation (m)')
    plt.pause(0.1)

def move_and_touch(env, preds, normals, args, logger=None, path_viz=None):
    gel_local = compute_gel_offset()
    if path_viz is not None:
        path_viz.reset()
    for idx, target in enumerate(preds):
        print(f"[{idx+1}/{len(preds)}] 目标触点: {target}")
        pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        Rbw = R.from_quat(ori)
        gel_off_w = Rbw.apply(gel_local)

        norm = normals[idx]
        safety = 0.03
        safe_pt = target - norm * safety
        base_target = (safe_pt - gel_off_w)

        curr_norm = Rbw.apply([1,0,0])
        rot_align, _ = R.align_vectors([norm], [curr_norm])
        new_ori = (rot_align * Rbw).as_quat().tolist()

        step_size, thresh = 0.0003, 1.5
        n_steps = int(safety / step_size)
        last_safe = safe_pt
        for k in range(n_steps):
            sub = safe_pt + norm * (k * step_size)
            env.digit_body.set_base_pose(sub, new_ori)
            
            p.stepSimulation()
            if path_viz is not None:
                path_viz.draw(env)
            forces = env.digits.get_force('cam0')
            if sum(forces.values()) >= thresh:
                env.digit_body.set_base_pose(last_safe, new_ori)
                colors, depths, gel_pc = env.digits.render()
                # visualize_pressure_map(depths[0], idx)
                break
            last_safe = sub

        print(f"采样停留 {args.pause}s...")
        time.sleep(args.pause)
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        print(f"Result: reward={reward:.4f}, done={done}")

        try:
            recompute_coverage(env, np.asarray(env.recon.pcd))
        except Exception:
            pass
        if logger is not None:
            logger.log_touch_step(env, touch_idx=idx, reward=reward, done=done)

        if done:
            print("环境终止，退出。")
            break
        yield np.asarray(env.recon.pcd)

def merge_point_clouds(existing_pcd: np.ndarray,
                       new_pcd: np.ndarray,
                       threshold: float = 1e-3) -> np.ndarray:
    """
    合并两个点云并去重合并。

    参数：
      existing_pcd: (M,3) 已有部分点云
      new_pcd:      (N,3) 新采集到的触觉点云
      threshold:    float, 去重距离阈值（米），在此距离内视为重复点

    返回：
      merged_pcd:   (K,3) 合并去重后的点云，K <= M+N
    """
    # 如果已有点云为空，直接返回 new_pcd 的拷贝
    if existing_pcd.size == 0:
        return new_pcd.copy()

    # 在已有点云上构建 KDTree
    tree = cKDTree(existing_pcd)
    # 查询 new_pcd 中每个点到已有点云的最近距离
    dists, _ = tree.query(new_pcd, k=1)
    # 只保留距离大于阈值的点
    mask = dists > threshold
    unique_new = new_pcd[mask]
    # 拼接并返回
    merged_pcd = np.vstack([existing_pcd, unique_new])
    return merged_pcd

def save_pcd(pcd_np: np.ndarray, filename: str):
    """
    将 (N,3) 的 numpy 点云保存为 PCD 文件。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(filename, pcd)
    print(f">> 已保存点云到: {filename}")

from predict_Multi_projection import densify_via_knn
def load_and_densify(npy_path: str, env, cfg,
                     knn_densify: int = 8,
                     num_interp_den: int = 3,
                     max_dist: float = 0.02) -> np.ndarray:
    # 原有加载逻辑
    pc = load_point_cloud(npy_path, env, cfg)
    print(f"[DENSIFY] 原始点云数: {pc.shape[0]}，开始 KNN 补点...")
    # KNN 补点
    pc_dense = densify_via_knn(
        pts=pc,
        knn=knn_densify,
        num_interp=num_interp_den,
        max_dist=max_dist
    )
    print(f"[DENSIFY] 补点后点数: {pc_dense.shape[0]}")
    return pc_dense

def densify_via_knn(pts: np.ndarray,
                    knn: int = 10,
                    num_interp: int = 3,
                    max_dist: float = None) -> np.ndarray:
    """
    基于 KNN 的局部线性插值补点。
    只在每个点与其 knn 近邻之间插 num_interp 段插值点，
    并可选地按 max_dist 剔除过远的邻居。

    输入：
      pts          (N×3) — 原始点
      knn            int — 最近邻数量（不含自身）
      num_interp     int — 每对连线的插值点数量
      max_dist     float — 两点距离超过此值则跳过插值；若为 None，则不限制

    返回：
      all_pts     (M×3) — 原始 pts + 新插值点
    """
    tree = cKDTree(pts)
    # 返回距离和索引，k=knn+1 包含自身
    dists, idxs = tree.query(pts, k=knn+1)
    new_pts = []
    # 插值参数 t，从 (1/(num_interp+1)) 到 (num_interp/(num_interp+1))
    ts = np.linspace(0,1,num_interp+2)[1:-1][:,None]  # shape (num_interp,1)

    N = pts.shape[0]
    for i in range(N):
        p0 = pts[i]
        for j_idx, d in zip(idxs[i,1:], dists[i,1:]):
            if max_dist is not None and d > max_dist:
                continue
            p1 = pts[j_idx]
            # 在 p0→p1 上做线性插值
            inters = p0 + ts * (p1 - p0)   # shape (num_interp,3)
            new_pts.append(inters)
    if not new_pts:
        return pts.copy()
    new_pts = np.vstack(new_pts)  # 所有插值点
    # 合并并返回
    return np.vstack([pts, new_pts])

def load_initial_pointcloud(env, rl_cfg, npy_path):
    obj_pos, _ = p.getBasePositionAndOrientation(env.obj.id)
    base_z = obj_pos[2]
    if os.path.isfile(npy_path):
        pcd = np.load(npy_path)
        zmin = pcd[:,2].min()
        pcd[:,2] += base_z - zmin
        print(">> 首次加载可见点云，Z 范围:", pcd[:,2].min(), pcd[:,2].max())
    else:
        pcd = np.asarray(env.recon.pcd)
        print(">> 使用触觉采集点云，Z 范围:", pcd[:,2].min(), pcd[:,2].max())
    return pcd.copy()

def main():
    args = parse_args()
    cfg = OmegaConf.create(yaml.safe_load(open(args.config, 'r', encoding='utf-8')))
    env = TactoEnv(cfg)
    env.reset()
    env.digits.visualize_gui = False; env.digits.show_depth=False; env.digits.render_point_cloud=True
    plt.ion()

    logger = RLLogger(log_dir=args.log_dir, run_name=args.run_name)
    # 解析颜色参数（"r,g,b"）
    try:
        rgb = tuple(float(x) for x in args.viz_color.split(','))
        if len(rgb) != 3: raise ValueError
    except Exception:
        rgb = (0,1,0)
    path_viz = PathViz(enabled=args.viz_path,
                       stride=args.viz_stride,
                       width=args.viz_width,
                       lifetime=args.viz_lifetime,
                       color=rgb)
    npy_path    = "output\cam/paitical_visible_from_camera.npy"
    pcd_initial = load_initial_pointcloud(env, cfg, npy_path)
    print("初始可见点云数:", pcd_initial.shape[0])

    recompute_coverage(env, pcd_initial)
    logger.log_start(env)

    pcd_global = pcd_initial.copy()

    while True:

        pcd_dense = densify_via_knn(pcd_global, knn=10, num_interp=3, max_dist=0.02)

        # 预测触点
        preds, normals = predict_touch_points(pcd_dense, args)
        if preds.size == 0:
                raise RuntimeError("未预测到触点，请检查参数。")
        
        # 收集触点
        collected_pcd = pcd_initial.copy()
        for new_pcd in move_and_touch(env, preds, normals, args, logger=logger, path_viz=path_viz):
            collected_pcd = new_pcd

        # 合并新采集的点云
        pcd_global = merge_point_clouds(pcd_global, np.vstack(collected_pcd), threshold=1e-3)
        print("合并后点云数:", pcd_global.shape[0])

        # 计算覆盖率
        env.recon.pcd = pcd_global
        env.recon.compute_coverage(accumulate=False, visualize=False)
        cov = env.recon.coverage
        if cov >= 90.0:
            print("覆盖率 ≥ 90%，结束循环。")
            break
    

    # create_visual_markers(preds)


    # save_pcd(merged_pcd, "output/first_merged.pcd")

    try:
        env.recon.pcd = pcd_global # .tolist()
        env.recon.visualize_result()
    except:
        pass

    logger.close()
    print(f"[SAVE] steps -> {logger.path_steps}")
    print(f"[SAVE] traj  -> {logger.path_traj}")

    env.close()

if __name__ == '__main__':
    main()
