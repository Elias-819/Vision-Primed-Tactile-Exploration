#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_to_pred.py — Refactored into模块化函数
在 AcTExplore 环境中，使用多视角点云外推算法驱动触觉传感器移动
--config conf/RL.yaml --viz_path
"""
import argparse
import yaml
import time
import os
import numpy as np
import pybullet as p
import pybulletX as px
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from env import TactoEnv
from predict_Multi_projection import multi_projection
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
# Monkey-patch px.init 兼容字符串模式
_orig_px_init = px.init
plt.ion()
pressure_fig = plt.figure("Pressure Map")
pressure_ax  = pressure_fig.add_subplot(111)
# —— 简单压力闭环（非 PID）参数 ——
F_TARGET   = 2.5     # 目标总力 N（按你的仿真标定调整）
F_TOL      = 0.15    # 死区 N：|F_TARGET - F| < F_TOL 时不调
KP_DEPTH   = 4e-4    # 单步比例系数 (m/N)：Δd = KP_DEPTH * (F_TARGET - F)
D_STEP_MAX = 3e-4    # 单步最大深度修正 (m)
NORMAL_SIGN = -1     # 沿哪侧法线增压；大多数情况 -1（朝物体内部）

F_EMERGENCY   = 5.0      # 紧急力阈值(N)——超过就快速后退
EMERGENCY_STEP= 1.5e-3   # 紧急后退步长(m)
PRE_CLEAR     = 5e-4     # 段首安全抬升(m)，避免一上来就过深
EPS_SIGN      = 2e-4     # 判定法线方向用的极小探测步长(m)

# —— 采样门控参数（按你的标定改）——
# 香蕉中部一块“规则的、与表面平行、大小≈传感面”的悬浮平面——
# 这几乎就是“把整块胶面在某个高度当作点云写入了”。
# 常见原因：在“安全抬升 PRE_CLEAR 后”或“刚开始沿法线微推但尚未接触”时，
# 接触判定被噪声触发，整帧以近似常量深度被投影到了世界坐标，形成一块与真实面平行的“副本”。
F_CONTACT_MIN   = 0.8        # 认为“接触”的最小总力 (N)
DEPTH_MIN       = 6e-5       # 认为“接触”的最小最大形变 (m) ~ 0.06 mm
AREA_FRAC_MIN   = 0.015      # 形变>阈值的像素占比(≥1.5%)，太小可能是噪声或边沿
STABLE_FRAMES   = 3          # 连续满足多少帧才算稳定接触

class ContactGate:
    def __init__(self):
        self.ok_frames = 0

    def ready(self, env):
        F = float(sum(env.digits.get_force('cam0').values()))
        _, rel_depths, _ = env.digits.render()
        pm = rel_depths[0]
        mask = (pm > DEPTH_MIN)
        ok = (F >= F_CONTACT_MIN) and (mask.mean() >= AREA_FRAC_MIN)
        self.ok_frames = self.ok_frames + 1 if ok else 0
        return self.ok_frames >= STABLE_FRAMES

import csv
from pathlib import Path
from datetime import datetime

_orig_px_init = px.init
def _px_init(mode, *args, **kwargs):
    if isinstance(mode, str) and hasattr(p, mode):
        mode_val = getattr(p, mode)
    else:
        mode_val = mode
    return _orig_px_init({}, mode_val)
px.init = _px_init

def parse_args():
    parser = argparse.ArgumentParser(
        description="在 AcTExplore 中执行多点预测驱动的传感器移动示例"
    )
    parser.add_argument('--config',      type=str,   required=True, help='RL 配置文件路径 (YAML)')
    parser.add_argument('--num_interp',   type=int,   default=0,     help='边界插值数量')
    parser.add_argument('--step',         type=float, default=0.005,  help='外推步长 (米)')
    parser.add_argument('--knn_normals',  type=int,   default=30,      help='法线估计 KNN 数量')
    parser.add_argument('--pause',        type=float, default=0.2,     help='到达预测点后停留秒数')
    parser.add_argument('--dist_cull',   type=float, default=0.0001,  help='物理距离裁剪阈值(米)')
    parser.add_argument('--probe_radius',type=float, default=0.0001, help='探针球半径(米)')
    parser.add_argument('--log_csv',    type=str, default='outputs/sliding_logs/slide_log_2_10_5.csv', help='步数-覆盖率-路径日志CSV路径')
    parser.add_argument('--cov_stride', type=int, default=1, help='每多少个采样步重算一次覆盖率(1=每步)')

    parser.add_argument('--echo', action='store_true', help='同时在终端打印每条采样日志')
    parser.add_argument('--log_pose_stride', type=int, default=5, help='每多少个微步记录一次位姿与时间（1=每步）')
    parser.add_argument('--viz_path', action='store_true',
                        help='在仿真中绘制移动轨迹线')
    parser.add_argument('--viz_stride', type=int, default=1,
                        help='每多少个微步画一段线（1=每步都画）')
    parser.add_argument('--viz_width', type=float, default=2.0,
                        help='轨迹线宽度')
    parser.add_argument('--viz_lifetime', type=float, default=0.0,
                        help='轨迹线存活时间(秒)，0=永久')
    parser.add_argument('--viz_color', type=str, default='solid',
                        help="轨迹颜色：'solid' 固定色 或 'progress' 按进度渐变")
    parser.add_argument('--viz_max_segments', type=int, default=5000,
                        help='最多保留的线段数（超出将删除最旧的）')
    return parser.parse_args()

def compute_cov_act(env) -> float:
    try:
        if hasattr(env.recon, 'last_index_acc'):
            env.recon.last_index_acc = 0
        env.recon.compute_coverage(accumulate=False, visualize=False)
    except Exception as e:
        print(f"[warn] compute_coverage raised: {e}")
    cov = getattr(env.recon, 'coverage', None)
    if cov is None:
        cov = getattr(env.recon, 'total_coverage', 0.0)
    return float(cov)
class PathVisualizer:
    """简易轨迹可视化：在仿真里画相邻位姿的连线，并限制段数"""
    def __init__(self, enabled=False, stride=1, width=2.0, lifetime=0.0,
                 color_mode='progress', max_segments=5000):
        self.enabled = bool(enabled)
        self.stride = max(1, int(stride))
        self.width = float(width)
        self.lifetime = float(lifetime)
        self.color_mode = str(color_mode)
        self.max_segments = int(max_segments)

        self.prev = None
        self.count = 0
        self.uids = []  # 存已画线段的 uid，便于超限时删除最旧

    def _get_tip_pose(self, env):
        """优先取末端 link 位姿；没有则取 base 位姿"""
        tip_link = getattr(env, "digit_tip_link", None)
        if tip_link is not None:
            ls = p.getLinkState(env.digit_body.id, int(tip_link), computeForwardKinematics=True)
            pos, ori = ls[4], ls[5]
        else:
            pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        return np.asarray(pos, float), np.asarray(ori, float)

    def _color(self, t_frac):
        if self.color_mode.lower() == 'solid':
            return [0, 1, 0]  # 固定绿色；需要可改成参数
        # progress：从蓝(起)→青→绿→黄→红(终)
        t = float(np.clip(t_frac if t_frac is not None else 0.0, 0.0, 1.0))
        # 简单五段分段线性
        if t < 0.25:
            return [0, 4*t, 1]            # (0,0,1)→(0,1,1)
        elif t < 0.5:
            return [0, 1, 1 - 4*(t-0.25)] # (0,1,1)→(0,1,0)
        elif t < 0.75:
            return [4*(t-0.5), 1, 0]      # (0,1,0)→(1,1,0)
        else:
            return [1, 1 - 4*(t-0.75), 0] # (1,1,0)→(1,0,0)

    def reset(self, env=None):
        """清空缓存；如需也删除已画线段，可以遍历 removeUserDebugItem"""
        self.prev = None
        self.count = 0
        # 不强制删除旧线；若想删，取消注释：
        # for uid in self.uids:
        #     try: p.removeUserDebugItem(uid)
        #     except: pass
        # self.uids.clear()

    def draw(self, env, t_frac=None):
        if not self.enabled:
            return
        pos, _ = self._get_tip_pose(env)
        if self.prev is None:
            self.prev = pos
            return
        self.count += 1
        if (self.count % self.stride) != 0:
            self.prev = pos
            return
        if not (np.isfinite(self.prev).all() and np.isfinite(pos).all()):
            self.prev = pos
            return
        if np.linalg.norm(pos - self.prev) < 1e-6:
            self.prev = pos
            return
        color = self._color(t_frac)
        uid = p.addUserDebugLine(self.prev.tolist(), pos.tolist(),
                                 lineColorRGB=color, lineWidth=self.width,
                                 lifeTime=self.lifetime)
        self.uids.append(uid)
        if len(self.uids) > self.max_segments:
            old = self.uids.pop(0)
            try: p.removeUserDebugItem(old)
            except: pass
        self.prev = pos
class RLLogger:
    """
    每当触发 recon_tick() 采样时，调用 log_sample(...) 写入一行。
    列：
      time_iso, step, coverage(%), cov_fresh, pcd_points, path_len_m,
      pos_x,pos_y,pos_z, ori_x,ori_y,ori_z,ori_w
    """
    def __init__(self, csv_path: str, cov_stride: int = 1, echo: bool = False):
        Path(os.path.dirname(csv_path) or ".").mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.cov_stride = max(1, int(cov_stride))
        self.f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "time_iso","elapsed_s","step","coverage(%)","cov_fresh","pcd_points","path_len_m",
            "pos_x","pos_y","pos_z","ori_x","ori_y","ori_z","ori_w"
        ])
        self.step = 0
        self.last_cov = None
        self.last_pos = None
        self.path_len = 0.0
        self.echo = echo

        self.t0 = None
    def _pc_count(self, env):
        pcd = getattr(env.recon, 'pcd', None)
        if pcd is None:
            return 0
        if isinstance(pcd, list):
            n = 0
            for seg in pcd:
                arr = np.asarray(seg)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    n += arr.shape[0]
            return int(n)
        arr = np.asarray(pcd)
        return int(arr.shape[0]) if (arr.ndim == 2 and arr.shape[1] >= 3) else 0

    def log_baseline(self, env, cov=None):
        pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        if cov is None:
            cov = compute_cov_act(env)
        self.last_cov = float(cov)
        self.last_pos = np.asarray(pos)
        self.t0 = time.perf_counter()              # ★ 记录本回合起点
        t_iso = datetime.now().isoformat(timespec="seconds")
        self.w.writerow([t_iso, f"{0.000:.3f}", 0, f"{self.last_cov:.6f}", 1, self._pc_count(env),
                        f"{self.path_len:.6f}", pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3]])
        self.f.flush()
        if self.echo:
            print(f"[LOG] step=0 cov={self.last_cov:.2f}% path={self.path_len:.3f}m")

    def log_sample(self, env):
        # 当前位姿
        pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        pos = np.asarray(pos, dtype=float)

        # 累计路径长度（仅在“采样点”之间）
        if self.last_pos is not None:
            self.path_len += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos

        # 步数 +1
        self.step += 1

        # 覆盖率（按 stride）
        cov_fresh = 0
        cov = self.last_cov if (self.step % self.cov_stride != 0) else None
        if cov is None:
            cov = compute_cov_act(env)
            self.last_cov = float(cov)
            cov_fresh = 1

        t_iso = datetime.now().isoformat(timespec="seconds")
        elapsed = 0.0 if self.t0 is None else (time.perf_counter() - self.t0)  # ★ 新增

        self.w.writerow([t_iso, f"{elapsed:.3f}", self.step,                 # ★ 加上 elapsed_s
                        f"{float(self.last_cov):.6f}", int(cov_fresh), self._pc_count(env),
                        f"{self.path_len:.6f}",
                        pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3]])
        self.f.flush()

        if self.echo:
            print(
                f"[LOG] step={self.step} cov={self.last_cov:.2f}% "
                f"(fresh={cov_fresh}) path={self.path_len:.3f}m "
                f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
            )

    def log_pose(self, env):
        """按当前时刻记录 elapsed_s + 位姿（不计算覆盖率）"""
        pos, ori = p.getBasePositionAndOrientation(env.digit_body.id)
        pos = np.asarray(pos, dtype=float)
        # 累计路径长度
        if self.last_pos is not None:
            self.path_len += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos
        elapsed = 0.0 if self.t0 is None else (time.perf_counter() - self.t0)
        t_iso = datetime.now().isoformat(timespec="seconds")
        # step 不自增：保持与采样步对齐；如需自增可自行调整
        self.w.writerow([t_iso, f"{elapsed:.3f}", self.step,
                        f"{float(self.last_cov or 0.0):.6f}", 0, self._pc_count(env),
                        f"{self.path_len:.6f}",
                        pos[0],pos[1],pos[2], ori[0],ori[1],ori[2],ori[3]])
        self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    return OmegaConf.create(cfg_dict)

def init_environment(rl_cfg):
    env = TactoEnv(rl_cfg)
    obs = env.reset()
    # 关闭额外窗口
    env.digits.visualize_gui = False
    env.digits.show_depth   = False
    plt.ion()
    fig, ax = plt.subplots(num="Depth Map")
    return env, fig, ax

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

def predict_touch_points(pcd, args):
    preds, dirs, normals, edges, arrows = multi_projection(
        pcd,
        num_interp=args.num_interp,
        step=args.step,
        knn_normals=args.knn_normals
    )
    if preds.shape[0] == 0:
        raise RuntimeError("预测未产出任何点，请检查输入或参数设置。")
    return preds, normals

def visualize_predictions(preds):
    sphere_vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.002,
        rgbaColor=[1,0,0,1],
    )
    for pt in preds:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=sphere_vis,
            basePosition=pt.tolist()
        )

def visualize_pressure_map(env, subtitle=None):
    """在同一个名为 "Pressure Map" 的窗口里，清空上一次内容，然后渲染新的压力热图。"""
    # 从 sensor 取数据
    _, rel_depths, _ = env.digits.render()
    pm = rel_depths[0]

    # 彻底清空 figure（包括所有 axes, colorbar）
    plt.figure("Pressure Map")
    pressure_fig.clf()

    # 重新建 axes、画图
    ax = pressure_fig.add_subplot(111)
    im = ax.imshow(pm, cmap="hot", vmin=0, vmax=pm.max())
    title = "Pressure Map"
    if subtitle:
        title += f" — {subtitle}"
    ax.set_title(title)

    # 新建 colorbar
    pressure_fig.colorbar(im, ax=ax, label="Deformation (m)")
    plt.pause(0.01)

def jump_and_touch(env, target, normal, gel_offset_local,
                   safety_dist, step_size, force_thresh,
                   pause_sec):
    """
    对单个预测点执行：
      1) 瞬移到贴合点；
      2) 反向法线平移到安全位；
      3) 沿法线推进直到检测到接触；渲染压力图；
      4) 停留并采样；
    返回 (tip_prev, ori)：
      - tip_prev: 胶面在触碰后真实停留处（world）
      - ori:      触碰完成后的姿态
    """
    # 计算当前 base→gel tip 偏移
    curr_pos, curr_ori = p.getBasePositionAndOrientation(env.digit_body.id)
    Rbw = R.from_quat(curr_ori)
    gel_off_w = Rbw.apply(gel_offset_local)

    # 对齐姿态：把 gel X 轴对齐到 surface normal
    curr_gel_norm = Rbw.apply([1,0,0])
    rot_align, _  = R.align_vectors([normal], [curr_gel_norm])
    new_rot       = rot_align * Rbw
    ori           = new_rot.as_quat().tolist()

    # 1) 瞬移到贴合点
    env.digit_body.set_base_pose((target - gel_off_w).tolist(), ori)
    p.stepSimulation()

    # 2) 反向法线到安全位
    safe_pt = target - normal * safety_dist
    env.digit_body.set_base_pose((safe_pt - gel_off_w).tolist(), ori)
    p.stepSimulation()

    # 3) 沿法线推进直到接触
    last_safe = safe_pt.copy()
    n_steps   = int(np.ceil(safety_dist / step_size))
    for k in range(n_steps+1):
        sub = safe_pt + normal * min(k * step_size, safety_dist)
        env.digit_body.set_base_pose((sub - gel_off_w).tolist(), ori)
        p.stepSimulation()
        total_f = sum(env.digits.get_force('cam0').values())
        if total_f > force_thresh:
            # 回退到最后安全位
            env.digit_body.set_base_pose((last_safe - gel_off_w).tolist(), ori)
            p.stepSimulation()
            # 渲染压力图
            #visualize_pressure_map(env, subtitle="Pressure Map at first touch")
            break
        last_safe = sub

    # 4) 停留采样
    time.sleep(pause_sec)
    obs, reward, done, *_ = env.step(env.action_space.sample())
    print(f" first point sampled: reward={reward:.3f}, done={done}")
    if done:
        return None, None  # 提前退出

    # tip_prev 是实际接触后胶面的 world 坐标
    tip_prev = last_safe
    # 把 new_rot 当作下一段滑动的起始姿态
    prev_ori = ori
    return tip_prev, prev_ori

def sort_preds_closed_loop(preds: np.ndarray,
                           normals: np.ndarray,
                           method: str = 'angular',
                           **kwargs):
    """
    将 preds, normals 按闭环顺序排序。

    参数
    ----
    preds : (N,3) array
        原始预测点的世界坐标。
    normals : (N,3) array
        每个预测点对应的表面法线。
    method : {'angular','tsp'}
        排序方法：
          - 'angular' : 在 XY 平面（可通过 kwargs 指定其他平面）按质心极角排序。
          - 'tsp'     : 用近似旅行商算法求最短环路。
    kwargs :
      # 对 method='angular' 有效：
      plane_axes : tuple[int,int] = (0,1)
          用于算极角的坐标轴，(0,1) 表示 XY 平面，(0,2) 表示 XZ 平面，等等。
      # 对 method='tsp' 有效：
      tsp_cycle   : bool = True
          返回的 cycle 是否包含闭环最后回到起点的节点序号；通常设为 False 返回 N 个点。

    返回
    ----
    preds_sorted   : (M,3) array
    normals_sorted : (M,3) array
    """
    if method == 'angular':
        # 在指定平面上做极角排序
        ax0, ax1 = kwargs.get('plane_axes',(0,1))
        centroid = preds.mean(axis=0)
        d0 = preds[:,ax0] - centroid[ax0]
        d1 = preds[:,ax1] - centroid[ax1]
        angles = np.arctan2(d1, d0)
        order = np.argsort(angles)
    
    elif method == 'tsp':
        import networkx as nx
        from networkx.algorithms.approximation import traveling_salesman_problem

        # 构造完全图
        dist_mat = squareform(pdist(preds))
        G = nx.Graph()
        N = len(preds)
        for i in range(N):
            for j in range(i+1, N):
                G.add_edge(i, j, weight=float(dist_mat[i,j]))
        # 求近似环路
        cycle = traveling_salesman_problem(G, weight='weight', cycle=True)
        include_last = kwargs.get('tsp_cycle', False)
        order = cycle if include_last else cycle[:-1]
    else:
        raise ValueError(f"Unknown method '{method}'")

    preds_sorted   = preds[order]
    normals_sorted = normals[order]
    return preds_sorted, normals_sorted

def visualize_sorted_preds_with_tangents_in_sim(preds, normals, slide_step=0.02, life_time=0):
    """
    在 PyBullet 中可视化排序后的预测点和切线箭头：
      - 小红球标记每个预测点
      - 真正的箭头（shaft + head）表示切线方向

    preds:      (N,3) numpy array, 排序后的 world 坐标
    normals:    (N,3) numpy array, 对应的表面法线
    slide_step: float, 箭杆的长度
    life_time:  float, debug line 存活时长 (秒)，0=永久
    """
    # 先把所有点用小红球标记一次
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.002, rgbaColor=[1,0,0,1])
    for pt in preds:
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=sphere_vis,
                          basePosition=pt.tolist())

    # 然后对每个点画一个箭头
    for i in range(len(preds)-1):
        p_i = preds[i]
        p_j = preds[i+1]

        # 1) 计算切线方向并单位化
        d = p_j - p_i
        tangent = d - np.dot(d, normals[i]) * normals[i]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-6:
            continue
        tangent /= norm_t

        # 箭杆起止
        start = p_i
        end   = p_i + tangent * slide_step

        # 2) 画箭杆
        p.addUserDebugLine(start.tolist(),
                           end.tolist(),
                           lineColorRGB=[0,0,1],
                           lineWidth=2,
                           lifeTime=life_time)

        # 3) 箭头头：在 end 处画两条短线
        #    长度取 slide_step*0.2，方向在切平面内垂直于 tangent
        head_len = slide_step * 0.2
        # perp 先取 tangent × normal（一定不平行于 tangent）
        perp = np.cross(tangent, normals[i])
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-6:
            # 如果太小，随便换一个垂直方向
            perp = np.cross(tangent, [1,0,0])
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-6:
                perp = np.cross(tangent, [0,1,0])
                perp_norm = np.linalg.norm(perp)
        perp /= perp_norm

        # 两个箭头头方向：向内和向外
        left_end  = end - tangent * head_len + perp * head_len
        right_end = end - tangent * head_len - perp * head_len

        p.addUserDebugLine(end.tolist(),
                           left_end.tolist(),
                           lineColorRGB=[0,0,1],
                           lineWidth=2,
                           lifeTime=life_time)
        p.addUserDebugLine(end.tolist(),
                           right_end.tolist(),
                           lineColorRGB=[0,0,1],
                           lineWidth=2,
                           lifeTime=life_time)

def subsample_representative_points(preds, normals, eps=0.005, min_samples=3):
    """
    使用 DBSCAN 在 3D 空间中聚类预测点，
    然后对每个聚类取中心点 (mean) 及对应平均法线，作为代表点返回。
    
    参数：
      preds:      (N,3) array 原始预测点
      normals:    (N,3) array 对应的法向
      eps:        float, DBSCAN 的半径参数（米），可根据点云密度调
      min_samples: int, DBSCAN 的最小簇大小

    返回：
      reps:       (M,3) array，M<<N 的代表预测点
      rep_norms:  (M,3) array，每个代表点对应的平均法线（归一化）
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(preds)
    labels = clustering.labels_
    unique_labels = [l for l in set(labels) if l >= 0]

    reps = []
    rep_norms = []
    for l in unique_labels:
        pts = preds[labels == l]
        nms = normals[labels == l]
        # 聚类中心
        center = pts.mean(axis=0)
        # 法线取平均再归一化
        avg_n  = nms.mean(axis=0)
        avg_n /= np.linalg.norm(avg_n) + 1e-9

        reps.append(center)
        rep_norms.append(avg_n)

    # 如果有噪声点（label==-1），也可以把它们当成单点簇保留：
    noise_idx = np.where(labels == -1)[0]
    for i in noise_idx:
        reps.append(preds[i])
        rep_norms.append(normals[i] / (np.linalg.norm(normals[i]) + 1e-9))

    return np.vstack(reps), np.vstack(rep_norms)

def visualize_clustered_sequence(reps, normals, order, figsize=(6,6)):
    """
    使用 matplotlib 在单独窗口中可视化聚类后并排序的代表点序列：
      - 红色圆点表示代表点
      - 按给定顺序用蓝色箭头依次从第 i 点指向第 i+1 点
    参数:
      reps:    (M,3) np.ndarray, 代表点位置
      normals: (M,3) np.ndarray, 对应表面法线（本函数暂未使用）
      order:   list[int], 排序索引顺序
      figsize: 窗口大小
    """
    fig = plt.figure("Clustered Sequence", figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 按顺序绘制代表点
    pts = reps[order]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c='r', s=30, label='Representatives')

    # 用箭头依次连线：箭头从 p0 指向 p1，长度正好是 p1−p0
    for i in range(len(order)-1):
        p0 = reps[order[i]]
        p1 = reps[order[i+1]]
        vec = p1 - p0
        # 画箭杆和箭头：normalize=False 时，(U,V,W) 就是实际长度
        ax.quiver(p0[0], p0[1], p0[2],
                  vec[0], vec[1], vec[2],
                  arrow_length_ratio=0.2,
                  linewidth=1,
                  color='b',
                  normalize=False)

    ax.set_title('Clustered & Sorted Sequence')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def align_ori_to_normal(base_ori, normal):
    R_init = R.from_quat(base_ori)
    gel_dir = R_init.apply([1,0,0])
    rot_align, _ = R.align_vectors([normal], [gel_dir])
    return (rot_align * R_init).as_quat().tolist()

def compute_base_position(contact_pt: np.ndarray,
                          gel_offset_local: np.ndarray,
                          ori: list) -> list:
    """
    contact_pt: gel 面中心在 world 的坐标 (3,)
    gel_offset_local: gel 面中心在 base local frame 的偏移 (3,)
    ori: 当前 base 在 world 下的四元数 [x,y,z,w]
    """
    # 1) 将本地偏移旋转到 world：
    Rbw = R.from_quat(ori)
    gel_off_w = Rbw.apply(gel_offset_local)
    # 2) base position = contact_pt - gel_off_w
    base_pos = contact_pt - gel_off_w
    return base_pos.tolist()

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

def lerp_normal(n0: np.ndarray, n1: np.ndarray, t: float) -> np.ndarray:
    """线性插值法线并归一化，t∈[0,1]"""
    v = (1.0 - t) * n0 + t * n1
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def total_force(env) -> float:
    return float(sum(env.digits.get_force('cam0').values()))

def decide_push_sign(env, pt, n, ori, gel_offset_local) -> int:
    """
    ★ 判定“沿 +n 还是 −n 会让压力变大”。返回 +1 或 -1：
      +1 表示朝 +n 推会增压；-1 表示朝 −n 推会增压。
    """
    base0 = compute_base_position(pt, gel_offset_local, ori)
    env.digit_body.set_base_pose(base0, ori); p.stepSimulation()
    F0 = total_force(env)

    base_plus = compute_base_position(pt + EPS_SIGN * n, gel_offset_local, ori)
    env.digit_body.set_base_pose(base_plus, ori); p.stepSimulation()
    Fp = total_force(env)

    # 回到原位（避免状态漂移）
    env.digit_body.set_base_pose(base0, ori); p.stepSimulation()

    return +1 if Fp > F0 else -1

def recon_tick(env, logger: RLLogger = None):
    # 确保 curr_pos/curr_ori 是最新的（_get_observation 里会用到）
    env.curr_pos, env.curr_ori = p.getBasePositionAndOrientation(env.digit_body.id)
    _ = env._get_observation()   # 这里内部会 insert_data，把新一帧点云写入重建
    if logger is not None:
        logger.log_sample(env)

def slide_between(env, start_pt, start_n, 
                  end_pt, end_n,
                  gel_offset_local, curr_ori,
                  slide_step: float, pause: float,
                  show_pressure: bool = True,
                  vis_stride: int = 5,
                  logger: RLLogger = None,
                  log_pose_stride: int = 5,
                  path_viz: PathVisualizer = None):
    """
    在 start_pt→end_pt 之间“平滑滑动 + 简单恒压微调（非 PID）”：
      - 姿态：对齐 start_n → 对齐 end_n，并用 SLERP 插值
      - 轨迹：对 gel 面中心做线性插值
      - 力控：每微步读取压力，沿“本段增压方向”的法线做小幅深度修正
    返回：end_ori, end_base
    """
    gate = ContactGate()

    # 1) 起止姿态
    ori_start = align_ori_to_normal(curr_ori, start_n)
    ori_end   = align_ori_to_normal(ori_start, end_n)

    # ★ 1.5) 判定“本段的增压方向” (+n 或 −n)
    push_sign = decide_push_sign(env, start_pt, start_n, ori_start, gel_offset_local)

    # ★ 1.6) 段首“安全抬升”一点，避免过深起步
    safe_pt = start_pt - push_sign * PRE_CLEAR * start_n
    base_safe = compute_base_position(safe_pt, gel_offset_local, ori_start)
    env.digit_body.set_base_pose(base_safe, ori_start); p.stepSimulation()

    # 2) 按“接触点距离”计算步数
    tip_dist = np.linalg.norm(end_pt - start_pt)
    steps    = max(1, int(np.ceil(tip_dist / max(1e-6, slide_step))))

    # 3) SLERP
    key_rots = R.from_quat([ori_start, ori_end])
    slerp    = Slerp([0, 1], key_rots)

    # 4) （可选）小校准：把力拉进死区，减小后续抖动
    for _ in range(3):
        F0 = total_force(env)
        err0 = F_TARGET - F0
        if abs(err0) < F_TOL:
            break
        delta0 = np.clip(KP_DEPTH * err0, -D_STEP_MAX, D_STEP_MAX)
        pt0    = safe_pt + push_sign * delta0 * start_n
        base0  = compute_base_position(pt0, gel_offset_local, ori_start)
        env.digit_body.set_base_pose(base0, ori_start); p.stepSimulation()
        safe_pt = pt0  # 累计到新的安全点
        #if show_pressure:
        #    visualize_pressure_map(env, subtitle=f"Calib F={F0:.2f}N")

    # 5) 微步滑动：按接触点/法线插值 + 压力修正
    for k in range(1, steps + 1):
        a = k / steps
        pt_nom = (1 - a) * start_pt + a * end_pt
        n_nom  = lerp_normal(start_n, end_n, a)
        ori_k  = slerp(a).as_quat().tolist()

        F   = total_force(env)
        # ★ 紧急后退：防止 100N 级别的“顶爆”
        if F > F_EMERGENCY:
            # 直接按“减压方向”退一大步
            pt_emg = pt_nom - push_sign * EMERGENCY_STEP * n_nom
            base_emg = compute_base_position(pt_emg, gel_offset_local, ori_k)
            env.digit_body.set_base_pose(base_emg, ori_k); p.stepSimulation()
            #if show_pressure:
            #    visualize_pressure_map(env, subtitle=f"EMERG! F={F:.1f}N → retreat")
            # 跳过后续比例修正，进入下一个微步
            time.sleep(min(0.01, pause / max(steps, 1)))
            continue

        # 正常的简单比例修正
        err = F_TARGET - F
        if abs(err) < F_TOL:
            delta_d = 0.0
        else:
            delta_d = np.clip(KP_DEPTH * err, -D_STEP_MAX, D_STEP_MAX)

        # ★ 用“本段的增压方向”修正深度
        pt_corr = pt_nom + push_sign * delta_d * n_nom

        base_k = compute_base_position(pt_corr, gel_offset_local, ori_k)
        env.digit_body.set_base_pose(base_k, ori_k)
        p.stepSimulation()
        if path_viz is not None:
            path_viz.draw(env, t_frac=a)
        if logger is not None and (k % max(1, log_pose_stride) == 0 or k == steps):
            logger.log_pose(env)
        if show_pressure:
            if (k % vis_stride == 0) or (abs(err) > 2 * F_TOL) or (k == steps):
                tag = "High" if err < -F_TOL else ("Low" if err > F_TOL else "OK")
                #visualize_pressure_map(env, subtitle=f"{k}/{steps} | F={F:.2f}N | {tag}")
        
        # —— 新增：触发重建采样 —— 
        if gate.ready(env):
            recon_tick(env, logger=logger)
        time.sleep(min(0.01, pause / max(steps, 1)))

    # 6) 结束位姿
    base_end = compute_base_position(end_pt, gel_offset_local, ori_end) 
    return ori_end, base_end

def execute_touch_sequence(env, preds, normals, args, logger: RLLogger = None, path_viz: PathVisualizer = None):

    # —— Gel 面偏移 & 参数 —— 
    half_length_x   = 0.0323276 / 2.0
    half_thickness  = 0.0340165 / 2.0
    gel_offset_local = np.array([half_length_x, 0.0, half_thickness])

    slide_step = 0.001  # 每次移动 1 mm，你可以根据需要调大或调小

    # —— 1. 聚类 & 排序 —— 
    reps, rep_norms = subsample_representative_points(
        preds, normals,
        eps=0.005,
        min_samples=3
    )

    # 用 reps 作为新的 preds
    preds, normals = sort_preds_closed_loop(
        reps, rep_norms,
        method='tsp',
        tsp_cycle=True
    )

    # —— 2. 获取固定姿态 —— 
    #    这里用 env.reset() 后当前 base 的 orientation
    _, curr_ori  = p.getBasePositionAndOrientation(env.digit_body.id)

    # —— 3. 最大直插距离阈值（超过就插值）—— 
    max_gap = 0.01  # 单位：米，比如 1cm
    # 这里我们把 preds 看成一个闭环，所以最后一个点后面接回第一个
    n_pts = len(preds)

    # 先把机器人放到第一个点（避免第一步从远处“跳入”）
    first_pt  = preds[0]; first_n = normals[0]
    curr_ori  = align_ori_to_normal(curr_ori, first_n)
    first_base= compute_base_position(first_pt, gel_offset_local, curr_ori)
    env.digit_body.set_base_pose(first_base, curr_ori)
    p.stepSimulation()
    time.sleep(args.pause)
    if path_viz is not None:
        path_viz.reset(env)
    # —— 4. 每一段：生成路径点（含 max_gap 插值）+ 段内滑动（slide_step） —— 
    for i in range(n_pts):
        p0, n0 = preds[i],               normals[i]
        p1, n1 = preds[(i + 1) % n_pts], normals[(i + 1) % n_pts]

        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)

        # 4.1 路径点（段间插值）：way_pts / way_ns
        if seg_len > max_gap:
            n_sub = int(np.ceil(seg_len / max_gap))
            # 生成 n_sub 个中间路径点（不含端点）
            ts = [(k / (n_sub + 1)) for k in range(1, n_sub + 1)]
        else:
            ts = []

        way_pts = [p0] + [p0 + t * seg_vec for t in ts] + [p1]
        way_ns  = [n0] + [lerp_normal(n0, n1, t) for t in ts] + [n1]

        # 4.2 段内滑动（相邻路径点两两之间连续滑动）
        for j in range(len(way_pts) - 1):
            start_pt, start_n = way_pts[j],     way_ns[j]
            end_pt,   end_n   = way_pts[j + 1], way_ns[j + 1]

            # 真正滑动：位置线性 + 姿态 SLERP
            curr_ori, curr_base = slide_between(
                env,
                start_pt, start_n,
                end_pt,   end_n,
                gel_offset_local,
                curr_ori,
                slide_step=slide_step,
                pause=args.pause,
                show_pressure=True,   # 打开可视化
                vis_stride=5,          # 每 5 步刷一帧；卡顿就调大
                logger=logger,
                log_pose_stride=args.log_pose_stride,
                path_viz=path_viz
            )

        print(f"[{i+1}/{n_pts}] segment done.")

    return np.asarray(env.recon.pcd)

def finalize(env):
    try:
        env.recon.visualize_result()
    except:
        pass
    env.close()

def cull_by_bullet_distance(pcd_xyz, obj_body_id, thresh=0.003, probe_r=0.0005):
    """
    用 PyBullet 的 getClosestPoints 计算点到物体的最近距离（以小球近似每个点）。
    距离 > thresh 的点判为“漂浮”，将被剔除。

    参数：
      pcd_xyz: (N,3) ndarray，世界坐标
      obj_body_id: int，物体的 body id（例如 env.obj.id）
      thresh: 距离阈值（米），常用 0.002~0.004（2~4mm）
      probe_r: 探针小球半径（米），给个很小的值（如 0.5mm）

    返回：
      pcd_kept: 剩余点 (M,3)
      keep_mask: bool 数组，长度 N
    """
    pcd_xyz = np.asarray(pcd_xyz, dtype=float)
    if pcd_xyz.size == 0:
        return pcd_xyz, np.zeros((0,), dtype=bool)

    # 建一个“探针球”刚体，重复挪到各点位置做最近距离查询
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=probe_r)
    probe = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col)
    keep = np.ones(len(pcd_xyz), dtype=bool)
    max_dist = float(thresh + probe_r)

    try:
        for i, pt in enumerate(pcd_xyz):
            p.resetBasePositionAndOrientation(probe, pt.tolist(), [0,0,0,1])
            # 只查询 max_dist 内的最近点；若返回为空，说明距离 > 阈值
            cps = p.getClosestPoints(bodyA=probe, bodyB=obj_body_id, distance=max_dist)
            if len(cps) == 0:
                keep[i] = False
    finally:
        # 只需移除 body；collision shape 无需手动删除
        try: p.removeBody(probe)
        except: pass

    return pcd_xyz[keep], keep

def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """
    体素下采样：同一体素格内取质心。voxel 以米为单位（建议 0.8~2.0 mm）。
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts
    vids = np.floor(pts / voxel).astype(np.int64)           # (N,3) 每点所在体素的整数索引
    _, inv = np.unique(vids, axis=0, return_inverse=True)   # inv: 点→体素 分组编号
    out = np.zeros((inv.max() + 1, 3), dtype=float)
    np.add.at(out, inv, pts)                                # 分组累加坐标
    counts = np.bincount(inv)
    out /= counts[:, None]                                  # 取质心
    return out

def merge_dedup(base_pts: np.ndarray, new_pts: np.ndarray, voxel: float) -> np.ndarray:
    """
    合并已有点云和新捕获点云，并体素去重。
    """
    base_pts = np.asarray(base_pts, dtype=float)
    new_pts  = np.asarray(new_pts,  dtype=float)
    if base_pts.size == 0: 
        return voxel_downsample(new_pts, voxel)
    if new_pts.size == 0:  
        return voxel_downsample(base_pts, voxel)
    return voxel_downsample(np.vstack([base_pts, new_pts]), voxel)

def flatten_recon_pcd(recon_pcd) -> np.ndarray:
    """
    把重建器可能的 list[Mi×3] 或 (N,3) 统一摊平成 (N,3) float 数组。
    仅用于你的“预测输入”，不要回写到 env.recon.pcd。
    """
    if isinstance(recon_pcd, list):
        parts = []
        for seg in recon_pcd:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
                parts.append(arr[:, :3])
        return np.vstack(parts) if parts else np.empty((0,3), dtype=float)
    arr = np.asarray(recon_pcd, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    return np.empty((0,3), dtype=float)



def main():
    args = parse_args()
    rl_cfg = load_config(args.config)
    env, fig, ax = init_environment(rl_cfg)

    VOXEL      = 0.0015
    COV_TARGET = 90.0    # 百分数口径
    MAX_ITERS  = 9
    
    logger = RLLogger(csv_path=args.log_csv, cov_stride=args.cov_stride, echo=args.echo)
    path_viz = PathVisualizer(
        enabled=args.viz_path,
        stride=args.viz_stride,
        width=args.viz_width,
        lifetime=args.viz_lifetime,
        color_mode=args.viz_color,
        max_segments=args.viz_max_segments
    )
    # —— 初始全局点云（仅供预测用！）——
    npy_path    = "H:\projects\AcTExplore\output\cam\paitical_visible_from_camera.npy"
    # "H:\projects\AcTExplore\output\cam/paitical_visible_from_camera.npy"
    pcd_initial = load_initial_pointcloud(env, rl_cfg, npy_path)
    #pcd_global  = voxel_downsample(pcd_initial, VOXEL)   # ✅ 你自己的“预测用点云”
    pcd_global = pcd_initial
    print(f"> 初始预测用点云: {len(pcd_global)}")

    try:
        curr_flat = flatten_recon_pcd(getattr(env.recon, 'pcd', []))
        VOXEL = 0.0015  # 与上面保持一致
        seeded = merge_dedup(curr_flat, pcd_initial, VOXEL)
        env.recon.pcd = seeded
        print(f"[seed] recon.pcd: {len(curr_flat)} → {len(seeded)} points")
    except Exception as e:
        print(f"[warn] seeding recon with initial visible cloud failed: {e}")

    # —— 覆盖率（库内部已有初始可见点）——
    cov = compute_cov_act(env)
    print(f"> 初始覆盖率(库): {cov:.2f}% (目标 {COV_TARGET:.0f}%)")

    logger.log_baseline(env, cov=cov)

    for it in range(1, MAX_ITERS + 1):
        print(f"\n==== Iteration {it}/{MAX_ITERS} ====")

        # 1) 用“你的全局点云”做预测（仅供 multi_projection）
        pcd_dense = densify_via_knn(pcd_global, knn=10, num_interp=3, max_dist=0.02)
        preds, normals = predict_touch_points(pcd_dense, args)

        # 2) 执行探索（期间 recon_tick 会把帧段 append 进 env.recon）
        _ = execute_touch_sequence(env, preds, normals, args, logger=logger, path_viz=path_viz)

        # 3) 覆盖率：强制全量重算
        cov = compute_cov_act(env)
        print(f"> 探索后覆盖率(库): {cov:.2f}%")

        # 4) 更新“你的全局点云”（仅供下一轮预测，不写回 recon）
        recon_all = flatten_recon_pcd(env.recon.pcd)   # 把库里累计的真实点摊平
        pcd_clean, _ = cull_by_bullet_distance(
            recon_all, obj_body_id=env.obj.id,
            thresh=max(args.dist_cull, 0.002),      # 兜底阈值，避免误删
            probe_r=max(args.probe_radius, 0.0005)
        )
        old = len(pcd_global)
        pcd_global = merge_dedup(pcd_global, pcd_clean, VOXEL)
        print(f"> 合并(仅预测): 全局旧={old} 新净={len(pcd_clean)} → 全局新={len(pcd_global)}")

        if cov >= COV_TARGET:
            print("[完成] 覆盖率达标，停止迭代。")
            break
    logger.close()

    final_pts = flatten_recon_pcd(env.recon.pcd)
    out_path = 'outputs/recon/sliding_5.npy'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, final_pts)
    print(f'[save] final recon cloud: {final_pts.shape[0]} points -> {out_path}')     

    finalize(env)

if __name__ == '__main__':
    main()
