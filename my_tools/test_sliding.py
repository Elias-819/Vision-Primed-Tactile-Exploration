#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_to_pred.py — Refactored into模块化函数
在 AcTExplore 环境中，使用多视角点云外推算法驱动触觉传感器移动
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
    parser.add_argument('--max_cycles',  type=int,   default=5,     help='最多循环轮数')
    parser.add_argument('--min_gain',    type=float, default=0.3,   help='每轮覆盖率最小增益(百分点)；低于此值就停止')
    parser.add_argument('--target_cov',  type=float, default=98.0,  help='目标覆盖率，达到即停止')
    parser.add_argument('--voxel',       type=float, default=0.002, help='预测前体素下采样尺寸(米)，<=0禁用')
    parser.add_argument('--gt_snap',   type=float, default=0.003, help='吸附到GT的半径(米)：小于此距离的点吸到最近GT上')
    parser.add_argument('--gt_reject', type=float, default=0.004, help='剔除阈值(米)：大于此距离的点视为悬浮点并删除')
    parser.add_argument('--clean_each_cycle', action='store_true', help='每轮探索后都做一次基于GT的清洗/吸附')
    parser.add_argument('--min_touch_force', type=float, default=1.0,help='只有总力≥该值时才允许入库（N）')
    return parser.parse_args()

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

# —— 覆盖率日志容器 ——
coverage_log = []

def coverage_tick(env, annotate_ax=None, tag=None):
    """
    计算并记录当前覆盖率（累计）。可选在图上加标题注释。
    """
    try:
        # 只增量处理“新加入”的触点，内部已做 last_index_acc 优化
        env.recon.compute_coverage(accumulate=True, visualize=False)
        cov = float(getattr(env.recon, "coverage", 0.0))
    except Exception as e:
        cov = 0.0

    # 记录：时间戳 + （若有）环境步数 + 覆盖率
    step_idx = getattr(env, 'horizon_counter', 0)
    coverage_log.append((time.time(), step_idx, cov))

    # 可选：把覆盖率写到传入的 Matplotlib 坐标轴标题里
    if annotate_ax is not None:
        title = f"Coverage {cov:.2f}%"
        if tag:
            title += f" | {tag}"
        annotate_ax.set_title(title)

    return cov

def visualize_pressure_map(env, subtitle=None):
    """
    在同一个名为 "Pressure Map" 的窗口里，清空上一次内容，
    然后渲染新的压力热图，并在标题里显示当前覆盖率。
    """
    # 从 sensor 取数据
    _, rel_depths, _ = env.digits.render()
    pm = rel_depths[0]

    # 计算覆盖率（增量），安全起见做 try/except
    cov_text = ""
    try:
        if hasattr(env, "recon"):
            env.recon.compute_coverage(accumulate=True, visualize=False)
            cov = float(getattr(env.recon, "coverage", 0.0))
            cov_text = f" — coverage: {cov:.2f}%"
    except Exception:
        pass

    # 彻底清空 figure（包括所有 axes, colorbar）
    fig = plt.figure("Pressure Map")
    fig.clf()

    # 重新建 axes、画图
    ax = fig.add_subplot(111)
    vmax = float(pm.max()) if pm.size else 0.0
    im = ax.imshow(pm, cmap="hot", vmin=0.0, vmax=max(vmax, 1e-9))

    title = "Pressure Map"
    if subtitle:
        title += f" — {subtitle}"
    title += cov_text
    ax.set_title(title)

    # 新建 colorbar
    fig.colorbar(im, ax=ax, label="Deformation (m)")
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
            visualize_pressure_map(env, subtitle="Pressure Map at first touch")
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
                           lineColorRGB=[1,0,0],
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
                           lineColorRGB=[1,0,0],
                           lineWidth=2,
                           lifeTime=life_time)
        p.addUserDebugLine(end.tolist(),
                           right_end.tolist(),
                           lineColorRGB=[1,0,0],
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
                  color='r',
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

def recon_tick(env):
    # 确保 curr_pos/curr_ori 是最新的（_get_observation 里会用到）
    env.curr_pos, env.curr_ori = np.array(p.getBasePositionAndOrientation(env.digit_body.id))
    _ = env._get_observation()   # 内部会按接触条件把点云插入 recon

def slide_between(env, start_pt, start_n, 
                  end_pt, end_n,
                  gel_offset_local, curr_ori,
                  slide_step: float, pause: float,
                  show_pressure: bool = True,
                  vis_stride: int = 5,
                  min_touch_force: float = 1.0):  # <<< 新增：接触入库的最小力阈值
    """
    在 start_pt→end_pt 之间平滑滑动（恒压微调）；
    仅在“确实接触(总力≥min_touch_force)”时入库，避免悬浮片层。
    返回：end_ori, end_base
    """

    # —— 1) 起止姿态
    ori_start = align_ori_to_normal(curr_ori, start_n)
    ori_end   = align_ori_to_normal(ori_start, end_n)

    # —— 1.1) 更稳的增压方向判定（±eps 双向探测，取增力更大者）
    def robust_push_sign(pt_w, n_w, ori_w, eps=1.0e-3):  # 1mm 试探
        base0 = compute_base_position(pt_w, gel_offset_local, ori_w)
        env.digit_body.set_base_pose(base0, ori_w); p.stepSimulation()
        F0 = total_force(env)

        base_p = compute_base_position(pt_w + eps * n_w, gel_offset_local, ori_w)
        env.digit_body.set_base_pose(base_p, ori_w); p.stepSimulation()
        Fp = total_force(env)

        base_m = compute_base_position(pt_w - eps * n_w, gel_offset_local, ori_w)
        env.digit_body.set_base_pose(base_m, ori_w); p.stepSimulation()
        Fm = total_force(env)

        # 复位
        env.digit_body.set_base_pose(base0, ori_w); p.stepSimulation()
        # 哪个方向增力更大取哪侧；若都不增力，默认朝 +n
        return +1 if (Fp - F0) >= (Fm - F0) else -1

    push_sign = robust_push_sign(start_pt, start_n, ori_start)

    # —— 1.2) 段首“安全抬升”
    safe_pt   = start_pt - push_sign * PRE_CLEAR * start_n
    base_safe = compute_base_position(safe_pt, gel_offset_local, ori_start)
    env.digit_body.set_base_pose(base_safe, ori_start); p.stepSimulation()

    # —— 1.3) 段首“寻触”：先把力拉到死区或至少达到 min_touch_force
    calib_iters = 0
    while calib_iters < 25:
        F0 = total_force(env)
        if (F0 >= min_touch_force) and (abs(F_TARGET - F0) < F_TOL):
            break
        err0   = max(0.0, F_TARGET - F0)  # 只在欠压时推进，过压不再加深
        delta0 = np.clip(KP_DEPTH * err0, 0.0, D_STEP_MAX)
        if delta0 <= 1e-6:  # 没有推进空间
            break
        safe_pt = safe_pt + push_sign * delta0 * start_n
        base0   = compute_base_position(safe_pt, gel_offset_local, ori_start)
        env.digit_body.set_base_pose(base0, ori_start); p.stepSimulation()
        calib_iters += 1
        if show_pressure and (calib_iters % 3 == 0):
            visualize_pressure_map(env, subtitle=f"Calib F={F0:.2f}N")

    # —— 2) 计算步数 & SLERP
    tip_dist = np.linalg.norm(end_pt - start_pt)
    steps    = max(1, int(np.ceil(tip_dist / max(1e-6, slide_step))))
    key_rots = R.from_quat([ori_start, ori_end])
    slerp    = Slerp([0, 1], key_rots)

    # —— 3) 微步滑动：插值 + 简单恒压
    for k in range(1, steps + 1):
        a = k / steps
        pt_nom = (1 - a) * start_pt + a * end_pt
        n_nom  = lerp_normal(start_n, end_n, a)
        ori_k  = slerp(a).as_quat().tolist()

        F = total_force(env)

        # 3.1 紧急退让
        if F > F_EMERGENCY:
            pt_emg  = pt_nom - push_sign * EMERGENCY_STEP * n_nom
            base_emg= compute_base_position(pt_emg, gel_offset_local, ori_k)
            env.digit_body.set_base_pose(base_emg, ori_k); p.stepSimulation()
            if show_pressure:
                visualize_pressure_map(env, subtitle=f"EMERG! F={F:.1f}N → retreat")
            time.sleep(min(0.01, pause / max(steps, 1)))
            continue

        # 3.2 恒压修正（仅欠压时推进；过压时不再增加法向深度）
        err = F_TARGET - F
        if abs(err) < F_TOL:
            delta_d = 0.0
        else:
            # 欠压：正向推进；过压：允许微退
            delta_d = np.clip(KP_DEPTH * err, -D_STEP_MAX, D_STEP_MAX)

        pt_corr = pt_nom + push_sign * delta_d * n_nom
        base_k  = compute_base_position(pt_corr, gel_offset_local, ori_k)
        env.digit_body.set_base_pose(base_k, ori_k); p.stepSimulation()

        # 3.3 可视化
        if show_pressure:
            if (k % vis_stride == 0) or (abs(err) > 2 * F_TOL) or (k == steps):
                tag = "High" if err < -F_TOL else ("Low" if err > F_TOL else "OK")
                visualize_pressure_map(env, subtitle=f"{k}/{steps} | F={F:.2f}N | {tag}")

        # 3.4 ★ 仅在“确实接触”时入库（关键改动）
        if (k % 2 == 0) or (k == steps):
            if total_force(env) >= min_touch_force:
                # 仅此时才写入重建，避免近场/未触点铺成“悬浮薄壳”
                env.curr_pos, env.curr_ori = np.array(p.getBasePositionAndOrientation(env.digit_body.id))
                _ = env._get_observation()

        # 3.5 （可选）覆盖率打点
        if (k % 10 == 0) or (k == steps):
            coverage_tick(env)

        time.sleep(min(0.01, pause / max(steps, 1)))

    # —— 4) 结束位姿
    base_end = compute_base_position(end_pt, gel_offset_local, ori_end)
    return ori_end, base_end


def execute_touch_sequence(env, preds, normals, args):

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
                show_pressure=False,   # 打开可视化
                vis_stride=5,          # 每 5 步刷一帧；卡顿就调大
                min_touch_force=args.min_touch_force
            )

        print(f"[{i+1}/{n_pts}] segment done.")

    return np.asarray(env.recon.pcd)

def voxel_downsample_np(pts: np.ndarray, voxel: float) -> np.ndarray:
    if pts is None or len(pts) == 0 or voxel <= 0:
        return pts
    # 以最小点为原点，对应体素网格编号
    origin = pts.min(axis=0)
    grid = np.floor((pts - origin) / voxel).astype(np.int64)
    _, uniq_idx = np.unique(grid, axis=0, return_index=True)
    return pts[np.sort(uniq_idx)]

def iterative_explore(env, seed_pcd, args):
    """
    以 seed_pcd 为初始点云，循环执行：预测→闭环滑动采样→合并点云→评估覆盖率。
    终止条件：达到 target_cov，或单轮 coverage 增益 < min_gain，或超过 max_cycles。
    """
    # 初始覆盖率
    prev_cov = coverage_tick(env, tag="init")

    curr_pcd = seed_pcd.copy()
    for cycle in range(1, args.max_cycles + 1):
        # 1) 作为预测输入：可选体素下采样 + 稍微加密（KNN 插值已有）
        pcd_for_pred = voxel_downsample_np(curr_pcd, args.voxel)
        pcd_dense    = densify_via_knn(
            pcd_for_pred,
            knn=10,            # 也可以拉到 args 里
            num_interp=3,
            max_dist=0.02
        )

        # 2) 基于当前累计点云做新一轮预测
        preds, normals = predict_touch_points(pcd_dense, args)

        # 3) 围边界闭环滑动采样（内部会不断触发 recon_tick → insert_data 合并新点）
        execute_touch_sequence(env, preds, normals, args)

        # 4) 合并结果：直接从重建器里取“最新累计点云”
        curr_pcd = np.asarray(env.recon.pcd)

        # ★★★ 基于GT的清洗/吸附（去悬浮点+贴面） ★★★
        if args.clean_each_cycle:
            try:
                curr_pcd = clean_and_snap_to_gt(env, curr_pcd, args.gt_snap, args.gt_reject)
                env.recon.pcd = curr_pcd
                reset_and_recompute_lib_coverage(env)  # 库内覆盖率用“干净点云”重算
            except Exception as e:
                print("[WARN] GT-clean this cycle failed:", e)

        # 5) 评估覆盖率与停止条件
        curr_cov = coverage_tick(env, tag=f"cycle {cycle}")
        gain = curr_cov - prev_cov
        print(f"[cycle {cycle}] coverage={curr_cov:.2f}%  gain=+{gain:.2f}pp")

        if curr_cov >= args.target_cov:
            print(">> target coverage reached, stop.")
            break
        if gain < args.min_gain:
            print(">> marginal gain below threshold, stop.")
            break

        prev_cov = curr_cov

    return curr_pcd

def get_gt_points(env):
    """
    从 env/recon 中寻找全量GT采样点（世界坐标）。请按你库的实际字段名调整 cand_names。
    """
    cand_names = ['gt_points_world', 'gt_points', 'gt_pts', 'gt_xyz']
    for n in cand_names:
        pts = getattr(env.recon, n, None)
        if pts is not None and len(pts) > 0:
            return np.asarray(pts)
    raise RuntimeError("未在 env.recon 找到GT点，请把真实字段名填入 get_gt_points().")

def clean_and_snap_to_gt(env, pcd: np.ndarray, snap_r: float, reject_r: float) -> np.ndarray:
    """
    基于GT最近邻的两步操作：
      1) 剔除：距GT > reject_r 的点（悬浮/误触）
      2) 吸附：距GT ≤ snap_r 的点，直接替换成最近GT点坐标（把小偏差贴到表面）
    其余点保持原位（通常是 snap_r~reject_r 间的“边缘点”）
    """
    if pcd is None or len(pcd) == 0:
        return pcd
    gt = get_gt_points(env)
    tree = cKDTree(gt)
    d, j = tree.query(pcd, k=1)
    keep = d <= reject_r
    if not np.any(keep):
        return np.empty((0,3))
    pcd2 = pcd[keep].copy()
    d2   = d[keep]; j2 = j[keep]
    snap_mask = d2 <= snap_r
    if np.any(snap_mask):
        pcd2[snap_mask] = gt[j2[snap_mask]]  # 直接吸到GT上
    # 可选：轻体素去重，避免覆盖统计重复点过多
    pcd2 = voxel_downsample_np(pcd2, voxel=0.0015)
    return pcd2

def reset_and_recompute_lib_coverage(env):
    """
    修改了 env.recon.pcd 后，重置增量游标并用库函数“全量重算一次覆盖率”。
    """
    if hasattr(env.recon, "last_index_acc"):
        env.recon.last_index_acc = 0
    env.recon.compute_coverage(accumulate=False, visualize=False)

def prime_total_coverage_with_seed(env, seed_pcd: np.ndarray, voxel: float = 0.0):
    """
    将初始点云并入 env.recon.pcd，并重算一次覆盖率（作为“总覆盖率”的基线）。
    voxel>0 时会先做一次轻体素下采样，减少重复点。
    """
    if seed_pcd is None or len(seed_pcd) == 0:
        return
    seed = voxel_downsample_np(seed_pcd, voxel) if voxel > 0 else seed_pcd.copy()

    # 并入累计点云
    if getattr(env.recon, "pcd", None) is not None and len(env.recon.pcd) > 0:
        env.recon.pcd = np.vstack([np.asarray(env.recon.pcd), seed])
    else:
        env.recon.pcd = seed

    # 重置增量游标，让覆盖从头计算一次
    if hasattr(env.recon, "last_index_acc"):
        env.recon.last_index_acc = 0

    # 计算“总覆盖率”基线（不增量）
    try:
        env.recon.compute_coverage(accumulate=False, visualize=False)
        print(f">> primed total coverage with seed: {getattr(env.recon,'coverage',0.0):.2f}%")
    except Exception as e:
        print("[WARN] prime_total_coverage_with_seed failed:", e)

def finalize(env):
    try:
        env.recon.visualize_result()
    except:
        pass
    env.close()

def main():
    args = parse_args()
    rl_cfg = load_config(args.config)
    env, fig, ax = init_environment(rl_cfg)

    npy_path = "output/visible_from_camera.npy"
    pcd_init = load_initial_pointcloud(env, rl_cfg, npy_path)

    # 把相机已有点云并入重建器，并把覆盖率基线设成“相机+触觉”的总覆盖
    prime_total_coverage_with_seed(env, pcd_init, voxel=args.voxel)

    # 进入循环式探索
    _ = iterative_explore(env, pcd_init, args)
    finalize(env)

if __name__ == '__main__':
    main()
