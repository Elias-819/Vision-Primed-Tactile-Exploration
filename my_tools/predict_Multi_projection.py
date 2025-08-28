import argparse 
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import alphashape
import shapely.geometry as geom

def create_lineset(starts: np.ndarray,
                   ends: np.ndarray,
                   color=(0,1,0)) -> o3d.geometry.LineSet:
    pts = np.vstack([starts, ends])
    n   = len(starts)
    lines = [[i, i+n] for i in range(n)]
    colors = [color for _ in range(n)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

def densify_via_knn(pts: np.ndarray,
                    knn: int = 10,
                    num_interp: int = 5,
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

def multi_projection(points: np.ndarray,
                     num_interp: int = 10,
                     step: float = 0.005,
                     knn_normals: int = 30):
    """
    对输入点云 points 做多视角边缘外推，使用多尺度法线估计提高鲁棒性。
    并在全局上保证所有表面法线都朝向质心。

    参数：
      points       (N×3) — 输入点云
      num_interp     int — 每对边界点之间插值数
      step        float — 外推步长
      knn_normals    int — 法线估计的 KNN 数量

    返回：
      preds        (M×3) — 所有预测点
      pred_dirs    (M×3) — 外推方向 d（垂直于切线）
      surf_normals (M×3) — 表面法向量 n0（统一朝向质心）
      edge_points  (K×3) — 边界点
      arrow_lines (M,2,3) — 从插值点到预测点的线段，用于可视化
    """
    # —— 1. 边界提取 —— #
    axes_list = [(0,1), (1,2), (0,2)]
    bset = set()
    for axes in axes_list:
        proj = points[:, axes]
        hull = ConvexHull(proj)
        bset.update(hull.vertices.tolist())
    boundary_indices = sorted(bset)
    edge_points = points[boundary_indices]
    
    # —— 2. 多尺度法线估计并融合 —— #
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    scales = [max(5, knn_normals//2), knn_normals, knn_normals*2]
    normals_list = []
    for k in scales:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        normals_list.append(np.asarray(pcd.normals).copy())

    normals_stack = np.stack(normals_list, axis=0)  # [3, N, 3]
    weights = np.array([0.5, 0.3, 0.2], dtype=float)
    weights /= weights.sum()
    normals = np.tensordot(weights, normals_stack, axes=([0],[0]))  # [N,3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals /= norms  # 归一化后的单位法线

    # —— 3. 全局翻转：保证所有法线都朝向质心 —— #
    center = np.mean(points, axis=0)
    vec_to_center = center[np.newaxis, :] - points          # 每点到质心的向量
    dots = np.einsum('ij,ij->i', normals, vec_to_center)    # 点积
    epsilon = 1e-5
    normals[dots <= epsilon] *= -1                          # 小于等于阈值也翻转

    # —— 4. 对每个边界点做插值并外推 —— #
    preds        = []
    pred_dirs    = []
    surf_normals = []
    arrow_lines  = []

    for idx in boundary_indices:
        p0 = points[idx]
        n0 = normals[idx]  # 现在是已统一朝向质心的表面法线

        # 找最近的另一个边界点 p1
        dists = np.linalg.norm(edge_points - p0, axis=1)
        dists[dists < 1e-8] = np.inf
        nei = np.argmin(dists)
        if np.isinf(dists[nei]): 
            continue
        p1 = edge_points[nei]
        t = p1 - p0
        if np.linalg.norm(t) < 1e-8:
            continue
        t /= np.linalg.norm(t)

        for j in range(num_interp + 2):
            r = j / (num_interp + 1)
            pint = p0 + r * (p1 - p0)

            # 计算外推方向 d = -(n0 × t)，并归一化
            d = -np.cross(n0, t)
            if np.linalg.norm(d) > 0:
                d /= np.linalg.norm(d)

            # “向内”校正：保证 d 指向体内
            if np.linalg.norm(pint + step * d - center) < np.linalg.norm(pint - center):
                d = -d

            pred = pint + step * d

            preds.append(pred)
            pred_dirs.append(d)
            surf_normals.append(n0)
            arrow_lines.append((pint, pred))

    return (
        np.array(preds),        # M×3
        np.array(pred_dirs),    # M×3
        np.array(surf_normals), # M×3
        edge_points,            # K×3
        np.array(arrow_lines)   # M×2×3
    )

def visualize_all_new(orig_pts: np.ndarray,
                  new_pts: np.ndarray,
                  edge_pts: np.ndarray,
                  preds: np.ndarray,
                  arrow_lines: np.ndarray,
                  pred_normals: np.ndarray):
    """
    orig_pts — 原始稀疏点，灰色；
    new_pts  — 补充进来的新点，黄色；
    edge_pts — 边界点，蓝色；
    preds    — 外推预测点，红色；
    arrow_lines — 预测箭头，绿色；
    pred_normals — 预测点上的法线箭头，蓝色。
    """
    geoms = []
    # 原始点云（灰）
    pcd_orig = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(orig_pts))
    pcd_orig.paint_uniform_color([0.6,0.6,0.6])
    geoms.append(pcd_orig)
    # 新增密点（黄）
    pcd_new = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_pts))
    pcd_new.paint_uniform_color([1.0,0.8,0.0])  # 黄色
    geoms.append(pcd_new)
    # 边界点（蓝）
    edge = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(edge_pts))
    edge.paint_uniform_color([0,0,1])
    geoms.append(edge)
    # 预测点（红）
    pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(preds))
    pred_pcd.paint_uniform_color([1,0,0])
    geoms.append(pred_pcd)
    # 箭头线段（绿）
    if arrow_lines.size > 0:
        starts = arrow_lines[:,0,:]
        ends   = arrow_lines[:,1,:]
        ls = create_lineset(starts, ends, color=[0,1,0])
        geoms.append(ls)
    # 预测法线（蓝）
    if pred_normals is not None and len(preds)>0:
         length = 0.01
         n_starts = preds
         n_ends   = preds + pred_normals * length
         ls_normal = create_lineset(n_starts, n_ends, color=[0,0,1])
         geoms.append(ls_normal)

    o3d.visualization.draw_geometries(geoms)

def main():
        # —— 硬编码输入输出路径和参数 —— #
    input_pcd    = "output/visible_from_camera.pcd"       # 你的输入点云文件
    output_pcd   = "output/predicted.pcd"  # 你希望写出的预测点云
    normals_txt  = "output/normals.txt"    # 你希望写出的法线文件

    num_interp   = 0
    step         = 0.005
    knn_normals  = 30

    # KNN 补点参数
    knn_densify    = 8     # 每点找 8 个邻居
    num_interp_den = 3     # 每对连线插 3 个点
    max_dist       = 0.02  # 最远只插距 2cm 内的邻居，可根据需要调

    # 1) 读点云
    pcd = o3d.io.read_point_cloud(input_pcd)
    orig_pts = np.asarray(pcd.points)
    pts = np.asarray(pcd.points)
    # —— （可选）先补点 —— #
    pts_dense = densify_via_knn(
        orig_pts,
        knn=knn_densify,
        num_interp=num_interp_den,
        max_dist=max_dist
    )
    # 拆分：新点 = pts_dense[len(orig_pts):]
    new_pts = pts_dense[len(orig_pts):]

    # 2) 外推
    preds, pred_dirs, surf_normals, edge_pts, arrow_lines = multi_projection(
        pts_dense,
        num_interp=num_interp,
        step=step,
        knn_normals=knn_normals
    )
    # 3) 写文件
    out_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(preds))
    o3d.io.write_point_cloud(output_pcd, out_pcd)
    np.savetxt(normals_txt, surf_normals)
    print(f"预测完成，结果写入：\n  点云 → {output_pcd}\n  法线 → {normals_txt}")
    # 5) 可视化：传入 orig_pts, new_pts
    visualize_all_new(orig_pts, new_pts, edge_pts, preds, arrow_lines, surf_normals)

if __name__ == "__main__":
    main()