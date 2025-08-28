import numpy as np
import open3d as o3d
import os

def load_xyz(path):
    return np.loadtxt(path)

def save_pcd(pts, path):
    header = (
        "# .PCD v0.7 - point cloud\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(pts)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(pts)}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for x,y,z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def simulate_camera_view(points, R, t, width, height, fx, fy, cx, cy,
                         roi_px=None, roi_rel=None):
    """
    roi_px : (u0, u1, v0, v1)  像素坐标闭区间左闭右开, 0<=u<width, 0<=v<height
    roi_rel: (uc, vc, w, h)    以像素为单位的中心与宽高（可用相对比例计算后传入）
    二者二选一；都为 None 时即全视场。
    """
    pts_cam = (R @ (points - t).T).T
    mask_front = pts_cam[:,2] > 0
    pts_cam = pts_cam[mask_front]
    idxs   = np.where(mask_front)[0]

    u = (pts_cam[:,0]/pts_cam[:,2])*fx + cx
    v = (pts_cam[:,1]/pts_cam[:,2])*fy + cy
    u = np.round(u).astype(int); v = np.round(v).astype(int)

    # 先做视场裁剪
    mask_fov = (u>=0)&(u<width)&(v>=0)&(v<height)

    # 计算 ROI 边界
    if roi_px is not None:
        u0,u1,v0,v1 = roi_px
    elif roi_rel is not None:
        uc, vc, w, h = roi_rel
        u0 = int(uc - w/2); u1 = int(uc + w/2)
        v0 = int(vc - h/2); v1 = int(vc + h/2)
    else:
        u0, u1, v0, v1 = 0, width, 0, height

    # 叠加 ROI 掩膜（保持像素级密度不变）
    mask_roi = (u>=u0)&(u<u1)&(v>=v0)&(v<v1)
    mask = mask_fov & mask_roi

    u = u[mask]; v = v[mask]; zs = pts_cam[mask,2]
    idxs = idxs[mask]

    # 最近深度选择（每个像素只保留最近一点）
    depth = {}
    for px,py,z,idx in zip(u,v,zs,idxs):
        key = (px,py)
        if key not in depth or z < depth[key][0]:
            depth[key] = (z, idx)
    visible_idx = [val[1] for val in depth.values()]
    return visible_idx


if __name__ == "__main__":
    # 1. 先用 load_xyz，从没有头的 model.pcd 里读出 numpy 点阵
    pts = load_xyz("H:\projects\AcTExplore\objects\ycb\YcbBanana\model.pcd")
    print("总点数:", len(pts))

    # 2. 构造 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 3. 定义相机内外参（同前示例）
    cam_pos   = np.array([0.0, -1.0, 0.5])
    target    = np.array([0.0,  0.0, 0.0])
    up        = np.array([0.0,  0.0, 1.0])
    z_axis    = (target - cam_pos); z_axis /= np.linalg.norm(z_axis)
    x_axis    = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis    = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis])
    t = cam_pos
    width, height = 640, 480
    fx = fy = 525.0
    cx, cy = width/2, height/2

    # 4. 运行深度缓存算法，拿到可见点的索引
    uc, vc = width//2, height//2
    vis_idx = simulate_camera_view(np.asarray(pcd.points),
                                R, t, width, height, fx, fy, cx, cy,
                                roi_rel=(uc, vc, 10, 10))
    visible_pcd = pcd.select_by_index(vis_idx)
    print("可见点数:", len(visible_pcd.points))

    # 5. 可视化 & 保存
    visible_pcd.paint_uniform_color([1.0,0.706,0.0])
    o3d.visualization.draw_geometries([visible_pcd],
        window_name="Simulated Camera View")
    os.makedirs("H:\projects\AcTExplore\output/cam", exist_ok=True)
    save_pcd(np.asarray(visible_pcd.points),
             "H:\projects\AcTExplore\output/cam/paitical_visible_from_camera.pcd")
    print("已保存到 paitical_visible_from_camera.pcd")
