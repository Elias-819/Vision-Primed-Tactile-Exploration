import open3d as o3d

# 硬编码点云文件路径
PCD_PATH = "output/first_merged.pcd"

# 可视化函数
def visualize_pcd(file_path: str,
                  color: tuple = (0.6, 0.6, 0.6),
                  window_name: str = "PCD Viewer"):
    """
    读取并可视化一个点云文件（PCD/PLY/OBJ 支持）。

    参数：
      file_path:   点云文件路径
      color:       RGB 颜色三元组，范围 [0,1]
      window_name: 窗口标题
    """
    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        print(f"[Error] 无法读取点云或点云为空: {file_path}")
        return

    # 2. 上色
    pcd.paint_uniform_color(color)

    # 3. 可视化
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=800,
        height=600,
        left=50,
        top=50,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    print(f"正在可视化点云: {PCD_PATH}")
    visualize_pcd(PCD_PATH)