import numpy as np

def probe_until_touch(env, target_point, normal=None, max_steps=8, step_size=0.003, xy_steps=3, xy_step_size=0.002):
    """
    在目标点附近做小幅度多方向试探，重点是垂直(法线)方向。
    """
    if normal is None:
        normal = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    start = target_point - normal * step_size * (max_steps // 2)

    for i in range(max_steps):
        probe_pos = start + normal * step_size * i
        env.digit_body.set_base_pose(probe_pos, env.curr_ori)
        for _ in range(2):
            import pybullet as p
            p.stepSimulation()
        if hasattr(env, "is_touching"):
            if env.is_touching:
                return True
        elif hasattr(env, "_touch_indicator"):
            if env._touch_indicator():
                return True

    xy_dirs = [np.array([1,0,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,-1,0])]
    for dir_xy in xy_dirs:
        for j in range(1, xy_steps+1):
            probe_pos = target_point + dir_xy * xy_step_size * j
            env.digit_body.set_base_pose(probe_pos, env.curr_ori)
            for _ in range(2):
                import pybullet as p
                p.stepSimulation()
            if hasattr(env, "is_touching"):
                if env.is_touching:
                    return True
            elif hasattr(env, "_touch_indicator"):
                if env._touch_indicator():
                    return True
    return False
