# test_eval.py — 纯测评：不训练，只 rollout 并记录时间与位姿
import os
import time
import csv
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from env import TactoEnv
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# 读取位姿 & 画线（若可用）
try:
    import pybullet as p
except Exception:
    p = None

# 可选：TensorBoard（如未安装会自动跳过）
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ---------- 工具：动作/空间/步进兼容 ----------
def _is_discrete(space):
    try:
        from gymnasium.spaces import Discrete as GDiscrete
    except Exception:
        GDiscrete = None
    try:
        from gym.spaces import Discrete as ODiscrete
    except Exception:
        ODiscrete = None
    return (
        (GDiscrete is not None and isinstance(space, GDiscrete)) or
        (ODiscrete is not None and isinstance(space, ODiscrete)) or
        hasattr(space, "n")
    )

def _format_action(env, action):
    """离散空间把 ndarray/list/标量规范成 Python int；连续空间原样返回"""
    if _is_discrete(env.action_space):
        if isinstance(action, (list, tuple)):
            action = action[0]
        if hasattr(action, "shape"):
            arr = np.asarray(action)
            return int(arr.flat[0])
        return int(action)
    return action

def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}

def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:  # gymnasium
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:              # gym
        obs, reward, done, info = out
    return obs, reward, done, info

def _get_horizon(cfg, default_steps=5000):
    try:
        return int(cfg.termination.horizon_length)
    except Exception:
        return default_steps

def _extract_coverage(env, info, default=0.0):
    if isinstance(info, dict):
        v = info.get("coverage", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    for attr in ("coverage", "cov", "current_coverage"):
        if hasattr(env, attr):
            try:
                return float(getattr(env, attr))
            except Exception:
                pass
    if hasattr(env, "get_coverage"):
        try:
            return float(env.get_coverage())
        except Exception:
            pass
    return float(default)


# ---------- 位姿读取（仅改 test，不动 env） ----------
def _sensor_pose(env):
    """
    从 env 读取传感器位姿（世界坐标），优先 link，退回 base。
    返回 (pos_xyz[np.array(3)], ori_xyzw[np.array(4)])；若 pybullet 不可用会抛错。
    """
    if p is None:
        raise RuntimeError("pybullet 不可用，无法在纯测评中读取位姿。")

    body_id = None
    if hasattr(env, "digit_body") and hasattr(env.digit_body, "id"):
        body_id = int(env.digit_body.id)
    elif hasattr(env, "robot_id"):
        body_id = int(env.robot_id)
    elif hasattr(env, "body_id"):
        body_id = int(env.body_id)
    if body_id is None:
        raise RuntimeError("未能找到传感器刚体 ID（尝试 env.digit_body.id / env.robot_id / env.body_id 失败）。")

    tip_link = getattr(env, "digit_tip_link", None)
    if tip_link is not None:
        ls = p.getLinkState(body_id, int(tip_link), computeForwardKinematics=True)
        pos, ori = np.array(ls[4], float), np.array(ls[5], float)
    else:
        pos_b, ori_b = p.getBasePositionAndOrientation(body_id)
        pos, ori = np.array(pos_b, float), np.array(ori_b, float)
    return pos, ori


# ---------- 纯测评主函数 ----------
@hydra.main(config_path="conf", config_name="test")
def test(cfg: DictConfig, num_repeat: int = 1):
    print("Loaded test config:\n", OmegaConf.to_yaml(cfg))
    env = TactoEnv(cfg)

    # 加载模型（不训练）
    if cfg.RL.algorithm == "RecurrentPPO":
        model = RecurrentPPO.load(cfg.RL.pretrain_model_path, env)
        is_recurrent = True
    else:
        model = PPO.load(cfg.RL.pretrain_model_path, env)
        is_recurrent = False

    # Eval 选项（可在 conf/test.yaml 里加 eval: {deterministic: true, log_every_seconds: 0}）
    deterministic = bool(getattr(getattr(cfg, "eval", {}), "deterministic", True))
    log_every_seconds = float(getattr(getattr(cfg, "eval", {}), "log_every_seconds", 0.0))

    # 输出目录 & CSV
    out_dir = os.path.abspath("./outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "300000_model_5.csv")
    new_file = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_file:
        csv_w.writerow([
            "episode", "step", "elapsed_s", "elapsed_min",
            "pos_x", "pos_y", "pos_z",
            "coverage", "reward", "done"
        ])

    # 可选：TensorBoard
    writer = SummaryWriter(log_dir=out_dir) if SummaryWriter is not None else None
    if writer:
        print(f"[TB] writing to {out_dir}")

    horizon = _get_horizon(cfg, default_steps=5000)

    input("press any key to start")

    for ep in range(num_repeat):
        print(f"\n##### Episode {ep} (pure eval) #####")
        obs, info = _reset_env(env)
        episode_start = True
        lstm_state = None
        step = 0

        # 时间
        t0 = time.perf_counter()
        next_log_t = t0 + log_every_seconds if log_every_seconds > 0 else 0.0

        # 轨迹缓存（用于 path_ep*.csv）
        try:
            pos0, _ = _sensor_pose(env)
        except Exception:
            pos0 = np.array([np.nan, np.nan, np.nan], float)
        traj_xyz = [pos0.tolist()]
        prev_pos = pos0.copy()

        done = False
        total_reward = 0.0
        cov = 0.0

        while not done and step < horizon:
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=np.array([episode_start], dtype=bool),
                    deterministic=deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            act = _format_action(env, action)
            obs, reward, done, info = _step_env(env, act)
            total_reward += float(reward)
            step += 1
            episode_start = done  # 下一回合重置 LSTM

            # 覆盖率、位姿、时间
            cov = _extract_coverage(env, info, default=cov)
            try:
                pos, _ = _sensor_pose(env)
            except Exception:
                pos = np.array([np.nan, np.nan, np.nan], float)

            now = time.perf_counter()
            elapsed_s = now - t0
            elapsed_min = elapsed_s / 60.0

            # 记录轨迹 & 画线
            traj_xyz.append(pos.tolist())
            if p is not None and np.isfinite(prev_pos).all() and np.isfinite(pos).all():
                try:
                    p.addUserDebugLine(prev_pos.tolist(), pos.tolist(), [0, 1, 0], lineWidth=2.0, lifeTime=0)
                except Exception:
                    pass
            prev_pos = pos.copy()

            # 写日志（每步 or 每 N 秒）
            do_log = (log_every_seconds == 0.0) or (now >= next_log_t) or done
            if do_log:
                csv_w.writerow([
                    ep, step, f"{elapsed_s:.3f}", f"{elapsed_min:.3f}",
                    f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                    f"{cov:.6f}", f"{float(reward):.6f}", int(done),
                ])
                csv_f.flush()
                if writer is not None:
                    writer.add_scalar("eval/coverage_time_x", cov, int(elapsed_s))
                    writer.add_scalar("eval/reward_step_x", float(reward), step)
                    writer.add_scalar("eval/pos/x", float(pos[0]), step)
                    writer.add_scalar("eval/pos/y", float(pos[1]), step)
                    writer.add_scalar("eval/pos/z", float(pos[2]), step)
                if log_every_seconds > 0:
                    while next_log_t <= now:
                        next_log_t += log_every_seconds

        # 保存整段路径
        path_csv = os.path.join(out_dir, f"path_ep{ep}.csv")
        with open(path_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["x", "y", "z"])
            w.writerows(traj_xyz)
        print(f"[path] saved to {path_csv}")

        print(f"Episode {ep} finished: steps={step}, "
              f"coverage={cov:.3f}, return={total_reward:.2f}, "
              f"time={elapsed_min:.2f} min")

    csv_f.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    test()
