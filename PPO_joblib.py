import os
import hydra
from stable_baselines3 import PPO
import collections
import collections.abc
from joblib import Parallel, delayed  # 导入 joblib 来进行并行
from env import TactoEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from omegaconf import OmegaConf
import numpy as np
from sb3_contrib import RecurrentPPO
import pybullet as p  # 导入 pybullet


# 下面三行按需补齐所有被移走的名字
collections.Mapping        = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence       = collections.abc.Sequence


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get("dones")[-1]:
            self.logger.record("episode/coverage", self.locals.get("infos")[0]["coverage"])
            self.logger.record("episode/reward", self.locals.get("infos")[0]["acc_reward"])
            self.logger.record("episode/length", self.locals.get("infos")[0]["horizon_counter"])
        return True

def train_single_model(cfg, model_id):
    # 初始化pybullet，无GUI模式（确保每个并行进程都独立初始化）
    p.connect(p.DIRECT)  # 以DIRECT模式连接pybullet服务器（无GUI）
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 禁用GUI显示

    # 创建环境
    env = TactoEnv(cfg)
    # 为每个模型创建独立的保存路径
    save_dir = os.path.join('Training', 'Logs', f"{cfg.RL.save_dir}_{model_id}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    OmegaConf.save(cfg, os.path.join(save_dir, "RL.yaml"))
    
    # 设置checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.RL.save_freq,
        save_path=save_dir,
        name_prefix="model",
    )
    
    # 加载预训练模型或创建新模型
    if cfg.RL.pretrain_model_path:
        if cfg.RL.algorithm == "RecurrentPPO":
            model = RecurrentPPO.load(cfg.RL.pretrain_model_path, env, verbose=1, tensorboard_log=save_dir)
        else:
            model = PPO.load(cfg.RL.pretrain_model_path, env, verbose=1, tensorboard_log=save_dir)
    else:
        if cfg.RL.algorithm == "RecurrentPPO":
            model = RecurrentPPO(cfg.RL.policy_network, env, verbose=1, tensorboard_log=save_dir,
                                 policy_kwargs=dict(normalize_images=False), n_steps=1024)
        else:
            model = PPO(cfg.RL.policy_network, env, verbose=1, tensorboard_log=save_dir,
                        policy_kwargs=dict(normalize_images=False), n_steps=1024)
    
    coverage_callback = TensorboardCallback()
    
    # 开始训练
    model.learn(total_timesteps=100e4, callback=[checkpoint_callback, coverage_callback])

    # 训练完成后断开与PyBullet服务器的连接
    p.disconnect()

@hydra.main(config_path="conf", config_name="RL")
def train(cfg):
    # 使用 Joblib 来并行化多个模型的训练
    num_models = 4  # 假设并行训练 4 个模型，可以根据需要调整
    Parallel(n_jobs=num_models)(  # 运行并行训练
        delayed(train_single_model)(cfg, model_id) for model_id in range(num_models)
    )

if __name__ == "__main__":
    train()
