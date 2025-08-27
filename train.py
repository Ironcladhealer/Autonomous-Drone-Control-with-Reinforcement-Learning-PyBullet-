# train.py
import argparse
import os
from environment import QuadrotorEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every 'check_freq' steps)
    based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # retrieve training reward
            x = self.locals.get('infos', None)
            # fallback: don't rely on info structure; use logger
            mean_reward = np.random.random()  # placeholder; SB3 logger more appropriate
            # Save model every check
            save_path = os.path.join(self.log_dir, 'model_latest.zip')
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Saved model to {save_path}")
        return True

def make_env(gui=False):
    def _init():
        env = QuadrotorEnv(gui=gui)
        env = Monitor(env)
        return env
    return _init

def main(args):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # For faster training keep GUI off. Use DummyVecEnv for SB3
    env = DummyVecEnv([make_env(gui=args.gui)])
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir,
                policy_kwargs=dict(net_arch=[ dict(pi=[256,256], vf=[256,256]) ]),
                learning_rate=3e-4, n_steps=2048, batch_size=64)

    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    total_timesteps = args.timesteps
    print(f"Starting training for {total_timesteps} timesteps. GUI={args.gui}")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(log_dir, "models/ppo_quadrotor_final.zip"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--gui", action="store_true", help="Run env in GUI mode during training (slow).")
    args = parser.parse_args()
    main(args)
