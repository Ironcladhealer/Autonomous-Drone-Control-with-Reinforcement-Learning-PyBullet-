# train.py
import argparse
import os
import numpy as np
from environment import QuadrotorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def make_env(gui=False):
    def _init():
        env = QuadrotorEnv(gui=gui)
        return Monitor(env)   # Monitor logs rewards for evaluation
    return _init


def main(args):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)

    # ✅ Use DummyVecEnv (SB3 requires vectorized envs)
    env = DummyVecEnv([make_env(gui=args.gui)])

    # Define PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    # ✅ Evaluation callback (saves best model automatically)
    eval_env = DummyVecEnv([make_env(gui=False)])  # always eval headless
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "models"),
        log_path=log_dir,
        eval_freq=10000,  # evaluate every 10k steps
        deterministic=True,
        render=False
    )

    total_timesteps = args.timesteps
    print(f"🚀 Starting PPO training for {total_timesteps} timesteps | GUI={args.gui}")

    # Train PPO
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Final save
    final_model_path = os.path.join(log_dir, "models/ppo_quadrotor_final.zip")
    model.save(final_model_path)
    print(f"✅ Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)  # more timesteps for stability
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--gui", action="store_true", help="Run env in GUI mode during training (slower).")
    args = parser.parse_args()
    main(args)
