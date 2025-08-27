# test.py
import argparse
import time
import numpy as np
from environment import QuadrotorEnv
from stable_baselines3 import PPO
import os

def run_eval(model_path, episodes=5, render_fps=240):
    env = QuadrotorEnv(gui=True)
    model = PPO.load(model_path)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        start = time.time()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            ep_reward += reward
            # allow real-time rendering (approx.)
            time.sleep(1.0 / render_fps)
        print(f"Episode {ep+1} reward: {ep_reward:.2f}")
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logs/model_latest.zip", help="Path to trained model .zip")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    run_eval(args.model, episodes=args.episodes)
