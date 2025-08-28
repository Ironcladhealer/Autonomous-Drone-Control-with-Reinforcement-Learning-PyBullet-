# Quadrotor Continuous Control with PPO

This repository implements **Proximal Policy Optimization (PPO)** to train a quadrotor UAV in a **PyBullet physics simulation**. The environment has been designed to be realistic, with **PID-style stabilization** and a **quadrotor URDF** that can perform continuous control tasks such as:

* Hovering at a target position
* Moving in 3D space (up, down, left, right, forward, backward)
* Rotating clockwise/counterclockwise (yaw control)
* Avoiding obstacles while navigating toward a randomly defined target

The agent learns to balance thrust, roll, pitch, and yaw torques to achieve stable and efficient flight.

---

## 🚀 Features

* **Custom PyBullet Environment** (`QuadrotorEnv`)
* **Quadrotor URDF integration** for realistic physics
* **Continuous control actions**: thrust and angular torques
* **Reward shaping** for stable hovering, obstacle avoidance, and target reaching
* **PID-inspired stabilization baseline**
* **Trained using PPO** (from Stable-Baselines3)

---

## 📂 Repository Structure

```
quadrotor-ppo/
│── logs/ 
│   └──models/
│       └── trained_model.zip  # Saved PPO policies (after training)
│── urdf/
│   └── quadrotor.urdf     # Drone model for PyBullet (optional)
│── train.py               # PPO training script
│── environment.py         # Custom PyBullet quadrotor environment
│── test.py                # Policy evaluation and visualization
│── README.md              # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/quadrotor-ppo.git
cd quadrotor-ppo
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

---

## ▶️ Training

To train the PPO agent:

```bash
python train.py
```

Training logs and checkpoints will be saved in the `models/` directory.

---

## 🎮 Testing

To visualize a trained policy:

```bash
python test.py
```

The drone will spawn in the PyBullet simulation, stabilize, and attempt to hover and reach the target position while avoiding obstacles.

---

## 📊 Reward Function

The reward is designed to balance **stability, control, and goal reaching**:

* Negative penalty for deviation from target position
* Negative penalty for large angular deviations (instability)
* Penalty for collisions with obstacles
* Bonus reward for reaching and hovering above target

---

## 🛠️ Dependencies

* Python 3.9+
* [PyBullet](https://pybullet.org)
* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* NumPy, Gym

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 📌 Future Work

* Extend to **multi-agent quadrotor swarm control**
* Add **wind disturbance modeling**
* Deploy learned policy on a **real-world UAV testbed**

---

## 👨‍💻 Author

Developed by **Saim Raza** – Robotics Engineer & Full Stack Developer
