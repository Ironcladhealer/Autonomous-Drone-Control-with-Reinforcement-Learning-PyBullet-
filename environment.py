# environment.py
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym.spaces import Box
import time
import math
import os

DEG = math.pi / 180.0

class QuadrotorEnv(gym.Env):
    """
    PPO-ready quadrotor with cascaded PID control and obstacle-aware waypoint tracking.

    Action (continuous, [-1, 1]):
        [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]  (m/s, m/s, m/s, rad/s) high-level commands

    Observation:
        pos(3), euler_rpy(3), lin_vel(3), ang_vel(3), rel_target(3), rays(8)  => 23 dims

    Visuals:
        - Loads realistic URDF if available at assets/cf2x/cf2x.urdf
        - Otherwise builds a simple quad body so you always see a drone.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False, max_steps=1500, dt=1.0/240.0, target_bounds=((2.0, 3.0), (-1.0, 1.0), (1.0, 2.0))):
        super().__init__()
        self.gui = gui
        self.dt = dt
        self.max_steps = max_steps
        self.step_counter = 0

        # ====== Physics/client ======
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        # ====== Spaces ======
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + 3 + 8
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)

        # ====== Target bounds (x,y,z ranges) ======
        self.target_bounds = target_bounds  # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
        self.target_position = np.array([2.5, 0.0, 1.5], dtype=np.float32)

        # ====== Quad physical params & controller gains ======
        self.mass = 0.9
        self.arm = 0.11              # arm length (m)
        self.k_yaw = 0.003           # yaw torque coefficient
        self.max_motor = 4.5         # max thrust per rotor (N) ~ small quad scale
        self.g = 9.81

        # Outer loop gains (pos/vel)
        self.kp_pos = np.array([1.6, 1.6, 2.0])
        self.kd_pos = np.array([1.0, 1.0, 1.2])
        self.kp_vel = np.array([1.8, 1.8, 2.5])

        # Attitude loop gains
        self.kp_att = np.array([6.0, 6.0, 3.0])   # roll, pitch, yaw
        self.kd_att = np.array([0.25, 0.25, 0.15])

        # Command limits
        self.v_cmd_max = np.array([1.5, 1.5, 1.0])  # m/s caps for vx,vy,vz
        self.yaw_rate_cmd_max = 60 * DEG            # rad/s

        # Obstacle avoidance (raycast repulsion)
        self.n_rays = 8
        self.ray_max = 3.0
        self.avoid_strength = 1.2  # higher => more aggressive sidestep

        # Build scene and drone
        self._build_sim()

    # ---------------------------
    # World + drone construction
    # ---------------------------
    def _build_sim(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Simple obstacles (spheres)
        self._spawn_obstacles()

        # Try to load a realistic quad URDF; fallback to a procedural body if not found
        self.drone = self._load_quadrotor()

        # Rotor offsets (X configuration)
        self.rotor_offsets = np.array([
            [ self.arm,  0.0,   0.02],   # front
            [ 0.0,       self.arm, 0.02],   # left
            [-self.arm,  0.0,   0.02],   # back
            [ 0.0,      -self.arm, 0.02],   # right
        ], dtype=np.float32)

    def _spawn_obstacles(self):
        self.obstacles = []
        # Place a few spheres in front of start
        positions = [
            [1.2,  0.0, 0.6],
            [1.8,  0.5, 1.0],
            [2.1, -0.4, 1.1],
        ]
        for pos in positions:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.22, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.22, rgbaColor=[0.9, 0.3, 0.3, 1], physicsClientId=self.client)
            obs_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis,
                                       basePosition=pos,
                                       physicsClientId=self.client)
            self.obstacles.append(obs_id)

    def _load_quadrotor(self):
        """Load realistic quad URDF if available; else build a visible fallback."""
        # Preferred: Crazyflie-like URDF at assets/cf2x/cf2x.urdf
        cand = os.path.join("assets", "cf2x", "cf2x.urdf")
        start_pos = [0, 0, 0.6]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        if os.path.exists(cand):
            return p.loadURDF(cand, start_pos, start_ori, useFixedBase=False, flags=p.URDF_MAINTAIN_LINK_ORDER, physicsClientId=self.client)

        # Fallback: visible X-quad made from links
        col_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], physicsClientId=self.client)
        vis_body = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], rgbaColor=[0.2, 0.3, 0.8, 1], physicsClientId=self.client)
        drone = p.createMultiBody(baseMass=self.mass,
                                  baseCollisionShapeIndex=col_body,
                                  baseVisualShapeIndex=vis_body,
                                  basePosition=start_pos,
                                  baseOrientation=start_ori,
                                  physicsClientId=self.client)
        # cosmetic arms
        for axis in [(self.arm, 0, 0.02), (-self.arm, 0, 0.02), (0, self.arm, 0.02), (0, -self.arm, 0.02)]:
            vis_arm = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01, length=0.22, rgbaColor=[0.1, 0.1, 0.1, 1], physicsClientId=self.client)
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vis_arm,
                              basePosition=[axis[0], axis[1], axis[2]],
                              baseOrientation=p.getQuaternionFromEuler([0, math.pi/2 if axis[0]==0 else 0, 0]),
                              physicsClientId=self.client,
                              )
        return drone

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self):
        self.step_counter = 0
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 0.6], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.client)
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # Randomize target in configured bounds
        xb, yb, zb = self.target_bounds
        self.target_position = np.array([
            np.random.uniform(*xb),
            np.random.uniform(*yb),
            np.random.uniform(*zb),
        ], dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(orn)
        rel = self.target_position - np.array(pos)
        rays = self._ray_distances(self.n_rays, self.ray_max)
        obs = np.concatenate([np.array(pos), np.array(rpy), np.array(lin_vel), np.array(ang_vel), rel, rays]).astype(np.float32)
        return obs

    def _ray_distances(self, n_rays=8, max_dist=3.0):
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        pos = np.array(pos)
        distances = []
        for i in range(n_rays):
            angle = yaw + 2 * math.pi * (i / n_rays)
            dir_vec = np.array([math.cos(angle), math.sin(angle), 0.0])
            from_pos, to_pos = pos, pos + dir_vec * max_dist
            res = p.rayTest(from_pos.tolist(), to_pos.tolist(), physicsClientId=self.client)[0]
            hit_id, hit_frac = res[0], res[2]
            distances.append(hit_frac * max_dist if hit_id != -1 else max_dist)
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        self.step_counter += 1

        # ===== High-level command parsing =====
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        v_cmd = a[:3] * self.v_cmd_max  # desired body/world velocities (approx world frame)
        yaw_rate_cmd = float(a[3] * self.yaw_rate_cmd_max)

        # ===== Sense state =====
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        rpy = np.array(p.getEulerFromQuaternion(orn))
        pos = np.array(pos); lin_vel = np.array(lin_vel); ang_vel = np.array(ang_vel)

        # ===== Obstacle repulsion (ray-based) =====
        rays = self._ray_distances(self.n_rays, self.ray_max)
        repulse_xy = np.zeros(2, dtype=np.float32)
        if np.any(rays < self.ray_max):
            for i in range(self.n_rays):
                if rays[i] < self.ray_max:
                    rel = (self.ray_max - rays[i]) / self.ray_max
                    angle = rpy[2] + 2 * math.pi * (i / self.n_rays)
                    # push opposite to ray direction
                    repulse_xy -= self.avoid_strength * rel * np.array([math.cos(angle), math.sin(angle)])
        # Blend into commanded planar velocity
        v_cmd[:2] += repulse_xy
        v_cmd = np.clip(v_cmd, -self.v_cmd_max, self.v_cmd_max)

        # ===== Outer-loop (pos/vel) to desired accelerations =====
        rel = self.target_position - pos
        vel_err = v_cmd - lin_vel
        acc_cmd = self.kp_pos * rel + self.kp_vel * vel_err - self.kd_pos * lin_vel
        # feedforward gravity on z
        acc_cmd[2] += 0.0

        # Map desired accelerations to attitude setpoints (small-angle approx)
        # ax = g * (-theta), ay = g * phi  =>  phi ~ ay/g, theta ~ -ax/g
        phi_des   = np.clip(acc_cmd[1] / self.g, -25*DEG, 25*DEG)
        theta_des = np.clip(-acc_cmd[0] / self.g, -25*DEG, 25*DEG)
        yaw_rate_des = yaw_rate_cmd

        # Collective thrust to track z acceleration + gravity
        T_des = np.clip(self.mass * (self.g + acc_cmd[2]), 0.0, 4 * self.max_motor)

        # ===== Inner-loop (attitude rate) =====
        att_err = np.array([phi_des, theta_des, 0.0]) - rpy
        rate_des = self.kp_att * att_err  # desired p,q,r
        rate_err = rate_des - ang_vel
        tau = self.kd_att * rate_err
        # Add yaw control (r-axis)
        tau[2] += 0.5 * yaw_rate_des

        # ===== Mixer: [T, tau_x, tau_y, tau_z] -> motor thrusts f1..f4 =====
        # X-quad sign convention:
        #   tau_x ~ (f2 - f4)*L, tau_y ~ (f3 - f1)*L, tau_z ~ k*(f1 - f2 + f3 - f4)
        L = self.arm
        k = self.k_yaw
        # Solve linear system:
        # [ 1  1  1  1 ] [f1]   [T]
        # [-1  1  1 -1 ] [f2] = [tau_y / L]
        # [-1 -1  1  1 ] [f3]   [tau_x / L]
        # [ 1 -1  1 -1 ] [f4]   [tau_z / k]
        b1 = T_des
        b2 = tau[1] / L
        b3 = tau[0] / L
        b4 = tau[2] / k
        # Inverse of the above (pre-derived):
        f1 = 0.25 * ( b1 - b2 - b3 + b4 )
        f2 = 0.25 * ( b1 + b2 - b3 - b4 )
        f3 = 0.25 * ( b1 + b2 + b3 + b4 )
        f4 = 0.25 * ( b1 - b2 + b3 - b4 )
        motors = np.clip(np.array([f1, f2, f3, f4]), 0.0, self.max_motor)

        # ===== Apply forces at rotor positions (world frame) =====
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        for i in range(4):
            world_offset = R.dot(self.rotor_offsets[i])
            rotor_world = pos + world_offset
            force_world = [0.0, 0.0, float(motors[i])]
            p.applyExternalForce(self.drone, -1, force_world, rotor_world.tolist(), p.WORLD_FRAME, physicsClientId=self.client)

        # Light linear damping for numerical stability
        p.applyExternalForce(self.drone, -1, (-0.15 * lin_vel).tolist(), [0,0,0], p.LINK_FRAME, physicsClientId=self.client)

        # ===== Step sim =====
        p.stepSimulation(physicsClientId=self.client)
        if self.gui:
            time.sleep(self.dt)

        # ===== Compute obs/reward/done =====
        obs = self._get_obs()
        reward, done, info = self._reward_done(obs, motors)

        return obs, float(reward), bool(done), info

    # ---------------------------
    # Reward & termination
    # ---------------------------
    def _reward_done(self, obs, motors):
        pos = obs[0:3]
        rpy = obs[3:6]
        lin_vel = obs[6:9]
        rel = obs[12:15]
        dist = float(np.linalg.norm(rel))

        # Stability shaping
        tilt = abs(rpy[0]) + abs(rpy[1])
        speed = np.linalg.norm(lin_vel)

        # Penalties
        crash = pos[2] < 0.08 or abs(rpy[0]) > 70*DEG or abs(rpy[1]) > 70*DEG
        contact = len(p.getContactPoints(bodyA=self.drone, physicsClientId=self.client)) > 0
        control_cost = 0.0005 * float(np.sum(motors**2))

        reward = -1.2 * dist - 0.15 * tilt - 0.02 * speed - control_cost + 1.0  # alive bonus

        if dist < 0.25:
            reward += 100.0
            done = True
        elif crash or contact:
            reward -= 200.0
            done = True
        else:
            done = (self.step_counter >= self.max_steps)

        info = {"dist": dist, "tilt": float(tilt)}
        return reward, done, info

    # ---------------------------
    # Rendering helpers
    # ---------------------------
    def render(self, mode="human"):
        if self.client < 0:
            return
        # target marker
        p.addUserDebugLine(self.target_position.tolist(),
                           (self.target_position + np.array([0,0,0.25])).tolist(),
                           [0,1,0], lineWidth=4, lifeTime=0.1, physicsClientId=self.client)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass
