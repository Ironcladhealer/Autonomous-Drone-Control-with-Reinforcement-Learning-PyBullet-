# environment.py
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym.spaces import Box
import time
import math
import os

class QuadrotorEnv(gym.Env):
    """
    Simplified quadrotor environment in PyBullet.
    Action: 4 rotor thrusts in [0,1].
    Observation: position (3), orientation (3 Euler), linear vel (3), angular vel (3),
                 relative target vector (3), rays (8) => total 23
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 gui=False,
                 target_position=None,
                 max_steps=1000,
                 dt=1./240.):
        super().__init__()
        self.gui = gui
        self.dt = dt
        self.max_steps = max_steps
        self.step_counter = 0
        self._connect()
        # action: 4 rotors
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # observations
        # pos(3), euler(3), lin vel(3), ang vel(3), rel target(3), rays(8)
        obs_dim = 3 + 3 + 3 + 3 + 3 + 8
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)

        self.target_position = np.array([3.0, 0.0, 1.5]) if target_position is None else np.array(target_position)
        self._build_sim()

    def _connect(self):
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

    def _build_sim(self):
        # clear and load plane
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)

        # simple box as drone base
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.03], physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.03], rgbaColor=[0.2, 0.2, 0.8, 1], physicsClientId=self.client)
        mass = 1.2
        start_pos = [0, 0, 1.0]
        start_ori = p.getQuaternionFromEuler([0,0,0])
        self.drone = p.createMultiBody(baseMass=mass,
                                       baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis,
                                       basePosition=start_pos,
                                       baseOrientation=start_ori,
                                       physicsClientId=self.client)

        # rotor relative positions (front-right, front-left, back-left, back-right)
        self.rotor_offsets = np.array([
            [ 0.10, -0.10, 0.0],
            [ 0.10,  0.10, 0.0],
            [-0.10,  0.10, 0.0],
            [-0.10, -0.10, 0.0],
        ])

        # simple spherical obstacles for avoidance testing (optional)
        self._spawn_obstacles()

    def _spawn_obstacles(self):
        # put a few spheres between start and target
        self.obstacles = []
        positions = [
            [1.0, 0.0, 0.7],
            [1.7,  0.5, 1.0],
            [2.0, -0.5, 1.2],
        ]
        for i, pos in enumerate(positions):
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0.8,0.2,0.2,1], physicsClientId=self.client)
            obs_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis,
                                       basePosition=pos,
                                       physicsClientId=self.client)
            self.obstacles.append(obs_id)

    def reset(self):
        self.step_counter = 0
        p.resetBasePositionAndOrientation(self.drone, [0,0,1.0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.client)
        p.resetBaseVelocity(self.drone, [0,0,0], [0,0,0], physicsClientId=self.client)
        jitter = np.random.uniform(-0.5, 0.5, size=(3,))
        jitter[2] = np.clip(jitter[2], -0.2, 0.2)
        self.target_position = np.array([3.0, 0.0, 1.5]) + jitter
        obs = self._get_obs()
        return obs   # ✅ no (obs, info), just obs

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        rel_target = np.array(self.target_position) - np.array(pos)
        rays = self._ray_distances()
        obs = np.concatenate([np.array(pos), np.array(euler), np.array(lin_vel), np.array(ang_vel), rel_target, rays])
        return obs.astype(np.float32)

    def _ray_distances(self, n_rays=8, max_dist=3.0):
        # cast rays in horizontal plane + some tilt
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        pos = np.array(pos)
        euler = np.array(p.getEulerFromQuaternion(orn))
        yaw = euler[2]
        distances = []
        for i in range(n_rays):
            angle = yaw + 2*math.pi*(i/n_rays)
            dir_vec = np.array([math.cos(angle), math.sin(angle), 0.0])
            from_pos = pos + np.array([0,0,0.0])
            to_pos = from_pos + dir_vec * max_dist
            res = p.rayTest(from_pos.tolist(), to_pos.tolist(), physicsClientId=self.client)[0]
            hit_fraction = res[2]
            if hit_fraction < 1.0:
                distances.append(hit_fraction * max_dist)
            else:
                distances.append(max_dist)
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        self.step_counter += 1
        action = np.array(action, dtype=np.float32)
        # map normalized rotor thrusts to forces (N)
        max_thrust = 13.0  # N per rotor (tuneable)
        thrusts = action * max_thrust

        # apply forces at rotor offsets in world frame
        pos, orn = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3,3)

        # compute world-space rotor positions & apply upward forces
        for i in range(4):
            local_offset = self.rotor_offsets[i]
            world_offset = rot_matrix.dot(local_offset)
            rotor_world_pos = np.array(pos) + world_offset
            force_world = [0, 0, float(thrusts[i])]
            p.applyExternalForce(self.drone, -1, force_world, rotor_world_pos.tolist(), p.WORLD_FRAME, physicsClientId=self.client)

        # small drag to stabilize
        lin_vel, ang_vel = p.getBaseVelocity(self.drone, physicsClientId=self.client)
        p.applyExternalForce(self.drone, -1, [-0.5*lin_vel[0], -0.5*lin_vel[1], -0.2*lin_vel[2]], [0,0,0], p.LINK_FRAME, physicsClientId=self.client)

        # step sim
        p.stepSimulation(physicsClientId=self.client)
        if self.gui:
            time.sleep(self.dt)  # real-time

        obs = self._get_obs()
        reward, done, info = self._compute_reward_done(obs, action)
        return obs, reward, done, info

    def _compute_reward_done(self, obs, action):
        pos = obs[0:3]
        rel = obs[12:15]
        dist = np.linalg.norm(rel)
        # reward: negative distance (closer -> better) + alive bonus - control cost
        reach_bonus = 0.0
        done = False
        # reached target
        if dist < 0.25:
            reach_bonus = 100.0
            done = True

        # crashed / flipped
        euler = obs[3:6]
        if pos[2] < 0.12 or abs(euler[0]) > math.radians(60) or abs(euler[1]) > math.radians(60):
            done = True
            crash_penalty = -200.0
        else:
            crash_penalty = 0.0

        # small penalty for collisions with obstacles (contact points exist)
        contacts = p.getContactPoints(bodyA=self.drone, physicsClientId=self.client)
        contact_penalty = -50.0 if len(contacts) > 0 else 0.0

        # distance reward
        dist_reward = -dist * 2.0
        control_penalty = -0.1 * np.sum(np.square(action))

        reward = dist_reward + reach_bonus + control_penalty + crash_penalty + contact_penalty + 1.0  # alive bonus

        self_info = {}
        self_info['distance'] = float(dist)
        self_info['contacts'] = len(contacts)

        # step limit
        if self.step_counter >= self.max_steps:
            done = True

        return reward, done, self_info

    def render(self, mode='human'):
        # PyBullet GUI already shows the sim. Optionally draw target marker.
        if self.client < 0:
            return
        # draw a sphere marker at target
        p.addUserDebugLine(self.target_position.tolist(),
                           (self.target_position + np.array([0,0,0.2])).tolist(),
                           [0,1,0],
                           lineWidth=4,
                           lifeTime=0.1,
                           physicsClientId=self.client)
        return

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass
