# rrt_env.py (修复终点收敛问题的完整版本)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# —— RRT 算法 & 辅助函数 —— #

class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

def dist(a, b):
    return np.linalg.norm(a - b)

def nearest_node(nodes, rnd):
    dists = [dist(node.point, rnd) for node in nodes]
    return nodes[int(np.argmin(dists))]

def steer(from_p, to_p, step_size):
    direction = to_p - from_p
    length = np.linalg.norm(direction)
    if length <= step_size:
        return to_p
    return from_p + (direction / length) * step_size

def collision_free(p1, p2, obstacles, buffer):
    # 在 p1→p2 线段上均匀采样，检查与所有障碍的安全距离
    for t in np.linspace(0, 1, 10):
        pt = p1 + (p2 - p1) * t
        for (ox, oy, r) in obstacles:
            if np.linalg.norm(pt - np.array([ox, oy])) <= (r + buffer):
                return False
    return True

def rrt(start, goal, obstacles, map_size,
        max_iter=500, step_size=0.5, buffer=0.0, seed=None):
    """
    返回从 start 到 goal 的路径（N×2 ndarray via-points），或 None。
    碰撞检测使用同样的 buffer 安全阈值。
    添加 seed 参数确保可重现性。
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    nodes = [Node(start)]
    for _ in range(max_iter):
        rnd = goal if np.random.rand() < 0.05 else np.random.uniform([0,0], map_size)
        nearest = nearest_node(nodes, rnd)
        new_pt = steer(nearest.point, rnd, step_size)
        if collision_free(nearest.point, new_pt, obstacles, buffer):
            new_node = Node(new_pt, nearest)
            nodes.append(new_node)
            if dist(new_pt, goal) < step_size:
                goal_node = Node(goal, new_node)
                nodes.append(goal_node)
                # 回溯
                path = []
                node = goal_node
                while node:
                    path.append(node.point)
                    node = node.parent
                return np.array(path[::-1])
    return None

def generate_obstacles(n, radius_range, start, goal, map_size, buffer, seed=None):
    """
    生成 n 个不重叠圆形障碍列表 [(x,y,r),…]，
    与起终点及彼此间都保持 buffer 安全距离。
    添加 seed 参数确保可重现性。
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    obstacles = []
    min_r, max_r = radius_range
    attempts = 0
    while len(obstacles) < n and attempts < n * 50:
        attempts += 1
        r = np.random.uniform(min_r, max_r)
        x = np.random.uniform(r + buffer, map_size[0] - r - buffer)
        y = np.random.uniform(r + buffer, map_size[1] - r - buffer)
        center = np.array([x, y])
        if (np.linalg.norm(center - start) <= r + buffer or
            np.linalg.norm(center - goal)  <= r + buffer):
            continue
        overlap = False
        for ox, oy, orad in obstacles:
            if np.linalg.norm(center - np.array([ox, oy])) <= (r + orad + buffer):
                overlap = True
                break
        if not overlap:
            obstacles.append((x, y, r))
    return obstacles

# —— PID控制器类 —— #

class PIDController:
    def __init__(self, dt):
        self.dt = dt
        self.reset()
    
    def reset(self):
        """重置PID内部状态"""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.error_x = 0.0
        self.error_y = 0.0
    
    def compute(self, target_pos, current_pos, pid_params):
        """计算PID控制输出
        Args:
            target_pos: [x, y] 目标位置
            current_pos: [x, y] 当前位置
            pid_params: [Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y] PID参数
        Returns:
            [ax, ay] 加速度输出
        """
        # 解析PID参数
        Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y = pid_params
        
        # 计算误差
        self.error_x = target_pos[0] - current_pos[0]
        self.error_y = target_pos[1] - current_pos[1]
        
        # 积分项更新
        self.integral_x += self.error_x * self.dt
        self.integral_y += self.error_y * self.dt
        
        # 积分饱和处理
        max_integral = 10.0
        self.integral_x = np.clip(self.integral_x, -max_integral, max_integral)
        self.integral_y = np.clip(self.integral_y, -max_integral, max_integral)
        
        # 微分项计算
        derivative_x = (self.error_x - self.prev_error_x) / self.dt
        derivative_y = (self.error_y - self.prev_error_y) / self.dt
        
        # PID输出
        ax = Kp_x * self.error_x + Ki_x * self.integral_x + Kd_x * derivative_x
        ay = Kp_y * self.error_y + Ki_y * self.integral_y + Kd_y * derivative_y
        
        # 更新上一步误差
        self.prev_error_x = self.error_x
        self.prev_error_y = self.error_y
        
        return np.array([ax, ay], dtype=np.float32)
    
    def get_state(self):
        """获取PID内部状态用于观察值"""
        return np.array([
            self.error_x, self.error_y,
            self.integral_x, self.integral_y,
            self.prev_error_x, self.prev_error_y
        ], dtype=np.float32)

# —— 环境类 —— #

class RRTTrackingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # 类级种子设置，确保所有实例的一致性
    CLASS_SEED = 8  # 默认种子

    def __init__(self,
                 map_size=(10.0,10.0),
                 dt=0.1,
                 max_steps=200,
                 n_obstacles=8,
                 obstacle_radius_range=(0.3,1.0),
                 start=(0.5,0.5),
                 goal=(9.5,9.5),
                 traj=None,
                 obstacles=None,  # 允许外部传入障碍物
                 collision_buffer=0.2,  # 保持向后兼容
                 rrt_buffer=0.35,  # RRT路径生成使用的buffer（更大）
                 tracking_buffer=0.15,  # RL跟踪时碰撞检测使用的buffer（更小）
                 auto_generate_rrt=True,  # 自动生成RRT路径选项
                 seed=None,  # 添加种子参数
                 randomize_scenario=False,  # 是否随机化场景
                 obstacle_variance=0.2):  # 障碍物位置和大小的变化范围
        super().__init__()
        
        # 确定使用的种子
        self.seed = seed if seed is not None else RRTTrackingEnv.CLASS_SEED
        
        # 设置随机种子（确保环境初始化的一致性）
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.map_size = np.array(map_size, dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps
        self.collision_buffer = collision_buffer
        self.rrt_buffer = rrt_buffer
        self.tracking_buffer = tracking_buffer
        self.auto_generate_rrt = auto_generate_rrt
        
        # 随机化场景参数
        self.randomize_scenario = randomize_scenario
        self.obstacle_variance = obstacle_variance
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        
        # 全局随机化设置
        self.map_buffer = 0.5  # 地图边界缓冲区
        
        # 先生成基础障碍物
        if obstacles is not None:
            self.base_obstacles = obstacles
            self.obstacles = obstacles
        else:
            # 先用默认起终点生成障碍物
            temp_start = np.array(start, dtype=np.float32)
            temp_goal = np.array(goal, dtype=np.float32)
            self.base_obstacles = generate_obstacles(
                n_obstacles,
                obstacle_radius_range,
                temp_start,
                temp_goal,
                self.map_size,
                self.rrt_buffer,
                seed=self.seed
            )
            self.obstacles = self.base_obstacles
        
        # 如果使用随机场景，生成随机的起点和终点
        if self.randomize_scenario:
            self.start = self._generate_random_start(self.obstacles)
            self.goal = self._generate_random_goal(self.obstacles)
            # 确保起点和终点之间有足够的距离
            max_attempts = 50
            attempts = 0
            while dist(self.start, self.goal) < 4.0 and attempts < max_attempts:
                self.goal = self._generate_random_goal(self.obstacles)
                attempts += 1
            
            # 如果起终点距离还是太近，重新生成起点
            if dist(self.start, self.goal) < 4.0:
                self.start = self._generate_random_start(self.obstacles)
                attempts = 0
                while dist(self.start, self.goal) < 4.0 and attempts < max_attempts:
                    self.start = self._generate_random_start(self.obstacles)
                    attempts += 1
        else:
            self.start = np.array(start, dtype=np.float32)
            self.goal = np.array(goal, dtype=np.float32)

        # 如果使用随机场景，随机化障碍物
        if self.randomize_scenario and obstacles is not None:
            self.obstacles = self._randomize_obstacles(self.base_obstacles)

        # 参考轨迹 via-points
        if traj is not None:
            self.traj = traj
        elif auto_generate_rrt:
            # 自动尝试生成RRT路径，如果失败则重新生成场景
            if self.randomize_scenario:
                # 随机场景模式：如果RRT失败，重新生成整个场景
                self._generate_rrt_with_retry()
            else:
                # 固定场景模式：如果RRT失败，使用直线路径
                rrt_path = rrt(
                    self.start,
                    self.goal,
                    self.obstacles,
                    self.map_size,
                    buffer=self.rrt_buffer,
                    seed=self.seed  # 传递种子
                )
                if rrt_path is not None:
                    self.traj = rrt_path
                    print(f"[Seed:{self.seed}] 自动生成RRT路径成功，包含{len(rrt_path)}个路径点")
                else:
                    self.traj = np.array([self.start, self.goal])
                    print(f"[Seed:{self.seed}] 警告：RRT规划失败，使用直线路径")
        else:
            # 默认直线路径
            self.traj = np.array([self.start, self.goal])

        # 状态空间：[x, y, vx, vy, dx_ref, dy_ref, dx_next, dy_next, progress, at_final_ref, error_x, error_y, integral_x, integral_y, prev_error_x, prev_error_y]
        high = np.array([
            self.map_size[0], self.map_size[1],   # x,y
            5.0, 5.0,                            # vx,vy
            self.map_size[0], self.map_size[1],   # to_ref
            self.map_size[0], self.map_size[1],   # to_next_ref
            1.0,                                 # path_progress
            1.0,                                 # at_final_ref
            self.map_size[0], self.map_size[1],   # error_x, error_y
            10.0, 10.0,                          # integral_x, integral_y (积分项有限制)
            self.map_size[0], self.map_size[1]    # prev_error_x, prev_error_y
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 动作空间：[Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y] - 归一化到[-1,1]，后续映射到实际参数范围
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        
        # 渲染相关
        self.fig = None
        self.ax = None
        self.episode_frames = []  # 存储每一帧用于保存渲染图
        
        # 初始化状态变量
        self.pos = None
        self.vel = None
        self.step_count = None
        self.traj_idx = None
        
        # 初始化PID控制器
        self.pid_controller = PIDController(self.dt)
        
        # PID参数范围定义
        self.pid_param_ranges = {
            'Kp': [0.0, 5.0],    # 比例增益范围
            'Ki': [0.0, 1.0],    # 积分增益范围
            'Kd': [0.0, 3.0]     # 微分增益范围
        }

    def reset(self, seed=None, options=None):
        # 重置环境状态
        super().reset(seed=seed)
        
        # 设置随机种子（确保每次重置的一致性）
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
        
        self.pos       = self.start.copy()
        self.vel       = np.zeros(2, dtype=np.float32)
        self.step_count= 0
        self.traj_idx  = 0
        self.episode_frames = []  # 清空上一轮的帧
        
        # 重置PID控制器状态
        self.pid_controller.reset()
        
        # 如果启用随机场景，在每次重置时生成新的场景
        if self.randomize_scenario:
            self._generate_new_scenario()
        
        # 返回初始观察值和信息字典
        return self._get_obs(), {"seed": self.seed}
    
    def _map_action_to_pid_params(self, action):
        """将[-1,1]范围的动作映射到PID参数范围"""
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 将[-1,1]映射到[0,1]
        normalized = (action + 1.0) / 2.0
        
        # 映射到实际PID参数范围
        Kp_x = normalized[0] * (self.pid_param_ranges['Kp'][1] - self.pid_param_ranges['Kp'][0]) + self.pid_param_ranges['Kp'][0]
        Ki_x = normalized[1] * (self.pid_param_ranges['Ki'][1] - self.pid_param_ranges['Ki'][0]) + self.pid_param_ranges['Ki'][0]
        Kd_x = normalized[2] * (self.pid_param_ranges['Kd'][1] - self.pid_param_ranges['Kd'][0]) + self.pid_param_ranges['Kd'][0]
        Kp_y = normalized[3] * (self.pid_param_ranges['Kp'][1] - self.pid_param_ranges['Kp'][0]) + self.pid_param_ranges['Kp'][0]
        Ki_y = normalized[4] * (self.pid_param_ranges['Ki'][1] - self.pid_param_ranges['Ki'][0]) + self.pid_param_ranges['Ki'][0]
        Kd_y = normalized[5] * (self.pid_param_ranges['Kd'][1] - self.pid_param_ranges['Kd'][0]) + self.pid_param_ranges['Kd'][0]
        
        return np.array([Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y], dtype=np.float32)

    def _get_obs(self):
        # 当前参考点
        ref = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
              else self.traj[-1]
        
        # 下一个参考点（如果存在）
        if self.traj_idx < len(self.traj) - 1:
            next_ref = self.traj[self.traj_idx + 1]
        else:
            # 如枟已经是最后一个参考点，使用真正的终点
            next_ref = self.goal
        
        # 路径进度（修复）
        path_progress = min(1.0, self.traj_idx / max(1, len(self.traj)-1))
        
        # 是否到达最后参考点的标志
        at_final_ref = float(self.traj_idx >= len(self.traj) - 1)
        
        # 获取PID内部状态
        pid_state = self.pid_controller.get_state()
        
        return np.concatenate([
            self.pos,                    # 当前位置
            self.vel,                    # 当前速度
            ref - self.pos,              # 到当前参考点的向量
            next_ref - self.pos,         # 到下一个参考点/终点的向量
            [path_progress],             # 路径进度百分比
            [at_final_ref],              # 是否到达最后参考点
            pid_state                    # PID内部状态: [error_x, error_y, integral_x, integral_y, prev_error_x, prev_error_y]
        ]).astype(np.float32)

    def step(self, action):
        # 获取当前目标位置
        target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
                 else self.traj[-1]
        
        # 将RL动作映射为PID参数
        pid_params = self._map_action_to_pid_params(action)
        
        # 使用PID控制器计算加速度
        accel = self.pid_controller.compute(target, self.pos, pid_params)
        
        # 限制加速度幅度
        accel = np.clip(accel, -1.0, 1.0)
        
        # 更新速度和位置
        self.vel += accel * self.dt
        self.vel = np.clip(self.vel, -5.0, 5.0)  # 限速
        self.pos += self.vel * self.dt
        
        # 边界检查
        self.pos[0] = np.clip(self.pos[0], 0, self.map_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.map_size[1])
        
        self.step_count += 1

        # 获取当前参考点
        target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
                 else self.traj[-1]
        dist_to_ref = dist(self.pos, target)
        
        # ========== 改进的参考点切换机制 ==========
        progress_threshold = 0.5  # 位置条件阈值
        angle_threshold = np.pi/4  # 角度阈值 (45度)
        
        # 如果有下一个参考点
        if self.traj_idx < len(self.traj) - 1:
            next_target = self.traj[self.traj_idx + 1]
            
            # 计算到当前参考点和下一个参考点的向量
            to_ref = target - self.pos
            to_next = next_target - self.pos
            
            # 计算距离
            to_ref_dist = np.linalg.norm(to_ref)
            to_next_dist = np.linalg.norm(to_next)
            
            # 计算角度差
            if to_ref_dist > 1e-5 and to_next_dist > 1e-5:
                cos_angle = np.dot(to_ref, to_next) / (to_ref_dist * to_next_dist)
                cos_angle = np.clip(cos_angle, -1, 1)  # 确保在有效范围内
                angle_diff = np.arccos(cos_angle)
            else:
                angle_diff = 0
            
            # 主要切换条件：位置条件（必须满足）
            if to_ref_dist < progress_threshold:
                self.traj_idx += 1
                print(f"位置条件切换到参考点 {self.traj_idx}/{len(self.traj)}")
            
            # 辅助切换条件：方向条件（仅在接近参考点时考虑）
            elif (to_ref_dist < progress_threshold * 1.5 and  # 在1.5倍阈值范围内
                  to_next_dist < to_ref_dist and 
                  angle_diff < angle_threshold):
                self.traj_idx += 1
                print(f"方向辅助切换到参考点 {self.traj_idx}/{len(self.traj)}")
        # ========== 结束参考点切换机制 ==========
        
        # 更新参考点（如果切换发生）
        target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
                 else self.traj[-1]
        dist_to_ref = dist(self.pos, target)
        
        # ========== 修复的奖励函数 ==========
        # 1. 基础距离惩罚（指数衰减）
        dist_penalty = np.exp(-0.5 * dist_to_ref) - 1
        
        # 2. 修复的路径进度奖励
        progress_reward = 0
        if self.traj_idx > 0:
            # 基础进度奖励
            path_progress = self.traj_idx / len(self.traj)
            progress_reward = 10 * path_progress
            
            # 额外奖励：如果已到达最后一个参考点，给予到达真正终点的额外奖励
            if self.traj_idx >= len(self.traj) - 1:
                # 计算到真正终点的距离奖励
                dist_to_goal = dist(self.pos, self.goal)
                final_goal_reward = 20 * np.exp(-2.0 * dist_to_goal)  # 更强的终点吸引力
                progress_reward += final_goal_reward
        
        # 3. 改进的方向一致性奖励
        # 如果还没到达最后一个参考点，朝向下一个参考点
        if self.traj_idx < len(self.traj) - 1:
            target_dir = target - self.pos
            if np.linalg.norm(target_dir) > 1e-5:
                target_dir /= np.linalg.norm(target_dir)
                
                if np.linalg.norm(self.vel) > 0.1:
                    vel_dir = self.vel / np.linalg.norm(self.vel)
                    alignment = np.dot(target_dir, vel_dir)
                    direction_reward = 2 * max(0, alignment)
                else:
                    direction_reward = 0
            else:
                direction_reward = 0
        else:
            # 如果已到达最后一个参考点，朝向当前target（就是最后的参考点）
            target_dir = target - self.pos
            if np.linalg.norm(target_dir) > 1e-5:
                target_dir /= np.linalg.norm(target_dir)
                
                if np.linalg.norm(self.vel) > 0.1:
                    vel_dir = self.vel / np.linalg.norm(self.vel)
                    alignment = np.dot(target_dir, vel_dir)
                    direction_reward = 5 * max(0, alignment)  # 增强最后参考点方向奖励
                else:
                    direction_reward = 0
            else:
                direction_reward = 0
        
        # 4. 添加终点接近奖励
        dist_to_goal = dist(self.pos, self.goal)
        goal_proximity_reward = 0
        if self.traj_idx >= len(self.traj) - 1:  # 只在到达最后参考点后给予
            goal_proximity_reward = 15 * np.exp(-3.0 * dist_to_goal)
        
        # 总奖励
        reward = dist_penalty + progress_reward + direction_reward + goal_proximity_reward

        terminated = False
        truncated = False
        collision = False
        
        # 检查碰撞
        for (ox, oy, r) in self.obstacles:
            if dist(self.pos, np.array([ox, oy])) <= (r + self.tracking_buffer):
                reward -= 100
                terminated = True
                collision = True
                break

        # 到达最终目标 - 使用更严格的判断条件
        if dist(self.pos, self.goal) < 0.4 and not collision:
            reward += 1000
            terminated = True
            print(f"成功到达终点！最终距离: {dist(self.pos, self.goal):.3f}")
        
        # 超时
        if self.step_count >= self.max_steps:
            truncated = True
            print(f"超时结束，当前参考点索引: {self.traj_idx}, 到终点距离: {dist(self.pos, self.goal):.3f}")

        # 返回观察值、奖励、终止状态、截断状态和信息字典
        return self._get_obs(), reward, terminated, truncated, {"seed": self.seed}

    def render(self, mode='human'):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.map_size[0])
            self.ax.set_ylim(0, self.map_size[1])
            self.ax.set_aspect('equal')
            self.ax.set_title(f'RRT Tracking Environment (Seed: {self.seed})')
            self.ax.grid(True)
            
            # 绘制起点和终点
            self.start_marker = self.ax.scatter(
                self.start[0], self.start[1], c='green', marker='s', s=100, label='Start')
            self.goal_marker = self.ax.scatter(
                self.goal[0], self.goal[1], c='red', marker='*', s=200, label='Goal')
            
            # 绘制参考轨迹
            self.ref_line, = self.ax.plot(
                self.traj[:, 0], self.traj[:, 1], 'g--', linewidth=2, label='Reference')
            
            # 绘制障碍物
            self.obstacle_patches = []
            for (ox, oy, r) in self.obstacles:
                circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
                buffer = plt.Circle((ox, oy), r+self.tracking_buffer, 
                                   color='red', alpha=0.2, fill=False, linestyle='--')
                self.ax.add_patch(circle)
                self.ax.add_patch(buffer)
                self.obstacle_patches.append(circle)
                self.obstacle_patches.append(buffer)
            
            # 绘制智能体
            self.agent_marker, = self.ax.plot(
                self.pos[0], self.pos[1], 'bo', markersize=8, label='Agent')
            
            # 绘制当前参考点
            target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) else self.traj[-1]
            self.target_marker = self.ax.scatter(
                target[0], target[1], c='orange', marker='o', s=80, label='Current Target')
            
            self.ax.legend(loc='upper right')
            plt.pause(0.01)
        else:
            # 更新智能体位置
            self.agent_marker.set_data(self.pos[0], self.pos[1])
            
            # 更新当前目标点
            if self.traj_idx < len(self.traj):
                target = self.traj[self.traj_idx]
                self.target_marker.set_offsets([target])
            
            plt.pause(0.01)
            self.fig.canvas.draw()
        
        # 保存当前帧
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.episode_frames.append(image)
            return image
        else:
            return None

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def _generate_random_start(self, obstacles=None):
        """生成随机起点，确保不与障碍物碰撞"""
        max_attempts = 500  # 增加最大尝试次数
        
        for _ in range(max_attempts):
            # 在整个地图上随机生成起点
            x = np.random.uniform(self.map_buffer, self.map_size[0] - self.map_buffer)
            y = np.random.uniform(self.map_buffer, self.map_size[1] - self.map_buffer)
            point = np.array([x, y], dtype=np.float32)
            
            # 检查是否与障碍物碰撞
            if obstacles is not None:
                collision = False
                for ox, oy, r in obstacles:
                    if np.linalg.norm(point - np.array([ox, oy])) <= (r + self.rrt_buffer + 0.2):
                        collision = True
                        break
                if not collision:
                    return point
            else:
                return point
        
        # 如果多次尝试都失败，返回地图中心点
        center_x = self.map_size[0] / 2
        center_y = self.map_size[1] / 2
        print(f"警告：起点生成失败，使用地图中心点 ({center_x:.2f}, {center_y:.2f})")
        return np.array([center_x, center_y], dtype=np.float32)
    
    def _generate_random_goal(self, obstacles=None):
        """生成随机终点，确保不与障碍物碰撞"""
        max_attempts = 500  # 增加最大尝试次数
        
        for _ in range(max_attempts):
            # 在整个地图上随机生成终点
            x = np.random.uniform(self.map_buffer, self.map_size[0] - self.map_buffer)
            y = np.random.uniform(self.map_buffer, self.map_size[1] - self.map_buffer)
            point = np.array([x, y], dtype=np.float32)
            
            # 检查是否与障碍物碰撞
            if obstacles is not None:
                collision = False
                for ox, oy, r in obstacles:
                    if np.linalg.norm(point - np.array([ox, oy])) <= (r + self.rrt_buffer + 0.2):
                        collision = True
                        break
                if not collision:
                    return point
            else:
                return point
        
        # 如果多次尝试都失败，返回地图中心点
        center_x = self.map_size[0] / 2
        center_y = self.map_size[1] / 2
        print(f"警告：终点生成失败，使用地图中心点 ({center_x:.2f}, {center_y:.2f})")
        return np.array([center_x, center_y], dtype=np.float32)
    
    def _randomize_obstacles(self, base_obstacles):
        """对基础障碍物进行随机化"""
        randomized_obstacles = []
        for ox, oy, r in base_obstacles:
            # 随机化位置
            dx = np.random.uniform(-self.obstacle_variance, self.obstacle_variance)
            dy = np.random.uniform(-self.obstacle_variance, self.obstacle_variance)
            new_x = np.clip(ox + dx, r + self.rrt_buffer, 
                          self.map_size[0] - r - self.rrt_buffer)
            new_y = np.clip(oy + dy, r + self.rrt_buffer, 
                          self.map_size[1] - r - self.rrt_buffer)
            
            # 随机化大小（在原始大小的80%-120%范围内）
            size_factor = np.random.uniform(0.8, 1.2)
            new_r = np.clip(r * size_factor, 
                          self.obstacle_radius_range[0], 
                          self.obstacle_radius_range[1])
            
            randomized_obstacles.append((new_x, new_y, new_r))
        
        return randomized_obstacles
    
    def _generate_new_scenario(self):
        """生成新的随机场景，包含RRT规划失败后的重新生成机制"""
        max_scenario_attempts = 10  # 最大场景生成尝试次数
        
        for attempt in range(max_scenario_attempts):
            # 重新生成障碍物或随机化现有障碍物
            if hasattr(self, 'base_obstacles') and self.base_obstacles is not None:
                self.obstacles = self._randomize_obstacles(self.base_obstacles)
            else:
                self.obstacles = generate_obstacles(
                    self.n_obstacles,
                    self.obstacle_radius_range,
                    np.array([1.0, 1.0]),  # 临时起点，后面会重新生成
                    np.array([9.0, 9.0]),  # 临时终点，后面会重新生成
                    self.map_size,
                    self.rrt_buffer,
                    seed=None  # 使用当前随机状态
                )
            
            # 生成新的起点和终点，确保不与障碍物碰撞
            self.start = self._generate_random_start(self.obstacles)
            self.goal = self._generate_random_goal(self.obstacles)
            
            # 确保起点和终点之间有足够的距离
            max_distance_attempts = 50
            distance_attempt = 0
            while dist(self.start, self.goal) < 4.0 and distance_attempt < max_distance_attempts:
                if distance_attempt % 2 == 0:
                    self.goal = self._generate_random_goal(self.obstacles)
                else:
                    self.start = self._generate_random_start(self.obstacles)
                distance_attempt += 1
            
            if dist(self.start, self.goal) < 4.0:
                print(f"尝试 {attempt + 1}: 起终点距离过近 ({dist(self.start, self.goal):.2f} < 4.0)，重新生成场景...")
                continue
            
            # 重新生成RRT路径
            if self.auto_generate_rrt:
                rrt_path = rrt(
                    self.start,
                    self.goal,
                    self.obstacles,
                    self.map_size,
                    buffer=self.rrt_buffer,
                    seed=None  # 使用当前随机状态
                )
                if rrt_path is not None:
                    self.traj = rrt_path
                    print(f"场景生成成功（尝试 {attempt + 1}），RRT路径包含{len(rrt_path)}个点")
                    print(f"起点: ({self.start[0]:.2f}, {self.start[1]:.2f})")
                    print(f"终点: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
                    print(f"起终点距离: {dist(self.start, self.goal):.2f}")
                    return  # 成功生成，退出循环
                else:
                    print(f"尝试 {attempt + 1}: RRT规划失败，重新生成场景...")
                    continue
            else:
                self.traj = np.array([self.start, self.goal])
                print(f"场景生成成功（尝试 {attempt + 1}），使用直线路径")
                return  # 成功生成，退出循环
        
        # 如果所有尝试都失败，使用默认配置
        print(f"警告：{max_scenario_attempts} 次尝试后仍然失败，使用默认配置")
        self.start = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.obstacles = generate_obstacles(
            max(1, self.n_obstacles // 2),  # 减少障碍物数量
            self.obstacle_radius_range,
            self.start,
            self.goal,
            self.map_size,
            self.rrt_buffer,
            seed=None
        )
        self.traj = np.array([self.start, self.goal])
    
    def _generate_rrt_with_retry(self):
        """在初始化时生成RRT路径，如果失败则重新生成场景"""
        max_attempts = 5  # 最大重试次数
        
        for attempt in range(max_attempts):
            rrt_path = rrt(
                self.start,
                self.goal,
                self.obstacles,
                self.map_size,
                buffer=self.rrt_buffer,
                seed=None  # 使用当前随机状态
            )
            
            if rrt_path is not None:
                self.traj = rrt_path
                print(f"[Seed:{self.seed}] 初始化RRT路径成功（尝试 {attempt + 1}），包含{len(rrt_path)}个路径点")
                return
            else:
                print(f"[Seed:{self.seed}] 初始化RRT规划失败（尝试 {attempt + 1}），重新生成场景...")
                # 重新生成场景
                if hasattr(self, 'base_obstacles') and self.base_obstacles is not None:
                    self.obstacles = self._randomize_obstacles(self.base_obstacles)
                else:
                    self.obstacles = generate_obstacles(
                        self.n_obstacles,
                        self.obstacle_radius_range,
                        np.array([1.0, 1.0]),
                        np.array([9.0, 9.0]),
                        self.map_size,
                        self.collision_buffer,
                        seed=None
                    )
                
                # 重新生成起点和终点
                self.start = self._generate_random_start(self.obstacles)
                self.goal = self._generate_random_goal(self.obstacles)
                
                # 确保起点和终点之间有足够的距离
                max_distance_attempts = 30
                distance_attempt = 0
                while dist(self.start, self.goal) < 4.0 and distance_attempt < max_distance_attempts:
                    if distance_attempt % 2 == 0:
                        self.goal = self._generate_random_goal(self.obstacles)
                    else:
                        self.start = self._generate_random_start(self.obstacles)
                    distance_attempt += 1
        
        # 如果所有尝试都失败，使用直线路径
        print(f"[Seed:{self.seed}] 警告：{max_attempts} 次尝试后RRT规划仍然失败，使用直线路径")
        self.traj = np.array([self.start, self.goal])
    
    @classmethod
    def set_class_seed(cls, seed):
        """设置所有环境的默认种子"""
        cls.CLASS_SEED = seed
        np.random.seed(seed)
        random.seed(seed)