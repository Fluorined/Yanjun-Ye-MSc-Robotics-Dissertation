# pid_env.py - 基于PID控制的轨迹跟踪环境

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from pid_controller import PIDController, PIDPathTracker

# 复用原有的RRT算法和辅助函数
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
    for t in np.linspace(0, 1, 10):
        pt = p1 + (p2 - p1) * t
        for (ox, oy, r) in obstacles:
            if np.linalg.norm(pt - np.array([ox, oy])) <= (r + buffer):
                return False
    return True

def rrt(start, goal, obstacles, map_size,
        max_iter=500, step_size=0.5, buffer=0.0, seed=None):
    """RRT路径规划算法"""
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
    """生成障碍物"""
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

class PIDTrackingEnv:
    """基于PID控制的轨迹跟踪环境"""
    
    def __init__(self,
                 map_size=(10.0, 10.0),
                 dt=0.1,
                 max_steps=200,
                 n_obstacles=8,
                 obstacle_radius_range=(0.3, 1.0),
                 start=(0.5, 0.5),
                 goal=(9.5, 9.5),
                 traj=None,
                 obstacles=None,
                 rrt_buffer=0.35,
                 tracking_buffer=0.15,
                 auto_generate_rrt=True,
                 seed=None,
                 # PID参数
                 kp_x=2.0, ki_x=0.1, kd_x=1.0,
                 kp_y=2.0, ki_y=0.1, kd_y=1.0,
                 lookahead_distance=0.5):
        
        self.map_size = np.array(map_size, dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps
        self.rrt_buffer = rrt_buffer
        self.tracking_buffer = tracking_buffer
        self.auto_generate_rrt = auto_generate_rrt
        self.seed = seed if seed is not None else 8
        
        # 设置随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # 生成障碍物或使用提供的障碍物
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = generate_obstacles(
                n_obstacles,
                obstacle_radius_range,
                np.array(start),
                np.array(goal),
                self.map_size,
                self.rrt_buffer,
                seed=self.seed
            )
        
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        
        # 生成或使用提供的参考轨迹
        if traj is not None:
            self.traj = traj
        elif auto_generate_rrt:
            rrt_path = rrt(
                self.start,
                self.goal,
                self.obstacles,
                self.map_size,
                buffer=self.rrt_buffer,
                seed=self.seed
            )
            if rrt_path is not None:
                self.traj = rrt_path
                print(f"[Seed:{self.seed}] 自动生成RRT路径成功，包含{len(rrt_path)}个路径点")
            else:
                self.traj = np.array([self.start, self.goal])
                print(f"[Seed:{self.seed}] 警告：RRT规划失败，使用直线路径")
        else:
            self.traj = np.array([self.start, self.goal])
        
        # 初始化PID控制器
        self.pid_controller = PIDController(
            kp_x=kp_x, ki_x=ki_x, kd_x=kd_x,
            kp_y=kp_y, ki_y=ki_y, kd_y=kd_y,
            dt=dt, max_vel=5.0, max_accel=1.0
        )
        
        # 初始化路径跟踪器
        self.path_tracker = PIDPathTracker(
            self.traj, self.pid_controller, lookahead_distance
        )
        
        # 渲染相关
        self.fig = None
        self.ax = None
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置环境状态"""
        self.pos = self.start.copy()
        self.vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.total_reward = 0
        self.last_control_accel = np.zeros(2, dtype=np.float32)  # Initialize control acceleration tracking
        
        # 重置PID控制器和路径跟踪器
        self.path_tracker.reset()
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态信息"""
        target_pos = self.path_tracker.get_target_point(self.pos)
        dist_to_target = dist(self.pos, target_pos)
        dist_to_goal = dist(self.pos, self.goal)
        
        return {
            'position': self.pos.copy(),
            'velocity': self.vel.copy(), 
            'target': target_pos,
            'distance_to_target': dist_to_target,
            'distance_to_goal': dist_to_goal,
            'waypoint_index': self.path_tracker.current_waypoint_idx,
            'total_waypoints': len(self.traj),
            'step_count': self.step_count
        }
    
    def step(self):
        """执行一步仿真"""
        # 使用PID控制器计算控制输出
        control_output, target_pos = self.path_tracker.control_step(self.pos, self.vel)
        
        # 应用控制输出（加速度）- 已在PID控制器中限制
        accel = control_output
        self.last_control_accel = accel.copy()  # Store the actual control acceleration
        self.vel += accel * self.dt
        self.vel = np.clip(self.vel, -5.0, 5.0)  # 速度限制
        self.pos += self.vel * self.dt
        
        # 边界检查
        self.pos[0] = np.clip(self.pos[0], 0, self.map_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.map_size[1])
        
        self.step_count += 1
        
        # 计算奖励和检查终止条件
        reward = self._compute_reward(target_pos)
        self.total_reward += reward
        
        terminated, collision = self._check_termination()
        
        return (self._get_state(), reward, terminated, collision)
    
    def _compute_reward(self, target_pos):
        """计算奖励函数"""
        # 距离奖励
        dist_to_target = dist(self.pos, target_pos)
        dist_to_goal = dist(self.pos, self.goal)
        
        # 基础距离惩罚
        dist_penalty = -0.1 * dist_to_target
        
        # 进度奖励
        progress = self.path_tracker.current_waypoint_idx / max(1, len(self.traj) - 1)
        progress_reward = 10 * progress
        
        # 终点接近奖励
        goal_proximity_reward = 0
        if self.path_tracker.current_waypoint_idx >= len(self.traj) - 1:
            goal_proximity_reward = 15 * np.exp(-3.0 * dist_to_goal)
        
        # 方向一致性奖励
        direction_reward = 0
        target_dir = target_pos - self.pos
        if np.linalg.norm(target_dir) > 1e-5:
            target_dir /= np.linalg.norm(target_dir)
            if np.linalg.norm(self.vel) > 0.1:
                vel_dir = self.vel / np.linalg.norm(self.vel)
                alignment = np.dot(target_dir, vel_dir)
                direction_reward = 2 * max(0, alignment)
        
        return dist_penalty + progress_reward + goal_proximity_reward + direction_reward
    
    def _check_termination(self):
        """检查终止条件"""
        # 检查碰撞
        for (ox, oy, r) in self.obstacles:
            if dist(self.pos, np.array([ox, oy])) <= (r + self.tracking_buffer):
                return True, True  # terminated, collision
        
        # 检查是否到达终点
        if dist(self.pos, self.goal) < 0.4:
            return True, False  # terminated, no collision
        
        # 检查超时
        if self.step_count >= self.max_steps:
            return True, False  # terminated, no collision
        
        return False, False  # not terminated, no collision
    
    def run_simulation(self, max_time=None):
        """运行完整仿真"""
        if max_time is None:
            max_time = self.max_steps
        
        trajectory = []
        rewards = []
        targets = []
        accelerations = []  # Track actual control accelerations
        
        state = self.reset()
        trajectory.append(self.pos.copy())
        
        print(f"开始PID轨迹跟踪仿真 (种子: {self.seed})")
        print(f"起点: {self.start}, 终点: {self.goal}")
        print(f"参考路径包含 {len(self.traj)} 个点")
        
        while self.step_count < max_time:
            state, reward, terminated, collision = self.step()
            
            trajectory.append(self.pos.copy())
            rewards.append(reward)
            targets.append(state['target'].copy())
            accelerations.append(self.last_control_accel.copy())  # Store actual control acceleration
            
            if terminated:
                if collision:
                    print(f"仿真结束：发生碰撞 (步数: {self.step_count})")
                elif dist(self.pos, self.goal) < 0.3:
                    print(f"仿真结束：成功到达终点 (步数: {self.step_count})")
                else:
                    print(f"仿真结束：超时 (步数: {self.step_count})")
                break
        
        print(f"最终位置: {self.pos}")
        print(f"到终点距离: {dist(self.pos, self.goal):.3f}")
        print(f"完成路径点: {self.path_tracker.current_waypoint_idx}/{len(self.traj)-1}")
        print(f"总奖励: {self.total_reward:.2f}")
        
        return {
            'trajectory': np.array(trajectory),
            'rewards': rewards,
            'targets': targets,
            'accelerations': np.array(accelerations),  # Include actual control accelerations
            'final_distance': dist(self.pos, self.goal),
            'success': dist(self.pos, self.goal) < 0.3 and not collision,
            'collision': collision,
            'total_reward': self.total_reward,
            'steps': self.step_count
        }
    
    def render(self, trajectory=None, save_path=None):
        """可视化环境和轨迹"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # 绘制起点和终点
        plt.scatter(self.start[0], self.start[1], c='green', marker='s', s=100, label='Start')
        plt.scatter(self.goal[0], self.goal[1], c='red', marker='*', s=200, label='Goal')
        
        # 绘制障碍物
        for (ox, oy, r) in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
            ax.add_patch(circle)
            buffer = plt.Circle((ox, oy), r+self.tracking_buffer, 
                               color='red', alpha=0.2, fill=False, linestyle='--')
            ax.add_patch(buffer)
        
        # 绘制参考轨迹
        plt.plot(self.traj[:, 0], self.traj[:, 1], 'g--', linewidth=2, label='Reference Path')
        
        # 标记所有参考点
        for i, point in enumerate(self.traj):
            plt.scatter(point[0], point[1], c='purple', s=60, marker='o', alpha=0.5)
            plt.text(point[0] + 0.1, point[1] + 0.1, f'WP{i}', 
                     fontsize=8, color='purple', alpha=0.7)
        
        # 绘制实际轨迹
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='PID Trajectory')
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=100, marker='o', label='Final Position')
        else:
            plt.scatter(self.pos[0], self.pos[1], c='blue', s=100, marker='o', label='Current Position')
        
        plt.title(f'PID Trajectory Tracking (Seed: {self.seed})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    def set_pid_gains(self, kp=None, ki=None, kd=None):
        """动态调整PID增益"""
        self.pid_controller.set_gains(kp, ki, kd)
        print(f"PID增益已更新: Kp={self.pid_controller.kp}, Ki={self.pid_controller.ki}, Kd={self.pid_controller.kd}")