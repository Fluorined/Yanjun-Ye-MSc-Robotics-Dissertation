# rrt_env.py (Complete version fixing endpoint convergence issues)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# —— RRT Algorithm & Helper Functions —— #

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
    # Sample uniformly on p1→p2 line segment, check safety distance from all obstacles
    for t in np.linspace(0, 1, 10):
        pt = p1 + (p2 - p1) * t
        for (ox, oy, r) in obstacles:
            if np.linalg.norm(pt - np.array([ox, oy])) <= (r + buffer):
                return False
    return True

def rrt(start, goal, obstacles, map_size,
        max_iter=500, step_size=0.5, buffer=0.0, seed=None):
    """
    Return path from start to goal (N×2 ndarray via-points), or None.
    Collision detection uses the same buffer safety threshold.
    Added seed parameter to ensure reproducibility.
    """
    # Set random seed
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
                # Backtrack
                path = []
                node = goal_node
                while node:
                    path.append(node.point)
                    node = node.parent
                return np.array(path[::-1])
    return None

def generate_obstacles(n, radius_range, start, goal, map_size, buffer, seed=None):
    """
    Generate n non-overlapping circular obstacle list [(x,y,r),...],
    maintaining buffer safety distance from start/goal and each other.
    Added seed parameter to ensure reproducibility.
    """
    # Set random seed
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

# —— Environment Class —— #

class RRTTrackingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # Class-level seed setting to ensure consistency across all instances
    CLASS_SEED = 8  # Default seed

    def __init__(self,
                 map_size=(10.0,10.0),
                 dt=0.1,
                 max_steps=200,
                 n_obstacles=8,
                 obstacle_radius_range=(0.3,1.0),
                 start=(0.5,0.5),
                 goal=(9.5,9.5),
                 traj=None,
                 obstacles=None,  # Allow external obstacle input
                 collision_buffer=0.2,  # Maintain backward compatibility
                 rrt_buffer=0.35,  # Buffer used for RRT path generation (larger)
                 tracking_buffer=0.15,  # Buffer used for collision detection during RL tracking (smaller)
                 auto_generate_rrt=True,  # Auto-generate RRT path option
                 seed=None,  # Add seed parameter
                 randomize_scenario=False,  # Whether to randomize scenario
                 obstacle_variance=0.2):  # Variance range for obstacle position and size
        super().__init__()
        
        # Determine the seed to use
        self.seed = seed if seed is not None else RRTTrackingEnv.CLASS_SEED
        
        # Set random seed (ensure consistency of environment initialization)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.map_size = np.array(map_size, dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps
        self.collision_buffer = collision_buffer
        self.rrt_buffer = rrt_buffer
        self.tracking_buffer = tracking_buffer
        self.auto_generate_rrt = auto_generate_rrt
        
        # Randomized scenario parameters
        self.randomize_scenario = randomize_scenario
        self.obstacle_variance = obstacle_variance
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        
        # Global randomization settings
        self.map_buffer = 0.5  # Map boundary buffer
        
        # Generate base obstacles first
        if obstacles is not None:
            self.base_obstacles = obstacles
            self.obstacles = obstacles
        else:
            # First generate obstacles using default start/goal points
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
        
        # If using randomized scenario, generate random start and goal points
        if self.randomize_scenario:
            self.start = self._generate_random_start(self.obstacles)
            self.goal = self._generate_random_goal(self.obstacles)
            # Ensure sufficient distance between start and goal points
            max_attempts = 50
            attempts = 0
            while dist(self.start, self.goal) < 4.0 and attempts < max_attempts:
                self.goal = self._generate_random_goal(self.obstacles)
                attempts += 1
            
            # If start-goal distance is still too close, regenerate start point
            if dist(self.start, self.goal) < 4.0:
                self.start = self._generate_random_start(self.obstacles)
                attempts = 0
                while dist(self.start, self.goal) < 4.0 and attempts < max_attempts:
                    self.start = self._generate_random_start(self.obstacles)
                    attempts += 1
        else:
            self.start = np.array(start, dtype=np.float32)
            self.goal = np.array(goal, dtype=np.float32)

        # If using randomized scenario, randomize obstacles
        if self.randomize_scenario and obstacles is not None:
            self.obstacles = self._randomize_obstacles(self.base_obstacles)

        # Reference trajectory via-points
        if traj is not None:
            self.traj = traj
        elif auto_generate_rrt:
            # Automatically attempt to generate RRT path, regenerate scenario if failed
            if self.randomize_scenario:
                # Randomized scenario mode: if RRT fails, regenerate entire scenario
                self._generate_rrt_with_retry()
            else:
                # Fixed scenario mode: if RRT fails, use straight line path
                rrt_path = rrt(
                    self.start,
                    self.goal,
                    self.obstacles,
                    self.map_size,
                    buffer=self.rrt_buffer,
                    seed=self.seed  # Pass seed
                )
                if rrt_path is not None:
                    self.traj = rrt_path
                    print(f"[Seed:{self.seed}] Automatic RRT path generation successful, contains {len(rrt_path)} path points")
                else:
                    self.traj = np.array([self.start, self.goal])
                    print(f"[Seed:{self.seed}] Warning: RRT planning failed, using straight line path")
        else:
            # Default straight line path
            self.traj = np.array([self.start, self.goal])

        # State space: [x, y, vx, vy, dx_ref, dy_ref, dx_next, dy_next, progress, at_final_ref]
        high = np.array([
            self.map_size[0], self.map_size[1],   # x,y
            5.0, 5.0,                            # vx,vy
            self.map_size[0], self.map_size[1],   # to_ref
            self.map_size[0], self.map_size[1],   # to_next_ref
            1.0,                                 # path_progress
            1.0                                  # at_final_ref (newly added)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        
        # Rendering related
        self.fig = None
        self.ax = None
        self.episode_frames = []  # Store each frame for saving rendered images
        
        # Initialize state variables
        self.pos = None
        self.vel = None
        self.step_count = None
        self.traj_idx = None

    def reset(self, seed=None, options=None):
        # Reset environment state
        super().reset(seed=seed)
        
        # Set random seed (ensure consistency of each reset)
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
        
        self.pos       = self.start.copy()
        self.vel       = np.zeros(2, dtype=np.float32)
        self.step_count= 0
        self.traj_idx  = 0
        self.episode_frames = []  # Clear frames from previous episode
        
        # If randomized scenario is enabled, generate new scenario at each reset
        if self.randomize_scenario:
            self._generate_new_scenario()
        
        # Return initial observation and info dictionary
        return self._get_obs(), {"seed": self.seed}

    def _get_obs(self):
        # Current reference point
        ref = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
              else self.traj[-1]
        
        # Next reference point (if exists)
        if self.traj_idx < len(self.traj) - 1:
            next_ref = self.traj[self.traj_idx + 1]
        else:
            # If already at last reference point, use the actual goal
            next_ref = self.goal
        
        # Path progress (fixed)
        path_progress = min(1.0, self.traj_idx / max(1, len(self.traj)-1))
        
        # Flag for whether reached last reference point
        at_final_ref = float(self.traj_idx >= len(self.traj) - 1)
        
        return np.concatenate([
            self.pos,                    # Current position
            self.vel,                    # Current velocity
            ref - self.pos,              # Vector to current reference point
            next_ref - self.pos,         # Vector to next reference point/goal
            [path_progress],             # Path progress percentage
            [at_final_ref]               # Whether reached last reference point
        ]).astype(np.float32)

    def step(self, action):
        # Apply control action
        accel = np.clip(action, -1, 1)
        self.vel += accel * self.dt
        self.vel = np.clip(self.vel, -5.0, 5.0)  # Speed limit
        self.pos += self.vel * self.dt
        
        # Boundary check
        self.pos[0] = np.clip(self.pos[0], 0, self.map_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.map_size[1])
        
        self.step_count += 1

        # Get current reference point
        target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
                 else self.traj[-1]
        dist_to_ref = dist(self.pos, target)
        
        # ========== Improved reference point switching mechanism ==========
        progress_threshold = 0.5  # Position condition threshold
        angle_threshold = np.pi/4  # Angle threshold (45 degrees)
        
        # If there is a next reference point
        if self.traj_idx < len(self.traj) - 1:
            next_target = self.traj[self.traj_idx + 1]
            
            # Calculate vectors to current reference point and next reference point
            to_ref = target - self.pos
            to_next = next_target - self.pos
            
            # Calculate distances
            to_ref_dist = np.linalg.norm(to_ref)
            to_next_dist = np.linalg.norm(to_next)
            
            # Calculate angle difference
            if to_ref_dist > 1e-5 and to_next_dist > 1e-5:
                cos_angle = np.dot(to_ref, to_next) / (to_ref_dist * to_next_dist)
                cos_angle = np.clip(cos_angle, -1, 1)  # Ensure within valid range
                angle_diff = np.arccos(cos_angle)
            else:
                angle_diff = 0
            
            # Primary switching condition: position condition (must be satisfied)
            if to_ref_dist < progress_threshold:
                self.traj_idx += 1
                print(f"Position condition switching to reference point {self.traj_idx}/{len(self.traj)}")
            
            # Auxiliary switching condition: direction condition (only considered when close to reference point)
            elif (to_ref_dist < progress_threshold * 1.5 and  # Within 1.5x threshold range
                  to_next_dist < to_ref_dist and 
                  angle_diff < angle_threshold):
                self.traj_idx += 1
                print(f"Direction auxiliary switching to reference point {self.traj_idx}/{len(self.traj)}")
        # ========== End reference point switching mechanism ==========
        
        # Update reference point (if switching occurred)
        target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) \
                 else self.traj[-1]
        dist_to_ref = dist(self.pos, target)
        
        # ========== Fixed reward function ==========
        # 1. Basic distance penalty (exponential decay)
        dist_penalty = np.exp(-0.5 * dist_to_ref) - 1
        
        # 2. Fixed path progress reward
        progress_reward = 0
        if self.traj_idx > 0:
            # Basic progress reward
            path_progress = self.traj_idx / len(self.traj)
            progress_reward = 10 * path_progress
            
            # Additional reward: if reached last reference point, give extra reward for reaching actual goal
            if self.traj_idx >= len(self.traj) - 1:
                # Calculate distance reward to actual goal
                dist_to_goal = dist(self.pos, self.goal)
                final_goal_reward = 20 * np.exp(-2.0 * dist_to_goal)  # Stronger goal attraction
                progress_reward += final_goal_reward
        
        # 3. Improved direction consistency reward
        # If haven't reached last reference point, head towards next reference point
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
            # If already reached last reference point, head towards current target (which is the last reference point)
            target_dir = target - self.pos
            if np.linalg.norm(target_dir) > 1e-5:
                target_dir /= np.linalg.norm(target_dir)
                
                if np.linalg.norm(self.vel) > 0.1:
                    vel_dir = self.vel / np.linalg.norm(self.vel)
                    alignment = np.dot(target_dir, vel_dir)
                    direction_reward = 5 * max(0, alignment)  # Enhanced last reference point direction reward
                else:
                    direction_reward = 0
            else:
                direction_reward = 0
        
        # 4. Add goal proximity reward
        dist_to_goal = dist(self.pos, self.goal)
        goal_proximity_reward = 0
        if self.traj_idx >= len(self.traj) - 1:  # Only give after reaching last reference point
            goal_proximity_reward = 15 * np.exp(-3.0 * dist_to_goal)
        
        # Total reward
        reward = dist_penalty + progress_reward + direction_reward + goal_proximity_reward

        terminated = False
        truncated = False
        collision = False
        
        # Check collision
        for (ox, oy, r) in self.obstacles:
            if dist(self.pos, np.array([ox, oy])) <= (r + self.tracking_buffer):
                reward -= 100
                terminated = True
                collision = True
                break

        # Reached final goal - using stricter judgment condition
        if dist(self.pos, self.goal) < 0.4 and not collision:
            reward += 1000
            terminated = True
            print(f"Successfully reached goal! Final distance: {dist(self.pos, self.goal):.3f}")
        
        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True
            print(f"Ended due to timeout, current reference point index: {self.traj_idx}, distance to goal: {dist(self.pos, self.goal):.3f}")

        # Return observation, reward, termination status, truncation status and info dictionary
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
            
            # Draw start and goal points
            self.start_marker = self.ax.scatter(
                self.start[0], self.start[1], c='green', marker='s', s=100, label='Start')
            self.goal_marker = self.ax.scatter(
                self.goal[0], self.goal[1], c='red', marker='*', s=200, label='Goal')
            
            # Draw reference trajectory
            self.ref_line, = self.ax.plot(
                self.traj[:, 0], self.traj[:, 1], 'g--', linewidth=2, label='Reference')
            
            # Draw obstacles
            self.obstacle_patches = []
            for (ox, oy, r) in self.obstacles:
                circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
                buffer = plt.Circle((ox, oy), r+self.tracking_buffer, 
                                   color='red', alpha=0.2, fill=False, linestyle='--')
                self.ax.add_patch(circle)
                self.ax.add_patch(buffer)
                self.obstacle_patches.append(circle)
                self.obstacle_patches.append(buffer)
            
            # Draw agent
            self.agent_marker, = self.ax.plot(
                self.pos[0], self.pos[1], 'bo', markersize=8, label='Agent')
            
            # Draw current reference point
            target = self.traj[self.traj_idx] if self.traj_idx < len(self.traj) else self.traj[-1]
            self.target_marker = self.ax.scatter(
                target[0], target[1], c='orange', marker='o', s=80, label='Current Target')
            
            self.ax.legend(loc='upper right')
            plt.pause(0.01)
        else:
            # Update agent position
            self.agent_marker.set_data(self.pos[0], self.pos[1])
            
            # Update current target point
            if self.traj_idx < len(self.traj):
                target = self.traj[self.traj_idx]
                self.target_marker.set_offsets([target])
            
            plt.pause(0.01)
            self.fig.canvas.draw()
        
        # Save current frame
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
        """Generate random start point, ensuring no collision with obstacles"""
        max_attempts = 500  # Increase maximum attempts
        
        for _ in range(max_attempts):
            # Randomly generate start point across entire map
            x = np.random.uniform(self.map_buffer, self.map_size[0] - self.map_buffer)
            y = np.random.uniform(self.map_buffer, self.map_size[1] - self.map_buffer)
            point = np.array([x, y], dtype=np.float32)
            
            # Check if collision with obstacles
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
        
        # If multiple attempts fail, return map center point
        center_x = self.map_size[0] / 2
        center_y = self.map_size[1] / 2
        print(f"Warning: Start point generation failed, using map center point ({center_x:.2f}, {center_y:.2f})")
        return np.array([center_x, center_y], dtype=np.float32)
    
    def _generate_random_goal(self, obstacles=None):
        """Generate random goal point, ensuring no collision with obstacles"""
        max_attempts = 500  # Increase maximum attempts
        
        for _ in range(max_attempts):
            # Randomly generate goal point across entire map
            x = np.random.uniform(self.map_buffer, self.map_size[0] - self.map_buffer)
            y = np.random.uniform(self.map_buffer, self.map_size[1] - self.map_buffer)
            point = np.array([x, y], dtype=np.float32)
            
            # Check if collision with obstacles
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
        
        # If multiple attempts fail, return map center point
        center_x = self.map_size[0] / 2
        center_y = self.map_size[1] / 2
        print(f"Warning: Goal point generation failed, using map center point ({center_x:.2f}, {center_y:.2f})")
        return np.array([center_x, center_y], dtype=np.float32)
    
    def _randomize_obstacles(self, base_obstacles):
        """Randomize the base obstacles"""
        randomized_obstacles = []
        for ox, oy, r in base_obstacles:
            # Randomize position
            dx = np.random.uniform(-self.obstacle_variance, self.obstacle_variance)
            dy = np.random.uniform(-self.obstacle_variance, self.obstacle_variance)
            new_x = np.clip(ox + dx, r + self.rrt_buffer, 
                          self.map_size[0] - r - self.rrt_buffer)
            new_y = np.clip(oy + dy, r + self.rrt_buffer, 
                          self.map_size[1] - r - self.rrt_buffer)
            
            # Randomize size (within 80%-120% of original size)
            size_factor = np.random.uniform(0.8, 1.2)
            new_r = np.clip(r * size_factor, 
                          self.obstacle_radius_range[0], 
                          self.obstacle_radius_range[1])
            
            randomized_obstacles.append((new_x, new_y, new_r))
        
        return randomized_obstacles
    
    def _generate_new_scenario(self):
        """Generate new randomized scenario, including regeneration mechanism after RRT planning failure"""
        max_scenario_attempts = 10  # Maximum scenario generation attempts
        
        for attempt in range(max_scenario_attempts):
            # Regenerate obstacles or randomize existing obstacles
            if hasattr(self, 'base_obstacles') and self.base_obstacles is not None:
                self.obstacles = self._randomize_obstacles(self.base_obstacles)
            else:
                self.obstacles = generate_obstacles(
                    self.n_obstacles,
                    self.obstacle_radius_range,
                    np.array([1.0, 1.0]),  # Temporary start point, will be regenerated later
                    np.array([9.0, 9.0]),  # Temporary goal point, will be regenerated later
                    self.map_size,
                    self.rrt_buffer,
                    seed=None  # Use current random state
                )
            
            # Generate new start and goal points, ensuring no collision with obstacles
            self.start = self._generate_random_start(self.obstacles)
            self.goal = self._generate_random_goal(self.obstacles)
            
            # Ensure sufficient distance between start and goal points
            max_distance_attempts = 50
            distance_attempt = 0
            while dist(self.start, self.goal) < 4.0 and distance_attempt < max_distance_attempts:
                if distance_attempt % 2 == 0:
                    self.goal = self._generate_random_goal(self.obstacles)
                else:
                    self.start = self._generate_random_start(self.obstacles)
                distance_attempt += 1
            
            if dist(self.start, self.goal) < 4.0:
                print(f"Attempt {attempt + 1}: Start-goal distance too close ({dist(self.start, self.goal):.2f} < 4.0), regenerating scenario...")
                continue
            
            # Regenerate RRT path
            if self.auto_generate_rrt:
                rrt_path = rrt(
                    self.start,
                    self.goal,
                    self.obstacles,
                    self.map_size,
                    buffer=self.rrt_buffer,
                    seed=None  # Use current random state
                )
                if rrt_path is not None:
                    self.traj = rrt_path
                    print(f"Scenario generation successful (attempt {attempt + 1}), RRT path contains {len(rrt_path)} points")
                    print(f"Start point: ({self.start[0]:.2f}, {self.start[1]:.2f})")
                    print(f"Goal point: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
                    print(f"Start-goal distance: {dist(self.start, self.goal):.2f}")
                    return  # Successfully generated, exit loop
                else:
                    print(f"Attempt {attempt + 1}: RRT planning failed, regenerating scenario...")
                    continue
            else:
                self.traj = np.array([self.start, self.goal])
                print(f"Scenario generation successful (attempt {attempt + 1}), using straight line path")
                return  # Successfully generated, exit loop
        
        # If all attempts fail, use default configuration
        print(f"Warning: Still failed after {max_scenario_attempts} attempts, using default configuration")
        self.start = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([9.0, 9.0], dtype=np.float32)
        self.obstacles = generate_obstacles(
            max(1, self.n_obstacles // 2),  # Reduce number of obstacles
            self.obstacle_radius_range,
            self.start,
            self.goal,
            self.map_size,
            self.rrt_buffer,
            seed=None
        )
        self.traj = np.array([self.start, self.goal])
    
    def _generate_rrt_with_retry(self):
        """Generate RRT path during initialization, regenerate scenario if failed"""
        max_attempts = 5  # Maximum retry attempts
        
        for attempt in range(max_attempts):
            rrt_path = rrt(
                self.start,
                self.goal,
                self.obstacles,
                self.map_size,
                buffer=self.rrt_buffer,
                seed=None  # Use current random state
            )
            
            if rrt_path is not None:
                self.traj = rrt_path
                print(f"[Seed:{self.seed}] Initialization RRT path generation successful (attempt {attempt + 1}), contains {len(rrt_path)} path points")
                return
            else:
                print(f"[Seed:{self.seed}] Initialization RRT planning failed (attempt {attempt + 1}), regenerating scenario...")
                # Regenerate scenario
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
                
                # Regenerate start and goal points
                self.start = self._generate_random_start(self.obstacles)
                self.goal = self._generate_random_goal(self.obstacles)
                
                # Ensure sufficient distance between start and goal points
                max_distance_attempts = 30
                distance_attempt = 0
                while dist(self.start, self.goal) < 4.0 and distance_attempt < max_distance_attempts:
                    if distance_attempt % 2 == 0:
                        self.goal = self._generate_random_goal(self.obstacles)
                    else:
                        self.start = self._generate_random_start(self.obstacles)
                    distance_attempt += 1
        
        # If all attempts fail, use straight line path
        print(f"[Seed:{self.seed}] Warning: RRT planning still failed after {max_attempts} attempts, using straight line path")
        self.traj = np.array([self.start, self.goal])
    
    @classmethod
    def set_class_seed(cls, seed):
        """Set default seed for all environments"""
        cls.CLASS_SEED = seed
        np.random.seed(seed)
        random.seed(seed)