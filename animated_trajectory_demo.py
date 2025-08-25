#!/usr/bin/env python3
"""
Animated Trajectory Demonstration for Three Controllers
Creates animated GIF demonstrations showing robot trajectories in real-time
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import time

# Add controller paths
sys.path.append('./gym3_change_big_different_buffer')
sys.path.append('./pid_tracking/pid_implementation')
sys.path.append('./rl_pid')

# Import controller related modules
SAC_AVAILABLE = False
PID_AVAILABLE = False
RLPID_AVAILABLE = False

# Try importing SAC controller
try:
    from gym3_change_big_different_buffer.rrt_env import RRTTrackingEnv as SACEnv
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SAC_AVAILABLE = True
    print("✅ SAC controller import successful")
except ImportError as e:
    print(f"❌ SAC controller import failed: {e}")

# Try importing PID controller
try:
    from pid_tracking.pid_implementation.pid_env import PIDTrackingEnv
    PID_AVAILABLE = True
    print("✅ PID controller import successful")
except ImportError as e:
    print(f"❌ PID controller import failed: {e}")

# Try importing RL-PID controller
try:
    from rl_pid import RRTTrackingEnv as RLPIDTrackingEnv
    RLPID_AVAILABLE = True
    print("✅ RL-PID controller import successful")
except ImportError as e:
    print(f"❌ RL-PID controller import failed: {e}")

class AnimatedTrajectoryDemo:
    """Animated Trajectory Demonstration for Three Controllers"""
    
    def __init__(self):
        self.scenario_configs = {
            'scenario_1': {
                'seed': 58,
                'start': (0.5, 0.5),
                'goal': (9.5, 9.5),
                'description': 'Scenario 1: (0.5,0.5) → (9.5,9.5)'
            },
            'scenario_2': {
                'seed': 8,
                'start': (9.5, 9.5),
                'goal': (9.5, 0.5),
                'description': 'Scenario 2: (9.5,9.5) → (0.5,9.5)'
            }
        }
        
        # Create output directory
        os.makedirs('./animated_demos', exist_ok=True)
    
    def filter_trajectory_jumps(self, positions, max_jump=2.0):
        """Filter out unrealistic position jumps"""
        if len(positions) <= 1:
            return positions
            
        filtered_positions = [positions[0]]
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            if distance < max_jump:
                filtered_positions.append(positions[i])
            else:
                break
        return np.array(filtered_positions)
    
    def collect_trajectory_data(self, config):
        """Collect trajectory data from all available controllers"""
        trajectory_data = {}
        
        # Collect SAC data
        if SAC_AVAILABLE:
            sac_data = self.run_sac_controller(config)
            if sac_data:
                trajectory_data['SAC'] = {
                    'trajectory': sac_data['trajectory'],
                    'color': 'blue',
                    'environment': sac_data['environment']
                }
        
        # Collect PID data
        if PID_AVAILABLE:
            pid_data = self.run_pid_controller(config)
            if pid_data:
                trajectory_data['PID'] = {
                    'trajectory': pid_data['trajectory'],
                    'color': 'orange',
                    'environment': pid_data['environment']
                }
        
        # Collect RL-PID data
        if RLPID_AVAILABLE:
            rl_pid_data = self.run_rl_pid_controller(config)
            if rl_pid_data:
                trajectory_data['SAC-PID'] = {
                    'trajectory': rl_pid_data['trajectory'],
                    'color': 'purple',
                    'environment': rl_pid_data['environment']
                }
        
        return trajectory_data
    
    def run_sac_controller(self, config):
        """Run SAC controller and return trajectory data"""
        try:
            model_path = './gym3_change_big_different_buffer/results/sac_different_buffer/final_model.zip'
            norm_path = './gym3_change_big_different_buffer/results/sac_different_buffer/vec_normalize.pkl'
            
            if not os.path.exists(model_path):
                return None
            
            env = SACEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                max_steps=500
            )
            
            vec_env = DummyVecEnv([lambda: env])
            
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            
            model = SAC.load(model_path, env=vec_env)
            
            obs = vec_env.reset()
            done = False
            step_count = 0
            positions = [env.pos.copy()]
            
            while not done and step_count < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                positions.append(env.pos.copy())
                step_count += 1
            
            positions = self.filter_trajectory_jumps(np.array(positions))
            
            return {
                'trajectory': positions,
                'environment': env
            }
            
        except Exception as e:
            print(f"SAC controller error: {e}")
            return None
    
    def run_pid_controller(self, config):
        """Run PID controller and return trajectory data"""
        try:
            env = PIDTrackingEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                kp_x=4.36, ki_x=0.1, kd_x=2.8,
                kp_y=3.24, ki_y=0.21, kd_y=2.08,
                max_steps=500
            )
            
            results = env.run_simulation()
            
            if 'trajectory' not in results:
                return None
            
            positions = self.filter_trajectory_jumps(results['trajectory'])
            
            return {
                'trajectory': positions,
                'environment': env
            }
            
        except Exception as e:
            print(f"PID controller error: {e}")
            return None
    
    def run_rl_pid_controller(self, config):
        """Run RL-PID controller and return trajectory data"""
        try:
            model_path = './rl_pid/results/rl_pid_-1to1/final_model.zip'
            norm_path = './rl_pid/results/rl_pid_-1to1/vec_normalize.pkl'
            
            if not os.path.exists(model_path):
                return None
            
            env = RLPIDTrackingEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                max_steps=500
            )
            
            vec_env = DummyVecEnv([lambda: env])
            
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            
            model = SAC.load(model_path, env=vec_env)
            
            obs = vec_env.reset()
            done = False
            step_count = 0
            positions = [env.pos.copy()]
            
            while not done and step_count < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                positions.append(env.pos.copy())
                step_count += 1
            
            positions = self.filter_trajectory_jumps(np.array(positions))
            
            return {
                'trajectory': positions,
                'environment': env
            }
            
        except Exception as e:
            print(f"RL-PID controller error: {e}")
            return None
    
    def create_animated_demo(self, scenario_name, config):
        """Create animated trajectory demonstration"""
        print(f"\n🎬 Creating animated demo for {config['description']}")
        
        # Collect trajectory data
        trajectory_data = self.collect_trajectory_data(config)
        
        if not trajectory_data:
            print("❌ No trajectory data available for animation")
            return False
        
        # Get environment from first available controller
        env = list(trajectory_data.values())[0]['environment']
        
        # Find the maximum trajectory length for animation timing
        max_length = max(len(data['trajectory']) for data in trajectory_data.values())
        
        # Setup the figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw static elements (obstacles, reference path, start/goal)
        if hasattr(env, 'obstacles'):
            for i, (ox, oy, r) in enumerate(env.obstacles):
                circle = plt.Circle((ox, oy), r, color='gray', alpha=0.8, zorder=5)
                ax.add_patch(circle)
        
        # Draw reference path
        if hasattr(env, 'traj') and len(env.traj) > 1:
            ax.plot(env.traj[:, 0], env.traj[:, 1], 'k--', linewidth=2, 
                   alpha=0.7, label='Reference Path')
            
            # Mark waypoints
            for i, point in enumerate(env.traj):
                ax.scatter(point[0], point[1], c='gray', s=30, 
                          marker='o', alpha=0.8, zorder=3)
        
        # Draw start and goal
        ax.scatter(env.start[0], env.start[1], c='green', marker='s', s=200, 
                  label='Start', edgecolor='black', linewidth=2, zorder=10)
        ax.scatter(env.goal[0], env.goal[1], c='red', marker='*', s=400, 
                  label='Goal', edgecolor='black', linewidth=2, zorder=10)
        
        # Initialize trajectory lines and robot markers
        lines = {}
        robots = {}
        trails = {}
        
        for controller_name, data in trajectory_data.items():
            color = data['color']
            # Trajectory line (will be updated frame by frame)
            line, = ax.plot([], [], color=color, linewidth=3, alpha=0.8, label=controller_name)
            lines[controller_name] = line
            
            # Robot marker (current position)
            robot = ax.scatter([], [], c=color, s=150, marker='o', 
                             edgecolor='black', linewidth=2, zorder=8)
            robots[controller_name] = robot
            
            # Trail (fading trajectory)
            trail, = ax.plot([], [], color=color, linewidth=1, alpha=0.3)
            trails[controller_name] = trail
        
        # Set plot properties
        ax.set_title(f'Animated Trajectory Demo - {config["description"]}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Set axis limits with some padding
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        
        def animate(frame):
            """Animation function called for each frame"""
            for controller_name, data in trajectory_data.items():
                trajectory = data['trajectory']
                
                # Calculate current position index based on frame
                if len(trajectory) > 1:
                    # Scale frame to trajectory length
                    traj_index = min(int(frame * len(trajectory) / max_length), len(trajectory) - 1)
                    
                    # Update trajectory line (show path up to current position)
                    if traj_index > 0:
                        lines[controller_name].set_data(
                            trajectory[:traj_index+1, 0], 
                            trajectory[:traj_index+1, 1]
                        )
                        
                        # Update trail (full trajectory with low alpha)
                        trails[controller_name].set_data(
                            trajectory[:, 0], 
                            trajectory[:, 1]
                        )
                    
                    # Update robot marker (current position)
                    current_pos = trajectory[traj_index]
                    robots[controller_name].set_offsets([current_pos])
                else:
                    # Single point trajectory
                    if len(trajectory) == 1:
                        robots[controller_name].set_offsets([trajectory[0]])
            
            return list(lines.values()) + list(robots.values()) + list(trails.values())
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_length + 20,  # Add extra frames at the end
            interval=100, blit=True, repeat=True
        )
        
        # Save as GIF
        gif_filename = f'./animated_demos/{scenario_name}_animated_demo.gif'
        print(f"📹 Saving animated GIF: {gif_filename}")
        
        try:
            # Use pillow writer for better GIF quality
            writer = animation.PillowWriter(fps=10)
            anim.save(gif_filename, writer=writer)
            print(f"✅ Animated demo saved: {gif_filename}")
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")
            # Fallback to PNG frames
            print("🔄 Saving as PNG sequence instead...")
            os.makedirs(f'./animated_demos/{scenario_name}_frames', exist_ok=True)
            for i in range(max_length + 20):
                animate(i)
                plt.savefig(f'./animated_demos/{scenario_name}_frames/frame_{i:03d}.png', 
                           dpi=150, bbox_inches='tight')
            print(f"✅ PNG sequence saved to: ./animated_demos/{scenario_name}_frames/")
        
        plt.close()
        return True
    
    def generate_all_demos(self):
        """Generate animated demonstrations for all scenarios"""
        print("🎬 Animated Trajectory Demonstration Generator")
        print("Creating GIF animations for trajectory comparisons")
        
        success_count = 0
        total_scenarios = len(self.scenario_configs)
        
        for scenario_name, config in self.scenario_configs.items():
            if self.create_animated_demo(scenario_name, config):
                success_count += 1
        
        print(f"\n🎉 Animation Generation Complete!")
        print(f"📊 Success Rate: {success_count}/{total_scenarios} scenarios completed")
        print(f"📁 All animated demos saved to: ./animated_demos/")
        
        return success_count == total_scenarios

def main():
    """Main function"""
    print("🎬 Animated Trajectory Demonstration")
    print("Generating GIF animations for controller comparisons")
    print()
    
    demo = AnimatedTrajectoryDemo()
    demo.generate_all_demos()

if __name__ == "__main__":
    main()