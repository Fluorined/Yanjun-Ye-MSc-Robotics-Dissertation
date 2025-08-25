#!/usr/bin/env python3
"""
Combined Trajectory Visualization for Three Controllers
Creates clean visualizations with all three controllers on the same map
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

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

class CombinedTrajectoryVisualization:
    """Combined Trajectory Visualization for Three Controllers"""
    
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
        os.makedirs('./combined_visualization', exist_ok=True)
        
        # Performance data storage
        self.performance_data = {}
    
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
    
    def run_sac_controller(self, config):
        """Run SAC controller and return trajectory data"""
        if not SAC_AVAILABLE:
            return None
            
        try:
            # Model and normalization paths
            model_path = './gym3_change_big_different_buffer/results/sac_different_buffer/final_model.zip'
            norm_path = './gym3_change_big_different_buffer/results/sac_different_buffer/vec_normalize.pkl'
            
            if not os.path.exists(model_path):
                return None
            
            # Create environment
            env = SACEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                max_steps=500
            )
            
            # Create vectorized environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Load normalization if available
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            
            # Load model with environment
            model = SAC.load(model_path, env=vec_env)
            
            # Run simulation
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
            
            # Filter jumps and calculate metrics
            positions = self.filter_trajectory_jumps(np.array(positions))
            final_distance = np.linalg.norm(env.pos - env.goal)
            success = final_distance < 0.5
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)) if len(positions) > 1 else 0
            
            return {
                'trajectory': positions,
                'steps': step_count,
                'success': success,
                'final_distance': final_distance,
                'path_length': path_length,
                'environment': env
            }
            
        except Exception as e:
            print(f"SAC controller error: {e}")
            return None
    
    def run_pid_controller(self, config):
        """Run PID controller and return trajectory data"""
        if not PID_AVAILABLE:
            return None
            
        try:
            # Create environment
            env = PIDTrackingEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                kp_x=4.36, ki_x=0.1, kd_x=2.8,
                kp_y=3.24, ki_y=0.21, kd_y=2.08,
                max_steps=500
            )
            
            # Run simulation
            results = env.run_simulation()
            
            if 'trajectory' not in results:
                return None
            
            # Filter jumps and extract data
            positions = self.filter_trajectory_jumps(results['trajectory'])
            
            return {
                'trajectory': positions,
                'steps': results.get('steps', len(positions)),
                'success': results.get('success', False),
                'final_distance': results.get('final_distance', 0),
                'path_length': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)) if len(positions) > 1 else 0,
                'environment': env
            }
            
        except Exception as e:
            print(f"PID controller error: {e}")
            return None
    
    def run_rl_pid_controller(self, config):
        """Run RL-PID controller and return trajectory data"""
        if not RLPID_AVAILABLE:
            return None
            
        try:
            # Model and normalization paths
            model_path = './rl_pid/results/rl_pid_-1to1/final_model.zip'
            norm_path = './rl_pid/results/rl_pid_-1to1/vec_normalize.pkl'
            
            if not os.path.exists(model_path):
                return None
            
            # Create environment
            env = RLPIDTrackingEnv(
                start=config['start'],
                goal=config['goal'],
                seed=config['seed'],
                max_steps=500
            )
            
            # Create vectorized environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Load normalization if available
            if os.path.exists(norm_path):
                vec_env = VecNormalize.load(norm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            
            # Load model with environment
            model = SAC.load(model_path, env=vec_env)
            
            # Run simulation
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
            
            # Filter jumps and calculate metrics
            positions = self.filter_trajectory_jumps(np.array(positions))
            final_distance = np.linalg.norm(env.pos - env.goal)
            success = final_distance < 0.5
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)) if len(positions) > 1 else 0
            
            return {
                'trajectory': positions,
                'steps': step_count,
                'success': success,
                'final_distance': final_distance,
                'path_length': path_length,
                'environment': env
            }
            
        except Exception as e:
            print(f"RL-PID controller error: {e}")
            return None
    
    def calculate_trajectory_mse(self, trajectory, reference_path):
        """Calculate MSE between trajectory and reference path"""
        if len(trajectory) == 0 or len(reference_path) == 0:
            return float('inf')
        
        # Find closest reference point for each trajectory point
        mse_values = []
        for traj_point in trajectory:
            distances = [np.linalg.norm(traj_point - ref_point) for ref_point in reference_path]
            min_distance = min(distances)
            mse_values.append(min_distance ** 2)
        
        return np.mean(mse_values)
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # If line segment has zero length, return distance to point
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point_vec)
        
        # Project point onto line segment
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        
        return np.linalg.norm(point - projection)
    
    def calculate_cross_track_error(self, trajectory, reference_path):
        """Calculate Cross Track Error (CTE) statistics"""
        if len(trajectory) == 0 or len(reference_path) < 2:
            return {'mean_cte': float('inf'), 'max_cte': float('inf'), 'rms_cte': float('inf'), 'mse_cte': float('inf')}
        
        cte_values = []
        
        for traj_point in trajectory:
            min_cte = float('inf')
            
            # Find minimum cross track error to any line segment in reference path
            for i in range(len(reference_path) - 1):
                line_start = reference_path[i]
                line_end = reference_path[i + 1]
                cte = self.point_to_line_distance(traj_point, line_start, line_end)
                min_cte = min(min_cte, cte)
            
            cte_values.append(min_cte)
        
        cte_array = np.array(cte_values)
        return {
            'mean_cte': np.mean(cte_array),
            'max_cte': np.max(cte_array),
            'rms_cte': np.sqrt(np.mean(cte_array ** 2)),
            'mse_cte': np.mean(cte_array ** 2)
        }
    
    def create_combined_visualization(self, scenario_name, config):
        """Create combined trajectory visualization for all three controllers"""
        print(f"\\n🎬 Creating combined visualization for {config['description']}")
        
        # Run all controllers
        sac_data = self.run_sac_controller(config)
        pid_data = self.run_pid_controller(config)
        rl_pid_data = self.run_rl_pid_controller(config)
        
        # Get valid data
        valid_controllers = []
        if sac_data:
            valid_controllers.append(('SAC', sac_data, 'blue'))
        if pid_data:
            valid_controllers.append(('PID', pid_data, 'orange'))
        if rl_pid_data:
            valid_controllers.append(('SAC-PID', rl_pid_data, 'purple'))
        
        if not valid_controllers:
            print("❌ No controllers succeeded")
            return False
        
        # Use environment from first valid controller
        env = valid_controllers[0][1]['environment']
        
        # Calculate MSE for each controller
        reference_path = env.traj if hasattr(env, 'traj') else []
        
        # Calculate reference trajectory length
        reference_length = 0
        if len(reference_path) > 1:
            reference_length = np.sum(np.linalg.norm(np.diff(reference_path, axis=0), axis=1))
        
        scenario_data = {'reference_length': reference_length}
        for controller_name, data, _ in valid_controllers:
            mse = self.calculate_trajectory_mse(data['trajectory'], reference_path)
            cte_stats = self.calculate_cross_track_error(data['trajectory'], reference_path)
            scenario_data[controller_name] = {
                'steps': data['steps'],
                'path_length': data['path_length'],
                'mse': mse,
                'mean_cte': cte_stats['mean_cte'],
                'max_cte': cte_stats['max_cte'],
                'rms_cte': cte_stats['rms_cte'],
                'mse_cte': cte_stats['mse_cte'],
                'success': data['success']
            }
        
        # Store performance data
        self.performance_data[scenario_name] = scenario_data
        
        # Create clean visualization
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Draw obstacles
        if hasattr(env, 'obstacles'):
            for i, (ox, oy, r) in enumerate(env.obstacles):
                circle = plt.Circle((ox, oy), r, color='gray', alpha=0.8, zorder=5)
                ax.add_patch(circle)
        
        # Draw reference path
        if hasattr(env, 'traj') and len(env.traj) > 1:
            plt.plot(env.traj[:, 0], env.traj[:, 1], 'k--', linewidth=2, 
                    alpha=0.7, label='Reference Path')
            
            # Mark waypoints
            for i, point in enumerate(env.traj):
                plt.scatter(point[0], point[1], c='gray', s=30, 
                          marker='o', alpha=0.8, zorder=3)
        
        # Draw start and goal
        plt.scatter(env.start[0], env.start[1], c='green', marker='s', s=200, 
                   label='Start', edgecolor='black', linewidth=2, zorder=10)
        plt.scatter(env.goal[0], env.goal[1], c='red', marker='*', s=400, 
                   label='Goal', edgecolor='black', linewidth=2, zorder=10)
        
        # Draw trajectories
        for controller_name, data, color in valid_controllers:
            trajectory = data['trajectory']
            if len(trajectory) > 1:
                plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, 
                        linewidth=3, alpha=0.8, label=controller_name)
        
        # Set clean plot properties
        plt.title(f'Controller Comparison - {config["description"]}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('X Position (m)', fontsize=14)
        plt.ylabel('Y Position (m)', fontsize=14)
        plt.legend(fontsize=16, loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Remove extra elements, keep only essentials
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save figure
        filename = f'./combined_visualization/{scenario_name}_combined_trajectories.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Combined visualization saved: {filename}")
        return True
    
    def generate_performance_report(self):
        """Generate performance comparison document"""
        report_content = """# Controller Performance Comparison Report

## Summary
This document presents a comprehensive comparison of three control algorithms:
- **SAC**: Soft Actor-Critic (Reinforcement Learning)
- **PID**: Proportional-Integral-Derivative (Classical Control)
- **SAC-PID**: Reinforcement Learning enhanced PID (Hybrid Approach)

## Test Scenarios
Two navigation scenarios were evaluated:
1. **Scenario 1**: Navigation from (0.5, 0.5) to (9.5, 9.5)
2. **Scenario 2**: Navigation from (9.5, 9.5) to (0.5, 9.5)

## Performance Metrics

### Scenario 1: (0.5,0.5) → (9.5,9.5)

**Reference Trajectory Length**: {ref_length_1:.3f} m

| Controller | Steps | Path Length (m) | MSE | Mean CTE | Max CTE | RMS CTE | MSE CTE | Success |
|------------|-------|-----------------|-----|----------|---------|---------|---------|---------|
"""
        
        for scenario_name, scenario_data in self.performance_data.items():
            if scenario_name == 'scenario_1':
                for controller, metrics in scenario_data.items():
                    if controller != 'reference_length' and isinstance(metrics, dict):
                        success_str = "✓" if metrics['success'] else "✗"
                        report_content += f"| {controller} | {metrics['steps']} | {metrics['path_length']:.3f} | {metrics['mse']:.6f} | {metrics['mean_cte']:.6f} | {metrics['max_cte']:.6f} | {metrics['rms_cte']:.6f} | {metrics['mse_cte']:.6f} | {success_str} |\\n"
        
        report_content += "\\n### Scenario 2: (9.5,9.5) → (0.5,9.5)\\n\\n"
        report_content += "**Reference Trajectory Length**: {ref_length_2:.3f} m\\n\\n"
        report_content += "| Controller | Steps | Path Length (m) | MSE | Mean CTE | Max CTE | RMS CTE | MSE CTE | Success |\\n"
        report_content += "|------------|-------|-----------------|-----|----------|---------|---------|---------|---------|\\n"
        
        for scenario_name, scenario_data in self.performance_data.items():
            if scenario_name == 'scenario_2':
                for controller, metrics in scenario_data.items():
                    if controller != 'reference_length' and isinstance(metrics, dict):
                        success_str = "✓" if metrics['success'] else "✗"
                        report_content += f"| {controller} | {metrics['steps']} | {metrics['path_length']:.3f} | {metrics['mse']:.6f} | {metrics['mean_cte']:.6f} | {metrics['max_cte']:.6f} | {metrics['rms_cte']:.6f} | {metrics['mse_cte']:.6f} | {success_str} |\\n"
        
        report_content += """
## Analysis

### Key Findings
- **Step Efficiency**: Number of simulation steps required to complete the task
- **Path Length**: Total distance traveled by the robot
- **MSE**: Mean Squared Error between actual trajectory and reference path (lower is better)
- **Mean CTE**: Average Cross Track Error - perpendicular distance from trajectory to reference path (lower is better)
- **Max CTE**: Maximum Cross Track Error - largest deviation from reference path (lower is better)
- **RMS CTE**: Root Mean Square Cross Track Error - overall tracking performance measure (lower is better)
- **MSE CTE**: Mean Squared Error of Cross Track Error - emphasizes larger deviations more heavily (lower is better)
- **Success Rate**: Whether the controller successfully reached the target within tolerance

### Performance Summary
The performance metrics provide insights into the trade-offs between different control approaches:
- Classical PID controllers offer predictable performance
- Reinforcement learning approaches may provide adaptive behavior
- Hybrid SAC-PID methods combine benefits of both approaches

---
*Report generated automatically from simulation data*
"""
        
        # Format the report with reference trajectory lengths
        ref_length_1 = self.performance_data.get('scenario_1', {}).get('reference_length', 0)
        ref_length_2 = self.performance_data.get('scenario_2', {}).get('reference_length', 0)
        
        formatted_report = report_content.format(
            ref_length_1=ref_length_1,
            ref_length_2=ref_length_2
        )
        
        # Save report
        report_path = './combined_visualization/performance_report.md'
        with open(report_path, 'w') as f:
            f.write(formatted_report)
        
        print(f"✅ Performance report saved: {report_path}")
    
    def run_comparison(self):
        """Run complete comparison for all scenarios"""
        print("🚀 Combined Trajectory Comparison")
        print("Generating clean visualizations with all three controllers")
        
        success_count = 0
        total_scenarios = len(self.scenario_configs)
        
        for scenario_name, config in self.scenario_configs.items():
            if self.create_combined_visualization(scenario_name, config):
                success_count += 1
        
        # Generate performance report
        if self.performance_data:
            self.generate_performance_report()
        
        print(f"\\n🎉 Comparison Complete!")
        print(f"📊 Success Rate: {success_count}/{total_scenarios} scenarios completed")
        print(f"📁 All files saved to: ./combined_visualization/")

def main():
    """Main function"""
    print("🚀 Combined Trajectory Visualization")
    print("Creating clean trajectory comparisons with performance metrics")
    print()
    
    comparison = CombinedTrajectoryVisualization()
    comparison.run_comparison()

if __name__ == "__main__":
    main()