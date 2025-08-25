#!/usr/bin/env python3
"""
Acceleration Data Extraction for Three Controllers
Based on combined_trajectory_visualization.py, extracts ax, ay acceleration data
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

class AccelerationDataExtraction:
    """Extract acceleration data for all three controllers"""
    
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
                'description': 'Scenario 2: (9.5,9.5) → (9.5,0.5)'
            }
        }
        
        # Create output directory
        os.makedirs('./acceleration_data', exist_ok=True)
        
        # All acceleration data storage
        self.all_acceleration_data = []
    
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
    
    def calculate_acceleration_from_positions(self, positions, dt=0.1):
        """Calculate acceleration from position data"""
        if len(positions) < 3:
            return np.array([]), np.array([])
        
        # Calculate velocity from positions
        velocities = np.diff(positions, axis=0) / dt
        
        # Calculate acceleration from velocities
        accelerations = np.diff(velocities, axis=0) / dt
        
        # Separate ax and ay
        ax = accelerations[:, 0]
        ay = accelerations[:, 1]
        
        return ax, ay
    
    def run_sac_controller_with_acceleration(self, config, scenario_name):
        """Run SAC controller and extract acceleration data"""
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
            actions = []
            
            while not done and step_count < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                actions.append(action[0].copy())  # Store action for analysis
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                positions.append(env.pos.copy())
                step_count += 1
            
            # Filter jumps and calculate acceleration
            positions = self.filter_trajectory_jumps(np.array(positions))
            ax, ay = self.calculate_acceleration_from_positions(positions)
            
            # Store acceleration data
            for i, (ax_val, ay_val) in enumerate(zip(ax, ay)):
                self.all_acceleration_data.append({
                    'scenario': scenario_name,
                    'controller': 'SAC',
                    'step': i + 1,
                    'ax': ax_val,
                    'ay': ay_val,
                    'acceleration_magnitude': np.sqrt(ax_val**2 + ay_val**2)
                })
            
            print(f"  SAC: {len(ax)} acceleration points recorded")
            return len(ax)
            
        except Exception as e:
            print(f"SAC controller error: {e}")
            return None
    
    def run_pid_controller_with_acceleration(self, config, scenario_name):
        """Run PID controller and extract acceleration data"""
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
            
            # Get acceleration data directly from PID environment
            positions = self.filter_trajectory_jumps(results['trajectory'])
            
            # Check if environment provides control accelerations
            if 'accelerations' in results:
                control_accels = results['accelerations']
                ax = control_accels[:, 0]
                ay = control_accels[:, 1]
            else:
                # Calculate from positions if not available
                ax, ay = self.calculate_acceleration_from_positions(positions)
            
            # Store acceleration data
            for i, (ax_val, ay_val) in enumerate(zip(ax, ay)):
                self.all_acceleration_data.append({
                    'scenario': scenario_name,
                    'controller': 'PID',
                    'step': i + 1,
                    'ax': ax_val,
                    'ay': ay_val,
                    'acceleration_magnitude': np.sqrt(ax_val**2 + ay_val**2)
                })
            
            print(f"  PID: {len(ax)} acceleration points recorded")
            return len(ax)
            
        except Exception as e:
            print(f"PID controller error: {e}")
            return None
    
    def run_rl_pid_controller_with_acceleration(self, config, scenario_name):
        """Run RL-PID controller and extract acceleration data"""
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
            actions = []
            
            while not done and step_count < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                actions.append(action[0].copy())  # Store action for analysis
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                positions.append(env.pos.copy())
                step_count += 1
            
            # Filter jumps and calculate acceleration
            positions = self.filter_trajectory_jumps(np.array(positions))
            ax, ay = self.calculate_acceleration_from_positions(positions)
            
            # Store acceleration data
            for i, (ax_val, ay_val) in enumerate(zip(ax, ay)):
                self.all_acceleration_data.append({
                    'scenario': scenario_name,
                    'controller': 'RL-PID',
                    'step': i + 1,
                    'ax': ax_val,
                    'ay': ay_val,
                    'acceleration_magnitude': np.sqrt(ax_val**2 + ay_val**2)
                })
            
            print(f"  RL-PID: {len(ax)} acceleration points recorded")
            return len(ax)
            
        except Exception as e:
            print(f"RL-PID controller error: {e}")
            return None
    
    def extract_acceleration_data(self, scenario_name, config):
        """Extract acceleration data for all controllers in a scenario"""
        print(f"\\n🔍 Extracting acceleration data for {config['description']}")
        
        # Run all controllers
        sac_count = self.run_sac_controller_with_acceleration(config, scenario_name)
        pid_count = self.run_pid_controller_with_acceleration(config, scenario_name)
        rl_pid_count = self.run_rl_pid_controller_with_acceleration(config, scenario_name)
        
        success_count = sum(1 for count in [sac_count, pid_count, rl_pid_count] if count is not None)
        print(f"  ✅ {success_count}/3 controllers completed successfully")
        
        return success_count > 0
    
    def export_acceleration_data(self):
        """Export all acceleration data to CSV"""
        if not self.all_acceleration_data:
            print("❌ No acceleration data to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_acceleration_data)
        
        # Export main CSV file
        csv_path = './acceleration_data/all_controllers_acceleration_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ All acceleration data exported: {csv_path}")
        
        # Export separate CSV files for each scenario and controller
        for scenario in df['scenario'].unique():
            for controller in df['controller'].unique():
                subset = df[(df['scenario'] == scenario) & (df['controller'] == controller)]
                if not subset.empty:
                    filename = f'./acceleration_data/{scenario}_{controller}_acceleration.csv'
                    subset.to_csv(filename, index=False)
                    print(f"  📁 {scenario} {controller}: {len(subset)} data points → {filename}")
        
        # Export summary statistics
        summary_stats = df.groupby(['scenario', 'controller']).agg({
            'ax': ['mean', 'std', 'min', 'max'],
            'ay': ['mean', 'std', 'min', 'max'],
            'acceleration_magnitude': ['mean', 'std', 'min', 'max']
        }).round(6)
        
        summary_path = './acceleration_data/acceleration_summary_statistics.csv'
        summary_stats.to_csv(summary_path)
        print(f"✅ Summary statistics exported: {summary_path}")
        
        # Print data overview
        print(f"\\n📊 Data Overview:")
        print(f"  Total acceleration points: {len(df)}")
        print(f"  Data distribution:")
        for scenario in df['scenario'].unique():
            print(f"    {scenario}:")
            for controller in df['controller'].unique():
                count = len(df[(df['scenario'] == scenario) & (df['controller'] == controller)])
                print(f"      {controller}: {count} points")
    
    def run_extraction(self):
        """Run complete acceleration data extraction"""
        print("🚀 Acceleration Data Extraction")
        print("Extracting ax, ay acceleration data for all three controllers")
        
        success_count = 0
        total_scenarios = len(self.scenario_configs)
        
        for scenario_name, config in self.scenario_configs.items():
            if self.extract_acceleration_data(scenario_name, config):
                success_count += 1
        
        # Export all data
        self.export_acceleration_data()
        
        print(f"\\n🎉 Extraction Complete!")
        print(f"📊 Success Rate: {success_count}/{total_scenarios} scenarios completed")
        print(f"📁 All data saved to: ./acceleration_data/")

def main():
    """Main function"""
    print("🚀 Controller Acceleration Data Extraction")
    print("Extracting ax, ay acceleration data with scenario and controller labels")
    print()
    
    extractor = AccelerationDataExtraction()
    extractor.run_extraction()

if __name__ == "__main__":
    main()