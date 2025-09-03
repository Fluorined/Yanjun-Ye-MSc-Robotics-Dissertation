# Comparison of SAC, SAC-PID, and PID Controllers

This project compares the performance of three different controllers (SAC, SAC-PID, and pure PID) in trajectory tracking tasks.

## Folder Structure

### `acceleration_data/`
Contains acceleration data and statistical analysis for all controllers
- CSV files with acceleration data for different scenarios and controllers
- Summary statistics

### `acceleration_plots/`
Visualization plots for acceleration data
- Box plots showing acceleration performance across scenarios

### `animated_demos/`
Animated demonstration files
- GIF animations showing controller performance in different scenarios

### `combined_visualization/`
Comprehensive visualization results
- Performance reports and trajectory comparison plots

### `gym3_change_big_different_buffer/`
SAC training environment with different buffer configurations
- RRT environment, validation scripts, and training results

### `pid_tracking/`
Pure PID controller implementation
- Core PID controller code and demonstration results

### `rl_pid/`
Reinforcement learning combined with PID implementation
- RL-PID controller code, training scripts, and results

## Main Script Files

- `acceleration_data_extraction.py` - Extract acceleration data from simulation results
- `acceleration_visualization.py` - Generate acceleration data visualizations
- `animated_trajectory_demo.py` - Create animated trajectory demonstrations
- `combined_trajectory_visualization.py` - Generate comprehensive trajectory comparison plots

## Usage

1. Run training scripts for each controller to train models
2. Use visualization scripts to generate comparison charts and animations
