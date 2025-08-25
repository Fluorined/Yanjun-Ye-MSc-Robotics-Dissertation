#!/usr/bin/env python3
"""
Acceleration Data Visualization
Creates comparison plots and boxplots for ax, ay acceleration data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class AccelerationVisualization:
    """Create acceleration comparison visualizations"""
    
    def __init__(self):
        # Create output directory
        os.makedirs('./acceleration_plots', exist_ok=True)
        
        # Load acceleration data
        self.data_path = './acceleration_data/all_controllers_acceleration_data.csv'
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Acceleration data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        # Rename RL-PID to SAC-PID for consistency
        self.df['controller'] = self.df['controller'].replace('RL-PID', 'SAC-PID')
        print(f"✅ Loaded acceleration data: {len(self.df)} data points")
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def apply_moving_average_filter(self, data, window_size=10):
        """Apply moving average filter to data"""
        if len(data) < window_size:
            return data
        
        filtered_data = []
        for i in range(len(data)):
            if i < window_size - 1:
                # For the first few points, use available data
                window_data = data[:i+1]
            else:
                # Use full window
                window_data = data[i-window_size+1:i+1]
            filtered_data.append(np.mean(window_data))
        
        return np.array(filtered_data)
    
    def plot_ax_comparison_by_scenario(self):
        """Plot ax acceleration comparison for each scenario with moving average filter"""
        scenarios = self.df['scenario'].unique()
        controllers = ['SAC', 'PID', 'SAC-PID']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for scenario in scenarios:
            # Create subplot figure for original and filtered data
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            scenario_data = self.df[self.df['scenario'] == scenario]
            
            for i, controller in enumerate(controllers):
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    # Original data
                    ax1.plot(controller_data['step'], controller_data['ax'], 
                           color=colors[i], linewidth=1.5, alpha=0.6, label=f'{controller} (Original)')
                    
                    # Filtered data
                    filtered_ax = self.apply_moving_average_filter(controller_data['ax'].values, window_size=10)
                    ax2.plot(controller_data['step'], filtered_ax, 
                           color=colors[i], linewidth=2.5, alpha=0.9, label=f'{controller} (Filtered)')
            
            # Set plot properties for original data
            scenario_title = "Scenario 1: (0.5,0.5) → (9.5,9.5)" if scenario == 'scenario_1' else "Scenario 2: (9.5,9.5) → (0.5,9.5)"
            ax1.set_title(f'Original X-axis Acceleration - {scenario_title}', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('X-axis Acceleration (ax)', fontsize=12)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # Set plot properties for filtered data
            ax2.set_title(f'Moving Average Filtered X-axis Acceleration (Window=10) - {scenario_title}', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('X-axis Acceleration (ax)', fontsize=12)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Limit (±1)')
            ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # Save plot
            plt.tight_layout()
            filename = f'./acceleration_plots/{scenario}_ax_comparison_with_filter.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {scenario} ax comparison with filter saved: {filename}")
            
            # Also create individual filtered comparison plot
            plt.figure(figsize=(14, 8))
            
            for i, controller in enumerate(controllers):
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    filtered_ax = self.apply_moving_average_filter(controller_data['ax'].values, window_size=10)
                    plt.plot(controller_data['step'], filtered_ax, 
                           color=colors[i], linewidth=2.5, alpha=0.9, label=f'{controller}')
            
            plt.title(f'Filtered X-axis Acceleration Comparison (Moving Average, Window=10)\n{scenario_title}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Step', fontsize=14)
            plt.ylabel('X-axis Acceleration (ax)', fontsize=14)
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Limit (+1)')
            plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Limit (-1)')
            
            filename_clean = f'./acceleration_plots/{scenario}_ax_filtered_comparison.png'
            plt.tight_layout()
            plt.savefig(filename_clean, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {scenario} filtered ax comparison saved: {filename_clean}")
    
    def plot_ay_comparison_by_scenario(self):
        """Plot ay acceleration comparison for each scenario with moving average filter"""
        scenarios = self.df['scenario'].unique()
        controllers = ['SAC', 'PID', 'SAC-PID']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for scenario in scenarios:
            # Create subplot figure for original and filtered data
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            scenario_data = self.df[self.df['scenario'] == scenario]
            
            for i, controller in enumerate(controllers):
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    # Original data
                    ax1.plot(controller_data['step'], controller_data['ay'], 
                           color=colors[i], linewidth=1.5, alpha=0.6, label=f'{controller} (Original)')
                    
                    # Filtered data
                    filtered_ay = self.apply_moving_average_filter(controller_data['ay'].values, window_size=10)
                    ax2.plot(controller_data['step'], filtered_ay, 
                           color=colors[i], linewidth=2.5, alpha=0.9, label=f'{controller} (Filtered)')
            
            # Set plot properties for original data
            scenario_title = "Scenario 1: (0.5,0.5) → (9.5,9.5)" if scenario == 'scenario_1' else "Scenario 2: (9.5,9.5) → (0.5,9.5)"
            ax1.set_title(f'Original Y-axis Acceleration - {scenario_title}', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('Y-axis Acceleration (ay)', fontsize=12)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # Set plot properties for filtered data
            ax2.set_title(f'Moving Average Filtered Y-axis Acceleration (Window=10) - {scenario_title}', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Y-axis Acceleration (ay)', fontsize=12)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Limit (±1)')
            ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # Save plot
            plt.tight_layout()
            filename = f'./acceleration_plots/{scenario}_ay_comparison_with_filter.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {scenario} ay comparison with filter saved: {filename}")
            
            # Also create individual filtered comparison plot
            plt.figure(figsize=(14, 8))
            
            for i, controller in enumerate(controllers):
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    filtered_ay = self.apply_moving_average_filter(controller_data['ay'].values, window_size=10)
                    plt.plot(controller_data['step'], filtered_ay, 
                           color=colors[i], linewidth=2.5, alpha=0.9, label=f'{controller}')
            
            plt.title(f'Filtered Y-axis Acceleration Comparison (Moving Average, Window=10)\n{scenario_title}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Step', fontsize=14)
            plt.ylabel('Y-axis Acceleration (ay)', fontsize=14)
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Limit (+1)')
            plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Limit (-1)')
            
            filename_clean = f'./acceleration_plots/{scenario}_ay_filtered_comparison.png'
            plt.tight_layout()
            plt.savefig(filename_clean, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {scenario} filtered ay comparison saved: {filename_clean}")
    
    def plot_acceleration_boxplots(self):
        """Create separate boxplots for each scenario with ax and ay side by side, including RMSE labels"""
        scenarios = ['scenario_1', 'scenario_2']
        scenario_titles = ['Scenario 1: (0.5,0.5) → (9.5,9.5)', 'Scenario 2: (9.5,9.5) → (0.5,9.5)']
        controllers = ['SAC', 'PID', 'SAC-PID']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, (scenario, title) in enumerate(zip(scenarios, scenario_titles)):
            # Create separate figure for each scenario
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            scenario_data = self.df[self.df['scenario'] == scenario]
            
            # Prepare data for grouped boxplot
            positions = []
            box_data = []
            labels = []
            box_colors = []
            rmse_values = []
            
            # Group ax and ay data
            for j, acceleration_type in enumerate(['ax', 'ay']):
                for k, controller in enumerate(controllers):
                    controller_data = scenario_data[scenario_data['controller'] == controller][acceleration_type].values
                    box_data.append(controller_data)
                    positions.append(j * 4 + k + 1)  # Spacing between groups
                    labels.append(f'{controller}')
                    box_colors.append(colors[k])
                    
                    # Calculate RMSE for this controller and acceleration type
                    rmse = np.sqrt(np.mean(controller_data**2))
                    rmse_values.append(rmse)
            
            # Create boxplot
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
            
            # Customize boxplot colors
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Make median lines more prominent
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
                median.set_alpha(1.0)
            
            # Add RMSE text labels slightly below the top of each box
            for i, (pos, rmse, color) in enumerate(zip(positions, rmse_values, box_colors)):
                # Get the 75th percentile (top of box) and position slightly below
                box_data_values = box_data[i]
                top_of_box = np.percentile(box_data_values, 75)
                median_val = np.percentile(box_data_values, 50)
                # Position between 75th percentile and median (around 65th percentile)
                label_position = median_val + 0.7 * (top_of_box - median_val)
                ax.text(pos, label_position, f'RMSE\n{rmse:.3f}', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))
            
            # Set x-axis labels and ticks
            ax.set_xticks([2, 6])  # Centers of the two groups
            ax.set_xticklabels(['ax (X-axis Acceleration)', 'ay (Y-axis Acceleration)'], fontsize=14)
            
            # Add legend with RMSE information
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[k], alpha=0.7, 
                              label=f'{controller}') for k, controller in enumerate(controllers)]
            ax.legend(handles=legend_elements, fontsize=16, loc='best', title='Controllers', title_fontsize=14)
            
            # Set plot properties
            ax.set_title(f'{title}\nAcceleration Distribution Comparison\n(RMSE values shown inside each box)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Acceleration Value', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Limit (+1)')
            ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Limit (-1)')
            
            # Save individual plot
            plt.tight_layout()
            filename = f'./acceleration_plots/{scenario}_acceleration_boxplots.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ {scenario} acceleration boxplots saved: {filename}")
    
    def plot_acceleration_magnitude_boxplots(self):
        """Create boxplots for acceleration magnitude"""
        plt.figure(figsize=(14, 8))
        
        scenarios = ['scenario_1', 'scenario_2']
        scenario_titles = ['Scenario 1', 'Scenario 2']
        controllers = ['SAC', 'PID', 'SAC-PID']
        
        # Prepare data for grouped boxplot
        plot_data = []
        for scenario, scenario_title in zip(scenarios, scenario_titles):
            scenario_data = self.df[self.df['scenario'] == scenario]
            for controller in controllers:
                controller_data = scenario_data[scenario_data['controller'] == controller]
                for magnitude in controller_data['acceleration_magnitude']:
                    plot_data.append({
                        'Scenario': scenario_title,
                        'Controller': controller,
                        'Acceleration Magnitude': magnitude
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped boxplot
        sns.boxplot(data=plot_df, x='Controller', y='Acceleration Magnitude', 
                   hue='Scenario', palette=['lightblue', 'lightcoral'])
        
        plt.title('Acceleration Magnitude Distribution Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Controller', fontsize=14)
        plt.ylabel('Acceleration Magnitude', fontsize=14)
        plt.legend(title='Scenario', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add reference line at √2 (maximum possible magnitude with ±1 limits)
        plt.axhline(y=np.sqrt(2), color='red', linestyle='--', alpha=0.7, 
                   label=f'Theoretical Max (√2 ≈ {np.sqrt(2):.3f})')
        
        # Save plot
        plt.tight_layout()
        filename = './acceleration_plots/acceleration_magnitude_boxplots.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Acceleration magnitude boxplots saved: {filename}")
    
    def generate_acceleration_statistics_report(self):
        """Generate detailed statistics report"""
        report_lines = ["# Acceleration Data Analysis Report\\n"]
        
        scenarios = ['scenario_1', 'scenario_2']
        scenario_titles = ['Scenario 1: (0.5,0.5) → (9.5,9.5)', 'Scenario 2: (9.5,9.5) → (0.5,9.5)']
        controllers = ['SAC', 'PID', 'SAC-PID']
        
        for scenario, title in zip(scenarios, scenario_titles):
            report_lines.append(f"## {title}\\n")
            scenario_data = self.df[self.df['scenario'] == scenario]
            
            # X-axis acceleration statistics
            report_lines.append("### X-axis Acceleration (ax) Statistics\\n")
            report_lines.append("| Controller | Mean | Std | Min | Max | Median |")
            report_lines.append("|------------|------|-----|-----|-----|--------|")
            
            for controller in controllers:
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    ax_stats = controller_data['ax'].describe()
                    report_lines.append(f"| {controller} | {ax_stats['mean']:.4f} | {ax_stats['std']:.4f} | {ax_stats['min']:.4f} | {ax_stats['max']:.4f} | {ax_stats['50%']:.4f} |")
            
            report_lines.append("\\n")
            
            # Y-axis acceleration statistics
            report_lines.append("### Y-axis Acceleration (ay) Statistics\\n")
            report_lines.append("| Controller | Mean | Std | Min | Max | Median |")
            report_lines.append("|------------|------|-----|-----|-----|--------|")
            
            for controller in controllers:
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    ay_stats = controller_data['ay'].describe()
                    report_lines.append(f"| {controller} | {ay_stats['mean']:.4f} | {ay_stats['std']:.4f} | {ay_stats['min']:.4f} | {ay_stats['max']:.4f} | {ay_stats['50%']:.4f} |")
            
            report_lines.append("\\n")
            
            # Acceleration magnitude statistics
            report_lines.append("### Acceleration Magnitude Statistics\\n")
            report_lines.append("| Controller | Mean | Std | Min | Max | Median |")
            report_lines.append("|------------|------|-----|-----|-----|--------|")
            
            for controller in controllers:
                controller_data = scenario_data[scenario_data['controller'] == controller]
                if not controller_data.empty:
                    mag_stats = controller_data['acceleration_magnitude'].describe()
                    report_lines.append(f"| {controller} | {mag_stats['mean']:.4f} | {mag_stats['std']:.4f} | {mag_stats['min']:.4f} | {mag_stats['max']:.4f} | {mag_stats['50%']:.4f} |")
            
            report_lines.append("\\n---\\n")
        
        # Save report
        report_content = "\\n".join(report_lines)
        report_path = './acceleration_plots/acceleration_statistics_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Statistics report saved: {report_path}")
    
    def create_all_visualizations(self):
        """Create all acceleration visualizations"""
        print("🚀 Creating Acceleration Visualizations")
        print("="*50)
        
        # Create ax comparison plots
        print("\\n📊 Creating ax acceleration comparison plots...")
        self.plot_ax_comparison_by_scenario()
        
        # Create ay comparison plots
        print("\\n📊 Creating ay acceleration comparison plots...")
        self.plot_ay_comparison_by_scenario()
        
        # Create boxplots
        print("\\n📦 Creating acceleration boxplots...")
        self.plot_acceleration_boxplots()
        
        # Create magnitude boxplots
        print("\\n📦 Creating acceleration magnitude boxplots...")
        self.plot_acceleration_magnitude_boxplots()
        
        # Generate statistics report
        print("\\n📋 Generating statistics report...")
        self.generate_acceleration_statistics_report()
        
        print("\\n🎉 All visualizations completed!")
        print(f"📁 All plots saved to: ./acceleration_plots/")

def main():
    """Main function"""
    print("🚀 Acceleration Data Visualization")
    print("Creating ax comparison plots and boxplots for all controllers")
    print()
    
    try:
        visualizer = AccelerationVisualization()
        visualizer.create_all_visualizations()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run acceleration_data_extraction.py first to generate the data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()