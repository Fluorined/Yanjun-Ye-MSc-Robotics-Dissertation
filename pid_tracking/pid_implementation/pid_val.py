# demo.py - PID控制系统演示脚本

import numpy as np
from pid_env import PIDTrackingEnv
import matplotlib.pyplot as plt
import argparse
import sys

def simple_demo(start=(0.5, 0.5), goal=(9.5, 9.5), seed=58, 
                kp_x=2.0, ki_x=0.1, kd_x=2,
                kp_y=2.0, ki_y=0.1, kd_y=2):
    """简单的PID轨迹跟踪演示"""
    print("PID轨迹跟踪系统演示")
    print("=" * 30)
    
    # 创建环境
    env = PIDTrackingEnv(
        start=start,
        goal=goal,
        seed=seed,
        kp_x=kp_x, ki_x=ki_x, kd_x=kd_x,
        kp_y=kp_y, ki_y=ki_y, kd_y=kd_y,
        max_steps=300
    )
    
    print(f"环境创建完成:")
    print(f"  起点: {env.start}")
    print(f"  终点: {env.goal}")
    print(f"  障碍物数量: {len(env.obstacles)}")
    print(f"  参考路径点数: {len(env.traj)}")
    print(f"  X轴PID参数: Kp_x={env.pid_controller.kp_x}, Ki_x={env.pid_controller.ki_x}, Kd_x={env.pid_controller.kd_x}")
    print(f"  Y轴PID参数: Kp_y={env.pid_controller.kp_y}, Ki_y={env.pid_controller.ki_y}, Kd_y={env.pid_controller.kd_y}")
    print(f"  速度限制: [-{env.pid_controller.max_vel}, {env.pid_controller.max_vel}]")
    print(f"  加速度限制: [-{env.pid_controller.max_accel}, {env.pid_controller.max_accel}]")
    
    # 运行仿真
    print(f"\n开始仿真...")
    results = env.run_simulation()
    
    # 显示结果
    print(f"\n仿真完成!")
    print(f"  成功: {'是' if results['success'] else '否'}")
    print(f"  步数: {results['steps']}")
    print(f"  最终距离: {results['final_distance']:.3f}")
    print(f"  总奖励: {results['total_reward']:.2f}")
    
    # 可视化
    env.render(trajectory=results['trajectory'], save_path="./demo_result.png")
    
    return results

if __name__ == "__main__":
    simple_demo()