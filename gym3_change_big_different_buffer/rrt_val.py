
# rrt_val.py - RRT 路径可视化验证脚本（带种子控制）

import numpy as np
import matplotlib.pyplot as plt
from rrt_env import generate_obstacles, rrt
import random  # 添加随机模块

def visualize_rrt_path(map_size=(10.0, 10.0),
                      start=(0.5, 0.5),
                      goal=(9.5, 9.5),
                      n_obstacles=8,
                      obstacle_radius_range=(0.3, 1.0),
                      collision_buffer=0.2,  # 保持向后兼容
                      rrt_buffer=0.35,  # RRT路径生成使用的buffer（更大）
                      tracking_buffer=0.15,  # 显示时使用的buffer（更小）
                      step_size=0.5,
                      max_iter=500,
                      seed=21,  # 添加种子参数
                      save_fig=False):
    """
    可视化验证 RRT 路径生成过程（带种子控制和双buffer机制）
    
    参数:
        map_size: 地图尺寸 (宽, 高)
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        n_obstacles: 障碍物数量
        obstacle_radius_range: 障碍物半径范围 (最小, 最大)
        collision_buffer: 安全缓冲区（保持向后兼容）
        rrt_buffer: RRT路径生成使用的buffer（更大的安全距离）
        tracking_buffer: 显示时使用的buffer（更小的安全距离）
        step_size: RRT 步长
        max_iter: RRT 最大迭代次数
        seed: 随机种子，确保可重现性
        save_fig: 是否保存图像
    """
    # 设置随机种子（确保可重现性）
    np.random.seed(seed)
    random.seed(seed)
    
    # 转换为 numpy 数组
    map_size = np.array(map_size)
    start = np.array(start)
    goal = np.array(goal)
    
    print("="*50)
    print("RRT 路径可视化验证（带种子控制和双buffer机制）")
    print(f"使用种子: {seed}")
    print(f"地图尺寸: {map_size}")
    print(f"起点: {start}, 终点: {goal}")
    print(f"障碍物: {n_obstacles}个, 半径范围: {obstacle_radius_range}")
    print(f"RRT缓冲区: {rrt_buffer}, 显示缓冲区: {tracking_buffer}, RRT步长: {step_size}, 最大迭代: {max_iter}")
    print("="*50)
    
    # 生成障碍物（传递种子）- 使用rrt_buffer生成障碍物
    obstacles = generate_obstacles(
        n_obstacles,
        obstacle_radius_range,
        start,
        goal,
        map_size,
        rrt_buffer,  # 使用更大的buffer生成障碍物
        seed=seed  # 传递种子
    )
    print(f"生成 {len(obstacles)} 个障碍物")
    
    # 运行 RRT 算法（传递种子）- 使用rrt_buffer进行路径规划
    print("运行 RRT 算法...")
    path = rrt(
        start,
        goal,
        obstacles,
        map_size,
        step_size=step_size,
        buffer=rrt_buffer,  # 使用更大的buffer进行路径规划
        max_iter=max_iter,
        seed=seed  # 传递种子
    )
    
    # 可视化
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='s', s=100, label='Start')
    plt.scatter(goal[0], goal[1], c='red', marker='*', s=200, label='Goal')
    
    # 绘制障碍物（显示双buffer）
    for i, (ox, oy, r) in enumerate(obstacles):
        # 障碍物本体
        obstacle_label = 'Obstacles' if i == 0 else None
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7, label=obstacle_label)
        
        # RRT规划buffer（大）
        rrt_label = f'RRT Buffer ({rrt_buffer})' if i == 0 else None
        rrt_buffer_circle = plt.Circle((ox, oy), r + rrt_buffer, 
                                     color='orange', alpha=0.15, fill=False, linestyle='-', linewidth=2, label=rrt_label)
        
        # 跟踪buffer（小）
        tracking_label = f'Tracking Buffer ({tracking_buffer})' if i == 0 else None
        tracking_buffer_circle = plt.Circle((ox, oy), r + tracking_buffer, 
                                          color='red', alpha=0.3, fill=False, linestyle='--', linewidth=1, label=tracking_label)
        
        ax.add_patch(circle)
        ax.add_patch(rrt_buffer_circle)
        ax.add_patch(tracking_buffer_circle)
        plt.text(ox, oy, f"{i+1}", ha='center', va='center', color='white', fontsize=8)
    
    # 绘制路径（如果找到）
    if path is not None:
        plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, label='RRT Path')
        plt.plot(path[:, 0], path[:, 1], 'go', markersize=4, alpha=0.5)
        plt.scatter(path[0, 0], path[0, 1], c='green', marker='s', s=100)  # 起点
        plt.scatter(path[-1, 0], path[-1, 1], c='red', marker='*', s=200)  # 终点
        
        # 标注路径点
        for i, point in enumerate(path):
            plt.text(point[0], point[1], f"{i}", 
                     ha='center', va='center', 
                     fontsize=8, color='blue',
                     bbox=dict(boxstyle="round,pad=0.3", 
                               fc="yellow", ec="b", lw=1, alpha=0.3))
        
        print(f"成功生成路径! 路径点数量: {len(path)}")
        print("路径点坐标:")
        for i, point in enumerate(path):
            print(f"  {i}: ({point[0]:.2f}, {point[1]:.2f})")
    else:
        print("警告: RRT 未能找到路径!")
    
    # 设置图形属性（添加种子信息和双buffer信息）
    plt.title(f"RRT Path Validation (Seed: {seed})\nObstacles: {n_obstacles}, RRT Buffer: {rrt_buffer}, Track Buffer: {tracking_buffer}, Step: {step_size}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.axis('equal')
    
    # 添加信息框（包含种子信息和双buffer信息）
    info_text = (
        f"Seed: {seed}\n"
        f"Start: ({start[0]}, {start[1]})\n"
        f"Goal: ({goal[0]}, {goal[1]})\n"
        f"Obstacles: {n_obstacles}\n"
        f"RRT Buffer: {rrt_buffer}\n"
        f"Track Buffer: {tracking_buffer}\n"
        f"Step Size: {step_size}\n"
        f"Max Iter: {max_iter}\n"
        f"Path Found: {'Yes' if path is not None else 'No'}"
    )
    plt.gcf().text(0.02, 0.98, info_text, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_fig:
        filename = f"rrt_val_seed{seed}_{n_obstacles}obs_rrt{rrt_buffer}_track{tracking_buffer}_{step_size}step.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图像已保存为: {filename}")
    
    plt.tight_layout()
    plt.show()

def run_multiple_tests(seed=42):
    """运行多个测试用例（带种子控制）"""
    test_cases = [
        # 简单场景
        {
            'n_obstacles': 0,
            'rrt_buffer': 0.35,
            'tracking_buffer': 0.15,
            'step_size': 0.5
        },
        # 中等障碍物
        {
            'n_obstacles': 5,
            'rrt_buffer': 0.35,
            'tracking_buffer': 0.15,
            'step_size': 0.5
        },
        # 复杂场景
        {
            'n_obstacles': 15,
            'rrt_buffer': 0.35,
            'tracking_buffer': 0.15,
            'step_size': 0.5
        },
        # 小跟踪缓冲区
        {
            'n_obstacles': 8,
            'rrt_buffer': 0.35,
            'tracking_buffer': 0.1,
            'step_size': 0.5
        },
        # 大RRT缓冲区
        {
            'n_obstacles': 8,
            'rrt_buffer': 0.5,
            'tracking_buffer': 0.15,
            'step_size': 0.5
        },
        # 挑战性场景
        {
            'n_obstacles': 20,
            'rrt_buffer': 0.4,
            'tracking_buffer': 0.2,
            'step_size': 0.3
        }
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"运行测试用例 #{i+1} (种子: {seed})")
        print(f"参数: {params}")
        
        visualize_rrt_path(
            n_obstacles=params['n_obstacles'],
            rrt_buffer=params['rrt_buffer'],
            tracking_buffer=params['tracking_buffer'],
            step_size=params['step_size'],
            seed=seed,  # 传递种子
            save_fig=True
        )

if __name__ == "__main__":
    # 设置默认种子
    SEED = 42
    
    # 询问用户是否要使用特定种子
    user_seed = input(f"请输入随机种子 (默认 {SEED}): ")
    if user_seed.strip():
        try:
            SEED = int(user_seed)
        except ValueError:
            print(f"无效的种子值 '{user_seed}'，使用默认种子 {SEED}")
    
    # 询问用户要运行单个测试还是多个测试
    test_type = input("运行测试类型: (1) 单个测试 (2) 多个测试 (默认 1): ") or "1"
    
    if test_type == "2":
        # 运行多个测试用例
        run_multiple_tests(seed=SEED)
    else:
        # 运行单个测试
        visualize_rrt_path(
            n_obstacles=8,
            rrt_buffer=0.35,
            tracking_buffer=0.15,
            step_size=0.5,
            seed=SEED,  # 传递种子
            save_fig=True
        )
    
    # 或者运行多个测试用例
    # run_multiple_tests()