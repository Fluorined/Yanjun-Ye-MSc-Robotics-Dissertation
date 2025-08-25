# visualize_trained_model.py (完整修复版)

import os
import numpy as np
import matplotlib.pyplot as plt
from rrt_env import RRTTrackingEnv, generate_obstacles, rrt, collision_free
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def is_path_safe(path, obstacles, buffer):
    """检查路径是否安全"""
    if path is None or len(path) < 2:
        return False
    
    for i in range(len(path) - 1):
        if not collision_free(path[i], path[i+1], obstacles, buffer):
            return False
    return True

def visualize_path_with_obstacles(path, obstacles, map_size, start, goal, buffer, seed):
    """可视化路径与障碍物的关系"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='s', s=100, label='Start')
    plt.scatter(goal[0], goal[1], c='red', marker='*', s=200, label='Goal')
    
    # 绘制障碍物
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
        ax.add_patch(circle)
        buffer_circle = plt.Circle((ox, oy), r + buffer, 
                                  color='red', alpha=0.2, fill=False, linestyle='--')
        ax.add_patch(buffer_circle)
    
    # 绘制路径
    if path is not None:
        plt.plot(path[:, 0], path[:, 1], 'b-o', linewidth=1, markersize=4, label='RRT Path')
        
        # 标记碰撞点
        for i in range(len(path) - 1):
            if not collision_free(path[i], path[i+1], obstacles, buffer):
                midpoint = (path[i] + path[i+1]) / 2
                plt.scatter(midpoint[0], midpoint[1], c='purple', s=100, marker='x', label='Collision')
    
    plt.title(f"RRT Path with Obstacles (Seed: {seed})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.grid(True)
    plt.legend()
    
    # 保存路径图
    os.makedirs("./visualization/", exist_ok=True)
    save_path = f"./visualization/rrt_path_seed{seed}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"RRT路径图已保存到: {save_path}")
    plt.close()

def visualize_single_episode(model, vec_env, episode_num, seed):
    """可视化单个回合 - 增强版：显示参考点和跟踪信息"""
    # 获取单一环境用于可视化
    if hasattr(vec_env, 'envs') and len(vec_env.envs) > 0:
        env = vec_env.envs[0]
    else:
        env = vec_env
    
    # 初始化环境
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    # 记录轨迹
    positions = []
    
    # 创建图表
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 绘制起点和终点
    plt.scatter(env.start[0], env.start[1], c='green', marker='s', s=100, label='Start')
    plt.scatter(env.goal[0], env.goal[1], c='red', marker='*', s=200, label='Goal')
    
    # 绘制障碍物
    for (ox, oy, r) in env.obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.7)
        ax.add_patch(circle)
        buffer = plt.Circle((ox, oy), r+env.tracking_buffer, 
                           color='red', alpha=0.2, fill=False, linestyle='--')
        ax.add_patch(buffer)
    
    # 绘制参考轨迹
    plt.plot(env.traj[:, 0], env.traj[:, 1], 'g--', linewidth=2, label='Reference Path')
    
    # 标记所有参考点
    for i, point in enumerate(env.traj):
        plt.scatter(point[0], point[1], c='purple', s=80, marker='o', alpha=0.5)
        # 添加参考点编号
        plt.text(point[0] + 0.1, point[1] + 0.1, f'WP{i}', 
                 fontsize=9, color='purple', alpha=0.7)
    
    # 当前参考点标记
    current_target = env.traj[env.traj_idx]
    target_point = plt.scatter(current_target[0], current_target[1], 
                              c='orange', s=120, marker='s', label='Current Target')
    
    # 设置图表属性
    plt.title(f"Episode {episode_num} - Trajectory Tracking (Seed: {seed})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, env.map_size[0])
    plt.ylim(0, env.map_size[1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 初始位置标记
    agent_point = plt.scatter(env.pos[0], env.pos[1], c='blue', s=50, label='Agent')
    
    # 轨迹线
    trajectory_line, = plt.plot([], [], 'b-', linewidth=1, alpha=0.5)
    
    # 信息文本框
    info_text = plt.text(0.05, 0.95, "", 
                         transform=ax.transAxes, 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 实时更新
    plt.ion()  # 开启交互模式
    plt.show()
    
    while not done:
        # 关键修复：正确处理归一化
        if hasattr(vec_env, 'normalize_obs'):
            # 手动归一化观察值
            obs_normalized = vec_env.normalize_obs(obs)
            action, _ = model.predict(obs_normalized, deterministic=True)
        else:
            # 如果没有归一化，直接使用原始观察值
            action, _ = model.predict(obs, deterministic=True)
        
        # 执行动作（在单一环境中）
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 更新状态
        total_reward += reward
        step_count += 1
        
        # 记录位置
        positions.append(env.pos.copy())
        
        # 更新智能体位置
        agent_point.set_offsets([env.pos])
        
        # 更新轨迹线
        if len(positions) > 1:
            trajectory_line.set_data(
                [p[0] for p in positions],
                [p[1] for p in positions]
            )
        
        # 更新当前目标点
        current_target = env.traj[env.traj_idx]
        target_point.set_offsets([current_target])
        
        # 计算距离并更新信息
        dist_to_target = np.linalg.norm(env.pos - current_target)
        dist_to_goal = np.linalg.norm(env.pos - env.goal)
        
        # 构建信息字符串
        info_str = (
            f"步数: {step_count}\n"
            f"当前参考点: {env.traj_idx}/{len(env.traj)-1}\n"
            f"到目标点距离: {dist_to_target:.2f}\n"
            f"到终点距离: {dist_to_goal:.2f}\n"
            f"累计奖励: {total_reward:.2f}\n"
            f"速度: ({env.vel[0]:.2f}, {env.vel[1]:.2f})"
        )
        info_text.set_text(info_str)
        
        # 更新图表
        plt.draw()
        plt.pause(0.01)  # 控制动画速度
        
        # 检查是否关闭窗口
        if not plt.fignum_exists(1):
            print("可视化窗口已关闭")
            break
    
    # 绘制完整轨迹
    if plt.fignum_exists(1):
        # 添加最终结果文本
        final_text = (
            f"总步数: {step_count}\n"
            f"总奖励: {total_reward:.2f}\n"
            f"终点距离: {dist_to_goal:.2f}\n"
            f"种子: {seed}\n"
            f"状态: {'成功' if dist_to_goal < 0.3 else '失败'}"
        )
        plt.text(0.05, 0.85, final_text, 
                 transform=ax.transAxes, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 保存结果
        save_path = f"./visualization/episode_{episode_num}_seed{seed}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
        
        # 暂停片刻后关闭
        plt.pause(2)
        plt.close()
    
    # 打印结果摘要
    print(f"回合完成! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
    print(f"起点: {env.start}, 终点: {env.goal}")
    print(f"最终位置: {env.pos}, 距离目标: {dist_to_goal:.2f}")
    print(f"路径点完成情况: {env.traj_idx}/{len(env.traj)-1}")

def visualize_trained_model(model_path, num_episodes=3, seed=42):
    """
    加载训练好的模型并在环境中可视化其表现
    
    参数:
        model_path: 模型保存路径 (不包含扩展名)
        num_episodes: 要可视化的回合数
        seed: 随机种子 (必须与训练时相同)
    """
    print(f"使用种子: {seed} 进行可视化")
    
    # 创建环境参数 (应与训练时一致)
    map_size = (10.0, 10.0)
    start = (0.5, 0.5)
    goal = (9.5, 9.5)
    n_obstacles = 8
    obstacle_radius_range = (0.3, 1.0)
    collision_buffer = 0.2
    rrt_buffer = 0.35
    tracking_buffer = 0.15
    
    visualize_trained_model_custom(model_path, num_episodes, seed, start, goal)

def visualize_trained_model_custom(model_path, num_episodes=3, seed=42, start=(0.5, 0.5), goal=(9.5, 9.5)):
    """
    加载训练好的模型并在环境中可视化其表现（支持自定义起终点）
    
    参数:
        model_path: 模型保存路径 (不包含扩展名)
        num_episodes: 要可视化的回合数
        seed: 随机种子
        start: 自定义起点
        goal: 自定义终点
    """
    print(f"使用种子: {seed} 进行可视化")
    print(f"起点: {start}, 终点: {goal}")
    
    # 创建环境参数
    map_size = (10.0, 10.0)
    n_obstacles = 8
    obstacle_radius_range = (0.3, 1.0)
    collision_buffer = 0.2
    rrt_buffer = 0.35
    tracking_buffer = 0.15
    
    # 生成障碍物 (使用相同种子)
    obstacles = generate_obstacles(
        n_obstacles,
        obstacle_radius_range,
        np.array(start),
        np.array(goal),
        np.array(map_size),
        rrt_buffer,
        seed=seed
    )
    
    # 打印障碍物信息
    print(f"使用种子 {seed} 生成的障碍物:")
    for i, (ox, oy, r) in enumerate(obstacles):
        print(f"  障碍物 {i+1}: 位置({ox:.2f}, {oy:.2f}), 半径={r:.2f}")
    
    # 生成RRT路径 (使用相同种子)
    rrt_path = rrt(
        np.array(start),
        np.array(goal),
        obstacles,
        np.array(map_size),
        buffer=rrt_buffer,
        seed=seed
    )
    
    # 验证路径是否安全（使用RRT的buffer进行验证）
    if not is_path_safe(rrt_path, obstacles, rrt_buffer):
        print("警告: RRT路径与障碍物碰撞!")
        visualize_path_with_obstacles(rrt_path, obstacles, map_size, start, goal, rrt_buffer, seed)
    
    # 如果RRT失败，使用直线路径
    if rrt_path is None:
        print("警告: RRT规划失败，使用直线路径")
        rrt_path = np.array([start, goal])
    
    # 创建基础环境
    env = RRTTrackingEnv(
        map_size=map_size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        traj=rrt_path,
        collision_buffer=collision_buffer,
        rrt_buffer=rrt_buffer,
        tracking_buffer=tracking_buffer,
        seed=seed
    )
    
    # 关键修复：加载归一化环境
    # 创建向量化环境包装
    vec_env = DummyVecEnv([lambda: env])
    
    # 确定归一化文件路径
    model_dir = os.path.dirname(model_path)
    vec_norm_candidates = [
        os.path.join(model_dir, f"vec_normalize_seed{seed}.pkl"),  # 训练脚本保存的格式
        model_path + "_vecnormalize.pkl",  # 其他可能格式
        os.path.join(model_dir, "vec_normalize.pkl"),  # 简单格式
    ]
    
    vec_norm_path = None
    for candidate in vec_norm_candidates:
        if os.path.exists(candidate):
            vec_norm_path = candidate
            print(f"找到归一化文件候选: {candidate}")
            break
    
    # 加载归一化参数
    if vec_norm_path and os.path.exists(vec_norm_path):
        print(f"加载环境归一化参数: {vec_norm_path}")
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        
        # 重要设置
        vec_env.training = False  # 禁用训练模式
        vec_env.norm_reward = False  # 不归一化奖励
    else:
        print(f"警告: 未找到归一化文件，使用原始环境")
        # 打印所有候选路径以便调试
        print("检查了以下路径:")
        for candidate in vec_norm_candidates:
            print(f"  - {candidate}")
    
    # 加载模型
    print(f"加载模型: {model_path}.zip")
    try:
        # 最佳方式：加载模型时绑定环境
        model = SAC.load(model_path, env=vec_env)
    except Exception as e:
        print(f"标准加载失败: {e}, 尝试备选方法")
        # 备选方案
        model = SAC.load(model_path)
        model.set_env(vec_env)  # 手动设置环境
    
    # 可视化多个回合（传递向量化环境）
    for episode in range(num_episodes):
        print(f"\n=== 回合 {episode+1}/{num_episodes} ===")
        visualize_single_episode(model, vec_env, episode+1, seed)

if __name__ == "__main__":
    # 配置参数
    MODEL_BASE_PATH = "./results/sac_different_buffer"  # 模型目录
    MODEL_NAME = "final_model"     # 模型基本名称
    NUM_EPISODES = 1
    
    # 完整模型路径
    MODEL_PATH = os.path.join(MODEL_BASE_PATH, MODEL_NAME)
    
    # 询问用户输入种子
    SEED = 8
    user_seed = input(f"请输入随机种子 (默认 {SEED}): ")
    if user_seed.strip():
        try:
            SEED = int(user_seed)
        except ValueError:
            print(f"无效的种子值 '{user_seed}'，使用默认种子 {SEED}")
    
    # 询问用户是否要自定义起点和终点
    DEFAULT_START = (0.5, 0.5)
    DEFAULT_GOAL = (9.5, 9.5)
    
    print(f"\n当前默认起点: {DEFAULT_START}")
    user_start = input("请输入自定义起点 (格式: x,y，留空使用默认值): ")
    start = DEFAULT_START
    if user_start.strip():
        try:
            x, y = map(float, user_start.split(','))
            if 0 <= x <= 10 and 0 <= y <= 10:
                start = (x, y)
                print(f"使用自定义起点: {start}")
            else:
                print("起点坐标超出地图范围 [0,10]，使用默认起点")
        except ValueError:
            print("起点格式无效，使用默认起点")
    
    print(f"\n当前默认终点: {DEFAULT_GOAL}")
    user_goal = input("请输入自定义终点 (格式: x,y，留空使用默认值): ")
    goal = DEFAULT_GOAL
    if user_goal.strip():
        try:
            x, y = map(float, user_goal.split(','))
            if 0 <= x <= 10 and 0 <= y <= 10:
                goal = (x, y)
                print(f"使用自定义终点: {goal}")
            else:
                print("终点坐标超出地图范围 [0,10]，使用默认终点")
        except ValueError:
            print("终点格式无效，使用默认终点")
    
    # 确保可视化目录存在
    os.makedirs("./visualization/", exist_ok=True)
    
    # 运行可视化 (传递种子和自定义起终点)
    visualize_trained_model_custom(MODEL_PATH, NUM_EPISODES, seed=SEED, start=start, goal=goal)

