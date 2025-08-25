
# train_sac_rrt_tracking.py (简化保存版本)

import os
import time
import numpy as np
import torch
import random
from rrt_env import RRTTrackingEnv, generate_obstacles, rrt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import imageio  # 用于保存GIF

# 配置参数
CONFIG = {
    "map_size": (10.0, 10.0),
    "start": (0.5, 0.5),
    "goal": (9.5, 9.5),
    "n_obstacles": 8,
    "obstacle_radius_range": (0.3, 1.0),
    "collision_buffer": 0.2,  # 保持向后兼容
    "rrt_buffer": 0.35,  # RRT路径生成使用的buffer
    "tracking_buffer": 0.15,  # RL跟踪时碰撞检测使用的buffer
    "max_steps": 500,
    "dt": 0.1,
    "total_timesteps": 10_000_000,
    "n_envs": 8,  # 并行环境数量
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "buffer_size": 1_000_000,
    "learning_starts": 10000,
    "batch_size": 512,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "train_freq": (1, "step"),
    "gradient_steps": 1,
    "tensorboard_log": "./sac_rrt_tensorboard/",
    "base_save_path": "./results/",  # 基础保存路径
    "eval_freq": 10000,  # 每隔多少步评估一次
    "eval_episodes": 10,  # 每次评估的回合数
    "verbose": 1,
    "seed": 8,
    "device": "auto",  # 自动选择最佳设备 (GPU优先)
    "use_sde": True,  # 使用状态依赖的探索噪声
    "sde_sample_freq": 64,  # SDE采样频率
    "save_gif": True,  # 是否保存GIF动画
    
    # 随机场景配置
    "randomize_scenario": True,  # 是否启用随机场景
    "obstacle_variance": 0.3,  # 障碍物位置和大小的变化范围
}

def setup_gpu():
    """设置GPU设备并打印信息"""
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个GPU设备:")
        
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 使用第一个GPU
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
        return device
    else:
        print("未发现GPU设备，使用CPU")
        return torch.device("cpu")

def create_rrt_env(obstacles=None, traj=None, seed=None, randomize_scenario=None):
    """创建并包装环境（添加种子参数和随机场景参数）"""
    if randomize_scenario is None:
        randomize_scenario = CONFIG["randomize_scenario"]
    
    # 当使用随机化时，不传递固定的start和goal
    if randomize_scenario:
        env = RRTTrackingEnv(
            map_size=CONFIG["map_size"],
            n_obstacles=CONFIG["n_obstacles"],
            obstacle_radius_range=CONFIG["obstacle_radius_range"],
            collision_buffer=CONFIG["collision_buffer"],
            rrt_buffer=CONFIG["rrt_buffer"],
            tracking_buffer=CONFIG["tracking_buffer"],
            max_steps=CONFIG["max_steps"],
            dt=CONFIG["dt"],
            obstacles=obstacles,
            traj=traj,
            auto_generate_rrt=True,
            seed=seed,
            randomize_scenario=randomize_scenario,
            obstacle_variance=CONFIG["obstacle_variance"]
        )
    else:
        env = RRTTrackingEnv(
            map_size=CONFIG["map_size"],
            start=CONFIG["start"],
            goal=CONFIG["goal"],
            n_obstacles=CONFIG["n_obstacles"],
            obstacle_radius_range=CONFIG["obstacle_radius_range"],
            collision_buffer=CONFIG["collision_buffer"],
            rrt_buffer=CONFIG["rrt_buffer"],
            tracking_buffer=CONFIG["tracking_buffer"],
            max_steps=CONFIG["max_steps"],
            dt=CONFIG["dt"],
            obstacles=obstacles,
            traj=traj,
            auto_generate_rrt=True,
            seed=seed,
            randomize_scenario=randomize_scenario,
            obstacle_variance=CONFIG["obstacle_variance"]
        )
    return env

def generate_training_obstacles(seed=None):
    """为训练环境生成一致的障碍物集合（添加种子参数）"""
    # 使用临时的起终点生成障碍物，随机化场景时这些点会被重新生成
    temp_start = np.array(CONFIG["start"])
    temp_goal = np.array(CONFIG["goal"])
    return generate_obstacles(
        CONFIG["n_obstacles"],
        CONFIG["obstacle_radius_range"],
        temp_start,
        temp_goal,
        np.array(CONFIG["map_size"]),
        CONFIG["rrt_buffer"],
        seed=seed  # 传递种子
    )

def generate_rrt_path(obstacles, seed=None):
    """生成RRT路径（添加种子参数）"""
    # 仅在固定场景模式下使用
    return rrt(
        np.array(CONFIG["start"]),
        np.array(CONFIG["goal"]),
        obstacles,
        np.array(CONFIG["map_size"]),
        buffer=CONFIG["rrt_buffer"],
        seed=seed  # 传递种子
    )

def get_next_result_dir(base_path):
    """获取下一个结果目录（sac1, sac2, ...）"""
    os.makedirs(base_path, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # 查找已有的sac目录
    sac_dirs = [d for d in existing_dirs if d.startswith("sac")]
    if not sac_dirs:
        return os.path.join(base_path, "sac1")
    
    # 提取序号并找到最大值
    indices = []
    for d in sac_dirs:
        try:
            idx = int(d[3:])
            indices.append(idx)
        except ValueError:
            continue
    
    next_idx = max(indices) + 1 if indices else 1
    return os.path.join(base_path, f"sac{next_idx}")

def setup_environment():
    """设置训练环境"""
    print("="*50)
    print("设置训练环境")
    
    # 设置全局随机种子
    seed = CONFIG["seed"]
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置环境类级种子
    RRTTrackingEnv.set_class_seed(seed)
    
    print(f"使用随机种子: {seed}")
    print(f"随机场景模式: {'启用' if CONFIG['randomize_scenario'] else '禁用'}")
    
    if CONFIG["randomize_scenario"]:
        print(f"全局随机化: 起点和终点可在整个地图范围内生成")
        print(f"地图大小: {CONFIG['map_size']}")
        print(f"障碍物变化范围: {CONFIG['obstacle_variance']}")
    else:
        print(f"固定起点: {CONFIG['start']}")
        print(f"固定终点: {CONFIG['goal']}")
        
    if CONFIG["randomize_scenario"]:
        # 在随机场景模式下，生成基础障碍物配置
        obstacles = generate_training_obstacles(seed=seed)
        print(f"生成基础障碍物模板: {len(obstacles)} 个")
        
        # 不预先生成路径，让环境自己生成
        rrt_path = None
        print("随机场景模式：路径将在每次重置时动态生成")
        print("注意：起点和终点将在每次重置时全局随机生成")
    else:
        # 固定场景模式
        obstacles = generate_training_obstacles(seed=seed)
        print(f"生成 {len(obstacles)} 个障碍物")
        
        # 生成RRT路径
        rrt_path = generate_rrt_path(obstacles, seed=seed)
        if rrt_path is None:
            print("警告：RRT规划失败，使用直线路径")
            rrt_path = np.array([CONFIG["start"], CONFIG["goal"]])
        print(f"RRT路径包含 {len(rrt_path)} 个路径点")
    
    # 创建训练环境函数
    def make_env():
        # 每个环境使用相同的种子
        env = create_rrt_env(obstacles=obstacles, traj=rrt_path, seed=seed, randomize_scenario=CONFIG["randomize_scenario"])
        return Monitor(env)
    
    # 创建评估环境函数（用于训练期间的评估回调）
    def make_eval_env():
        # 评估时使用固定场景以确保可重现性
        if CONFIG["randomize_scenario"] and rrt_path is None:
            # 为随机训练创建固定的评估场景
            eval_rrt_path = generate_rrt_path(obstacles, seed=seed)
            if eval_rrt_path is None:
                eval_rrt_path = np.array([CONFIG["start"], CONFIG["goal"]])
        else:
            eval_rrt_path = rrt_path
        
        env = create_rrt_env(obstacles=obstacles, traj=eval_rrt_path, seed=seed, randomize_scenario=False)
        return Monitor(env)
    
    # 创建并行环境
    vec_env = make_vec_env(
        make_env,
        n_envs=CONFIG["n_envs"],
        vec_env_cls=DummyVecEnv,
        seed=seed  # 传递种子
    )
    
    # 创建评估环境（用于训练期间的评估回调）
    eval_vec_env = make_vec_env(
        make_eval_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        seed=seed
    )
    
    # 添加环境标准化
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
    
    return vec_env, eval_vec_env, obstacles, rrt_path, seed  # 返回评估环境

def create_model(env):
    """创建SAC模型"""
    # 动作噪声（可选）
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    
    # 创建模型
    model = SAC(
        CONFIG["policy"],
        env,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        tau=CONFIG["tau"],
        gamma=CONFIG["gamma"],
        ent_coef=CONFIG["ent_coef"],
        action_noise=action_noise,
        train_freq=CONFIG["train_freq"],
        gradient_steps=CONFIG["gradient_steps"],
        tensorboard_log=CONFIG["tensorboard_log"],
        verbose=CONFIG["verbose"],
        device=CONFIG["device"],  # 设置设备
        seed=CONFIG["seed"],  # 设置模型种子
        use_sde=CONFIG["use_sde"],
        sde_sample_freq=CONFIG["sde_sample_freq"]
    )
    
    return model

def setup_callbacks(eval_env, save_path):
    """设置回调函数（简化版本）"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(CONFIG["tensorboard_log"], exist_ok=True)
    
    # 只保留评估回调，只保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=CONFIG["eval_freq"] // CONFIG["n_envs"],
        n_eval_episodes=CONFIG["eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 进度条回调
    progress_callback = ProgressBarCallback()
    
    return [eval_callback, progress_callback]

def train_model(model, callbacks, save_path):
    """训练模型（简化保存）"""
    print("="*50)
    print("开始训练SAC模型")
    print(f"总时间步数: {CONFIG['total_timesteps']}")
    print(f"并行环境: {CONFIG['n_envs']}")
    print(f"Tensorboard日志: {CONFIG['tensorboard_log']}")
    print(f"模型保存路径: {save_path}")
    print(f"使用设备: {model.device}")
    print(f"随机种子: {CONFIG['seed']}")
    
    start_time = time.time()
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=callbacks,
        log_interval=4,
        tb_log_name="sac_rrt_training"
    )
    
    training_time = time.time() - start_time
    print(f"训练完成! 耗时: {training_time:.2f} 秒")
    
    # 只保存最终模型和环境标准化参数
    model_save_path = os.path.join(save_path, "final_model")
    model.save(model_save_path)
    print(f"最终模型已保存到: {model_save_path}")
    
    # 保存环境标准化参数
    env_normalize_path = os.path.join(save_path, "vec_normalize.pkl")
    model.get_vec_normalize_env().save(env_normalize_path)
    print(f"环境标准化参数已保存到: {env_normalize_path}")
    
    return model

def evaluate_model(model, obstacles, rrt_path, seed, save_path):
    """评估模型性能并保存结果"""
    print("="*50)
    print("评估模型性能")
    
    # 在随机场景模式下，为评估创建固定场景
    eval_obstacles = obstacles
    eval_rrt_path = rrt_path
    
    if CONFIG["randomize_scenario"] and rrt_path is None:
        # 在随机场景模式下，生成一个固定的评估场景
        eval_rrt_path = generate_rrt_path(obstacles, seed=seed)
        if eval_rrt_path is None:
            print("警告：评估用RRT规划失败，使用直线路径")
            eval_rrt_path = np.array([CONFIG["start"], CONFIG["goal"]])
        print(f"评估用RRT路径包含 {len(eval_rrt_path)} 个路径点")
    
    # 创建评估环境函数（与训练时保持一致）
    def make_eval_env():
        # 评估时使用固定场景以确保可重现性
        env = create_rrt_env(obstacles=eval_obstacles, traj=eval_rrt_path, seed=seed, randomize_scenario=False)
        return Monitor(env)
    
    # 创建向量化环境
    eval_vec_env = make_vec_env(
        make_eval_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        seed=seed
    )
    
    # 加载训练时的VecNormalize参数
    vec_normalize_path = os.path.join(save_path, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"加载VecNormalize参数: {vec_normalize_path}")
        eval_vec_env = VecNormalize.load(vec_normalize_path, eval_vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
    else:
        print("警告：未找到VecNormalize参数文件")
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_vec_env,
        n_eval_episodes=10,
        deterministic=True
    )
    
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 渲染一个示例回合
    print("渲染一个示例回合...")
    render_env = create_rrt_env(obstacles=eval_obstacles, traj=eval_rrt_path, seed=seed, randomize_scenario=False)
    
    obs, _ = render_env.reset()
    done = False
    frames = []
    step_count = 0
    total_reward = 0
    reached_points = 0
    
    # 标准化观察值
    if os.path.exists(vec_normalize_path):
        obs_normalized = eval_vec_env.normalize_obs(obs)
    else:
        obs_normalized = obs
    
    while not done and step_count < CONFIG["max_steps"]:
        # 使用标准化的观察值进行预测
        action, _ = model.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        
        # 标准化新的观察值
        if os.path.exists(vec_normalize_path):
            obs_normalized = eval_vec_env.normalize_obs(obs)
        else:
            obs_normalized = obs
        
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        reached_points = max(reached_points, render_env.traj_idx)
        
        # 渲染并获取当前帧
        frame = render_env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    
    print(f"示例回合结果: 总奖励={total_reward:.2f}, 步数={step_count}, 到达路径点={reached_points}/{len(eval_rrt_path)}")
    
    # 保存结果文件
    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "total_reward": total_reward,
        "steps": step_count,
        "reached_points": reached_points,
        "total_points": len(eval_rrt_path),
        "seed": seed,
        "config": CONFIG
    }
    
    # 保存结果到txt文件
    results_path = os.path.join(save_path, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"训练结果摘要\n")
        f.write(f"="*30 + "\n")
        f.write(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"示例回合奖励: {total_reward:.2f}\n")
        f.write(f"示例回合步数: {step_count}\n")
        f.write(f"到达路径点: {reached_points}/{len(eval_rrt_path)}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"训练步数: {CONFIG['total_timesteps']}\n")
        f.write(f"随机场景模式: {'启用' if CONFIG['randomize_scenario'] else '禁用'}\n")
    
    print(f"结果摘要已保存到: {results_path}")
    
    # 只保存GIF动画（如果启用）
    if CONFIG["save_gif"] and frames:
        gif_path = os.path.join(save_path, "tracking_demo.gif")
        imageio.mimsave(gif_path, frames, duration=0.1)
        print(f"演示动画已保存到: {gif_path}")
    
    # 保存最终状态图
    if render_env.fig is not None:
        plt_path = os.path.join(save_path, "final_state.png")
        render_env.fig.savefig(plt_path)
        print(f"最终状态图已保存到: {plt_path}")
    
    # 关闭环境
    render_env.close()
    eval_vec_env.close()
    
    return mean_reward, std_reward

def start_tensorboard():
    """启动Tensorboard"""
    import threading
    import subprocess
    
    def run_tensorboard():
        command = f"tensorboard --logdir={CONFIG['tensorboard_log']} --port=6006"
        subprocess.run(command, shell=True)
    
    # 在后台线程中启动Tensorboard
    tensorboard_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tensorboard_thread.start()
    print(f"Tensorboard 已启动: http://localhost:6006")

def main():
    # 设置GPU设备
    device = setup_gpu()
    CONFIG["device"] = device
    
    # 启动Tensorboard
    start_tensorboard()
    
    # 获取下一个结果目录
    result_dir = get_next_result_dir(CONFIG["base_save_path"])
    print(f"本次训练结果将保存到: {result_dir}")
    
    # 设置环境
    vec_env, eval_vec_env, obstacles, rrt_path, seed = setup_environment()
    
    # 创建模型
    model = create_model(vec_env)
    
    # 设置回调
    callbacks = setup_callbacks(eval_vec_env, result_dir)
    
    # 训练模型
    trained_model = train_model(model, callbacks, result_dir)
    
    # 评估模型
    evaluate_model(trained_model, obstacles, rrt_path, seed, result_dir)
    
    # 关闭环境
    vec_env.close()
    eval_vec_env.close()
    
    print("="*50)
    print("训练完成！")
    print(f"保存的文件:")
    print(f"  - 最终模型: {result_dir}/final_model.zip")
    print(f"  - 最佳模型: {result_dir}/best_model.zip")
    print(f"  - 环境标准化: {result_dir}/vec_normalize.pkl")
    print(f"  - 结果摘要: {result_dir}/results.txt")
    print(f"  - 演示动画: {result_dir}/tracking_demo.gif")
    print(f"  - 最终状态图: {result_dir}/final_state.png")

if __name__ == "__main__":
    main()

