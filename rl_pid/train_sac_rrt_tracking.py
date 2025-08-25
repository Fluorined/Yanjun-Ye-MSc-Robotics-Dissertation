
# train_sac_rrt_tracking.py (simplified save version)

import os
import time
import numpy as np
import torch
import random
from rl_pid import RRTTrackingEnv, generate_obstacles, rrt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import imageio  # for saving GIF
import matplotlib.pyplot as plt  # for plotting PID parameter charts

# Configuration parameters
CONFIG = {
    "map_size": (10.0, 10.0),
    "start": (0.5, 0.5),
    "goal": (9.5, 9.5),
    "n_obstacles": 8,
    "obstacle_radius_range": (0.3, 1.0),
    "collision_buffer": 0.2,  # maintain backward compatibility
    "rrt_buffer": 0.35,  # buffer used for RRT path generation
    "tracking_buffer": 0.15,  # buffer used for collision detection during RL tracking
    "max_steps": 500,
    "dt": 0.1,
    "total_timesteps": 10_000_000,
    "n_envs": 8,  # number of parallel environments
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
    "base_save_path": "./results/",  # base save path
    "eval_freq": 10000,  # evaluate every N steps
    "eval_episodes": 10,  # number of episodes per evaluation
    "verbose": 1,
    "seed": 8,
    "device": "auto",  # automatically select best device (GPU priority)
    "use_sde": True,  # use state-dependent exploration noise
    "sde_sample_freq": 64,  # SDE sampling frequency
    "save_gif": True,  # whether to save GIF animation
    
    # Random scenario configuration
    "randomize_scenario": True,  # whether to enable random scenarios
    "obstacle_variance": 0.3,  # variance range for obstacle position and size
}

def setup_gpu():
    """Setup GPU device and print information"""
    # Check if GPU is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU device(s):")
        
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use the first GPU
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        return device
    else:
        print("No GPU device found, using CPU")
        return torch.device("cpu")

def create_rrt_env(obstacles=None, traj=None, seed=None, randomize_scenario=None):
    """Create and wrap environment (add seed parameter and random scenario parameter)"""
    if randomize_scenario is None:
        randomize_scenario = CONFIG["randomize_scenario"]
    
    # When using randomization, don't pass fixed start and goal
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
    """Generate consistent obstacle set for training environment (add seed parameter)"""
    # Use temporary start and end points to generate obstacles, these points will be regenerated in random scenarios
    temp_start = np.array(CONFIG["start"])
    temp_goal = np.array(CONFIG["goal"])
    return generate_obstacles(
        CONFIG["n_obstacles"],
        CONFIG["obstacle_radius_range"],
        temp_start,
        temp_goal,
        np.array(CONFIG["map_size"]),
        CONFIG["rrt_buffer"],
        seed=seed  # pass seed
    )

def generate_rrt_path(obstacles, seed=None):
    """Generate RRT path (add seed parameter)"""
    # Only used in fixed scenario mode
    return rrt(
        np.array(CONFIG["start"]),
        np.array(CONFIG["goal"]),
        obstacles,
        np.array(CONFIG["map_size"]),
        buffer=CONFIG["rrt_buffer"],
        seed=seed  # pass seed
    )

def get_next_result_dir(base_path):
    """Get next result directory (sac1, sac2, ...)"""
    os.makedirs(base_path, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Find existing sac directories
    sac_dirs = [d for d in existing_dirs if d.startswith("sac")]
    if not sac_dirs:
        return os.path.join(base_path, "sac1")
    
    # Extract indices and find maximum value
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
    """Setup training environment"""
    print("="*50)
    print("Setting up training environment")
    
    # Set global random seed
    seed = CONFIG["seed"]
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Set environment class-level seed
    RRTTrackingEnv.set_class_seed(seed)
    
    print(f"Using random seed: {seed}")
    print(f"Random scenario mode: {'Enabled' if CONFIG['randomize_scenario'] else 'Disabled'}")
    
    if CONFIG["randomize_scenario"]:
        print(f"Global randomization: start and end points can be generated anywhere within the map")
        print(f"Map size: {CONFIG['map_size']}")
        print(f"Obstacle variance: {CONFIG['obstacle_variance']}")
    else:
        print(f"Fixed start: {CONFIG['start']}")
        print(f"Fixed goal: {CONFIG['goal']}")
        
    if CONFIG["randomize_scenario"]:
        # In random scenario mode, generate base obstacle configuration
        obstacles = generate_training_obstacles(seed=seed)
        print(f"Generated base obstacle template: {len(obstacles)} obstacles")
        
        # Don't pre-generate path, let environment generate it
        rrt_path = None
        print("Random scenario mode: path will be generated dynamically on each reset")
        print("Note: start and end points will be randomly generated globally on each reset")
    else:
        # Fixed scenario mode
        obstacles = generate_training_obstacles(seed=seed)
        print(f"Generated {len(obstacles)} obstacles")
        
        # Generate RRT path
        rrt_path = generate_rrt_path(obstacles, seed=seed)
        if rrt_path is None:
            print("Warning: RRT planning failed, using straight line path")
            rrt_path = np.array([CONFIG["start"], CONFIG["goal"]])
        print(f"RRT path contains {len(rrt_path)} path points")
    
    # Create training environment function
    def make_env():
        # Each environment uses the same seed
        env = create_rrt_env(obstacles=obstacles, traj=rrt_path, seed=seed, randomize_scenario=CONFIG["randomize_scenario"])
        return Monitor(env)
    
    # Create evaluation environment function (for evaluation callback during training)
    def make_eval_env():
        # Use fixed scenario during evaluation for reproducibility
        if CONFIG["randomize_scenario"] and rrt_path is None:
            # Create fixed evaluation scenario for random training
            eval_rrt_path = generate_rrt_path(obstacles, seed=seed)
            if eval_rrt_path is None:
                eval_rrt_path = np.array([CONFIG["start"], CONFIG["goal"]])
        else:
            eval_rrt_path = rrt_path
        
        env = create_rrt_env(obstacles=obstacles, traj=eval_rrt_path, seed=seed, randomize_scenario=False)
        return Monitor(env)
    
    # Create parallel environment
    vec_env = make_vec_env(
        make_env,
        n_envs=CONFIG["n_envs"],
        vec_env_cls=DummyVecEnv,
        seed=seed  # pass seed
    )
    
    # Create evaluation environment (for evaluation callback during training)
    eval_vec_env = make_vec_env(
        make_eval_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        seed=seed
    )
    
    # Add environment normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
    
    return vec_env, eval_vec_env, obstacles, rrt_path, seed  # return evaluation environment

def create_model(env):
    """Create SAC model - PID control version"""
    # Action noise (for PID parameter adjustment)
    n_actions = env.action_space.shape[-1]  # now 6-dimensional
    print(f"PID control action space dimension: {n_actions}")
    
    # Smaller noise, because PID parameter variation should not be too large
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.05 * np.ones(n_actions)  # smaller noise
    )
    
    # Pass network architecture parameters when creating model
    policy_kwargs = CONFIG.get("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        # Convert string to actual activation function
        import torch.nn as nn
        if policy_kwargs["activation_fn"] == "relu":
            policy_kwargs["activation_fn"] = nn.ReLU
        elif policy_kwargs["activation_fn"] == "tanh":
            policy_kwargs["activation_fn"] = nn.Tanh
    
    # Create model
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
        device=CONFIG["device"],
        seed=CONFIG["seed"],
        use_sde=CONFIG["use_sde"],
        sde_sample_freq=CONFIG["sde_sample_freq"],
        policy_kwargs=policy_kwargs  # pass network architecture parameters
    )
    
    print(f"SAC model creation completed:")
    print(f"  State space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Network architecture: {policy_kwargs.get('net_arch', 'default')}")
    
    return model

def setup_callbacks(eval_env, save_path):
    """Setup callback functions (simplified version)"""
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(CONFIG["tensorboard_log"], exist_ok=True)
    
    # Only keep evaluation callback, only save best model
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
    
    # Progress bar callback
    progress_callback = ProgressBarCallback()
    
    return [eval_callback, progress_callback]

def log_pid_parameters(action, env, step_count=None):
    """记录和分析PID参数"""
    if not CONFIG.get("log_pid_params", False):
        return
    
    # 将动作映射为PID参数
    pid_params = env._map_action_to_pid_params(action)
    Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y = pid_params
    
    if step_count is not None and step_count % 1000 == 0:
        print(f"Step {step_count} - PID参数:")
        print(f"  X轴: Kp={Kp_x:.3f}, Ki={Ki_x:.3f}, Kd={Kd_x:.3f}")
        print(f"  Y轴: Kp={Kp_y:.3f}, Ki={Ki_y:.3f}, Kd={Kd_y:.3f}")
    
    return pid_params

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

def analyze_learned_pid_params(model, env, n_samples=100):
    """分析学到的PID参数分布"""
    print("\n=== 分析学到的PID参数 ===")
    
    pid_params_history = []
    
    # 采样多个状态下的PID参数
    for _ in range(n_samples):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        pid_params = env._map_action_to_pid_params(action)
        pid_params_history.append(pid_params)
    
    pid_params_array = np.array(pid_params_history)
    
    # 统计分析
    param_names = ['Kp_x', 'Ki_x', 'Kd_x', 'Kp_y', 'Ki_y', 'Kd_y']
    print(f"\nPID参数统计 (基于{n_samples}个采样):")
    for i, name in enumerate(param_names):
        mean_val = np.mean(pid_params_array[:, i])
        std_val = np.std(pid_params_array[:, i])
        min_val = np.min(pid_params_array[:, i])
        max_val = np.max(pid_params_array[:, i])
        print(f"  {name}: {mean_val:.3f} ± {std_val:.3f} [{min_val:.3f}, {max_val:.3f}]")
    
    return pid_params_array, param_names

def evaluate_model(model, obstacles, rrt_path, seed, save_path):
    """评估模型性能并保存结果 - PID版本"""
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
        # Use fixed scenario during evaluation for reproducibility
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
    
    # 分析学到的PID参数
    pid_params_array, param_names = analyze_learned_pid_params(model, render_env, n_samples=50)
    
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
        f.write(f"\nPID参数统计:\n")
        for i, name in enumerate(param_names):
            mean_val = np.mean(pid_params_array[:, i])
            std_val = np.std(pid_params_array[:, i])
            f.write(f"  {name}: {mean_val:.3f} ± {std_val:.3f}\n")
    
    print(f"结果摘要已保存到: {results_path}")
    
    # 保存PID参数分析图
    if CONFIG.get("log_pid_params", False):
        plt.figure(figsize=(12, 8))
        for i, name in enumerate(param_names):
            plt.subplot(2, 3, i+1)
            plt.hist(pid_params_array[:, i], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{name} Distribution')
            plt.xlabel('Parameter Value')
            plt.ylabel('Frequency')
            
            # 添加垂直线显示平均值
            mean_val = np.mean(pid_params_array[:, i])
            plt.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
            plt.legend()
        
        plt.tight_layout()
        pid_analysis_path = os.path.join(save_path, "pid_parameters_analysis.png")
        plt.savefig(pid_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PID参数分析图已保存到: {pid_analysis_path}")
    
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
    
    # Create model
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
    print(f"  - PID参数分析: {result_dir}/pid_parameters_analysis.png")

if __name__ == "__main__":
    main()

