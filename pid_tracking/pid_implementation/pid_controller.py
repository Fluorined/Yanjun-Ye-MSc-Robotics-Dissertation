# pid_controller.py - PID控制器实现

import numpy as np

class PIDController:
    """双PID控制器，分别控制ax和ay"""
    
    def __init__(self, kp_x=1.0, ki_x=0.1, kd_x=0.5, 
                 kp_y=1.0, ki_y=0.1, kd_y=0.5, 
                 dt=0.1, max_vel=5.0, max_accel=1.0):
        """
        初始化双PID控制器
        
        参数:
            kp_x, ki_x, kd_x: X轴PID增益
            kp_y, ki_y, kd_y: Y轴PID增益
            dt: 采样时间
            max_vel: 速度限制 [-max_vel, max_vel]
            max_accel: 加速度限制 [-max_accel, max_accel]
        """
        # X轴PID参数
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        
        # Y轴PID参数
        self.kp_y = kp_y
        self.ki_y = ki_y
        self.kd_y = kd_y
        
        self.dt = dt
        self.max_vel = max_vel
        self.max_accel = max_accel
        
        # 状态变量
        self.reset()
    
    def reset(self):
        """重置PID控制器状态"""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
    def compute(self, current_pos, target_pos, current_vel=None):
        """
        计算双PID控制输出
        
        参数:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            current_vel: 当前速度 [vx, vy] (可选，用于速度前馈)
            
        返回:
            control_output: 控制输出 [ax, ay] (加速度)
        """
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 计算位置误差
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        
        # X轴PID计算
        # 比例项
        p_term_x = self.kp_x * error_x
        
        # 积分项 (防止积分饱和)
        self.integral_x += error_x * self.dt
        max_integral_x = self.max_accel / (self.ki_x + 1e-8)
        self.integral_x = np.clip(self.integral_x, -max_integral_x, max_integral_x)
        i_term_x = self.ki_x * self.integral_x
        
        # 微分项
        error_derivative_x = (error_x - self.prev_error_x) / self.dt
        d_term_x = self.kd_x * error_derivative_x
        
        # X轴PID输出
        ax = p_term_x + i_term_x + d_term_x
        
        # Y轴PID计算
        # 比例项
        p_term_y = self.kp_y * error_y
        
        # 积分项 (防止积分饱和)
        self.integral_y += error_y * self.dt
        max_integral_y = self.max_accel / (self.ki_y + 1e-8)
        self.integral_y = np.clip(self.integral_y, -max_integral_y, max_integral_y)
        i_term_y = self.ki_y * self.integral_y
        
        # 微分项
        error_derivative_y = (error_y - self.prev_error_y) / self.dt
        d_term_y = self.kd_y * error_derivative_y
        
        # Y轴PID输出
        ay = p_term_y + i_term_y + d_term_y
        
        # 加速度限制
        ax = np.clip(ax, -self.max_accel, self.max_accel)
        ay = np.clip(ay, -self.max_accel, self.max_accel)
        
        # 更新历史误差
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        return np.array([ax, ay])
    

class PIDPathTracker:
    """基于PID的路径跟踪器"""
    
    def __init__(self, path, pid_controller, lookahead_distance=0.5):
        """
        初始化路径跟踪器
        
        参数:
            path: 路径点序列 [[x1,y1], [x2,y2], ...]
            pid_controller: PID控制器实例
            lookahead_distance: 前瞻距离
        """
        self.path = np.array(path)
        self.pid_controller = pid_controller
        self.lookahead_distance = lookahead_distance
        self.current_waypoint_idx = 0
        
    def get_target_point(self, current_pos):
        """获取当前目标点"""
        if self.current_waypoint_idx >= len(self.path):
            return self.path[-1]  # 返回最后一个点
        
        current_target = self.path[self.current_waypoint_idx]
        
        # 检查是否需要切换到下一个路径点
        distance_to_current = np.linalg.norm(np.array(current_pos) - current_target)
        
        if distance_to_current < self.lookahead_distance and self.current_waypoint_idx < len(self.path) - 1:
            self.current_waypoint_idx += 1
            return self.path[self.current_waypoint_idx]
        
        return current_target
    
    def control_step(self, current_pos, current_vel=None):
        """执行一步控制"""
        target_pos = self.get_target_point(current_pos)
        control_output = self.pid_controller.compute(current_pos, target_pos, current_vel)
        return control_output, target_pos
    
    def reset(self):
        """重置跟踪器状态"""
        self.current_waypoint_idx = 0
        self.pid_controller.reset()
    
    def is_finished(self, current_pos, threshold=0.3):
        """检查是否完成路径跟踪"""
        if len(self.path) == 0:
            return True
        
        final_target = self.path[-1]
        distance_to_final = np.linalg.norm(np.array(current_pos) - final_target)
        return distance_to_final < threshold