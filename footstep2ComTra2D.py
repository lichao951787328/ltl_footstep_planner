import numpy as np
import matplotlib.pyplot as plt

def generate_com_trajectory(footsteps, initial_com_pos, initial_com_vel, step_time, com_height, g=9.81):
    """
    使用ALIP/LIP模型从落脚点序列生成CoM轨迹。

    Args:
        footsteps (list of np.array): 落脚点 [x, y] 坐标序列。
        initial_com_pos (np.array): 初始CoM位置 [x, y]。
        initial_com_vel (np.array): 初始CoM速度 [vx, vy]。
        step_time (float): 每一步的持续时间 (T_ss)。
        com_height (float): 质心高度 (z_c)。
        g (float): 重力加速度。

    Returns:
        tuple: (com_positions, com_velocities) 轨迹列表。
    """
    omega0 = np.sqrt(g / com_height)
    
    # 初始化
    current_com_pos = np.array(initial_com_pos)
    current_com_vel = np.array(initial_com_vel)
    
    com_trajectory_pos = []
    com_trajectory_vel = []
    
    # 迭代每一步
    for i in range(len(footsteps) - 1):
        stance_foot_pos = np.array(footsteps[i])
        
        # 记录该步开始时的状态
        com_trajectory_pos.append(current_com_pos.copy())
        com_trajectory_vel.append(current_com_vel.copy())
        
        # 在单脚支撑阶段内进行精细采样以获得平滑轨迹
        dt = 0.01  # 采样时间步长
        t = dt
        while t < step_time:
            # 解析解公式
            c_t = np.cosh(omega0 * t)
            s_t = np.sinh(omega0 * t)
            
            pos_t = (current_com_pos - stance_foot_pos) * c_t + (current_com_vel / omega0) * s_t + stance_foot_pos
            vel_t = (current_com_pos - stance_foot_pos) * omega0 * s_t + current_com_vel * c_t
            
            com_trajectory_pos.append(pos_t)
            com_trajectory_vel.append(vel_t)
            t += dt
            
        # 计算并更新该步结束时的状态，作为下一步的初始状态
        c_T = np.cosh(omega0 * step_time)
        s_T = np.sinh(omega0 * step_time)
        
        next_com_pos = (current_com_pos - stance_foot_pos) * c_T + (current_com_vel / omega0) * s_T + stance_foot_pos
        next_com_vel = (current_com_pos - stance_foot_pos) * omega0 * s_T + current_com_vel * c_T
        
        current_com_pos = next_com_pos
        current_com_vel = next_com_vel

    # 添加最后一个状态点
    com_trajectory_pos.append(current_com_pos)
    com_trajectory_vel.append(current_com_vel)
        
    return com_trajectory_pos, com_trajectory_vel

# --- 示例使用 ---
if __name__ == '__main__':
    # 1. 输入参数
    footstep_sequence = [
        [0.0, 0.1],   # P0
        [0.3, -0.1],  # P1
        [0.6, 0.1],   # P2
        [0.9, -0.1],  # P3
        [1.2, 0.1]    # P4
    ]
    
    # 初始CoM通常在第一和第二落脚点之间
    initial_pos = [0.0, 0.0] 
    initial_vel = [0.4, 0.0]  # 初始向前速度
    
    T_step = 0.8  # s
    Z_c = 0.8     # m

    # 2. 生成轨迹
    com_pos, com_vel = generate_com_trajectory(
        footsteps=footstep_sequence,
        initial_com_pos=initial_pos,
        initial_com_vel=initial_vel,
        step_time=T_step,
        com_height=Z_c
    )
    
    # 3. 可视化结果
    footsteps_np = np.array(footstep_sequence)
    com_pos_np = np.array(com_pos)
    
    plt.figure(figsize=(10, 6))
    # 绘制落脚点
    plt.plot(footsteps_np[:, 0], footsteps_np[:, 1], 'bo', markersize=10, label='Footsteps (P_k)')
    for i, p in enumerate(footsteps_np):
        plt.text(p[0] + 0.02, p[1], f'P{i}', fontsize=12)
        
    # 绘制CoM轨迹
    plt.plot(com_pos_np[:, 0], com_pos_np[:, 1], 'r-', label='CoM Trajectory')
    # plt.plot(com_pos_np[0, 0], com_pos_np[0, 1], 'gx', markersize=12, label='Initial CoM')
    
    plt.title('CoM Trajectory Generation using ALIP Model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()