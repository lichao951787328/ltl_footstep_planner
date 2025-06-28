from gurobipy import *
import numpy as np
import math

# --- Helper Functions (Conceptual) ---
def get_alip_matrices(H_com, mass, g, T_ss_dt):
    """
    Calculates discretized ALIP dynamics matrices A_d, B_d.
    T_ss_dt is the time step for discretizing single stance dynamics.
    9B
    """
    # Continuous time ALIP matrices (from Acosta2023, Eq. 1)
    # x = [x_com, y_com, L_x, L_y]^T
    # u = ankle_torque_sagittal
    A_c = np.array([
        [0, 0, 0, 1 / (mass * H_com)],
        [0, 0, -1 / (mass * H_com), 0],
        [0, -mass * g, 0, 0],
        [mass * g, 0, 0, 0]
    ])

    B_c_paper = np.array([
        [0],
        [0], # This implies u is related to a force in y * H or CoP offset.
        [0],
        [1]
    ])
    # Discretization (e.g., using scipy.linalg.expm, or first-order Euler for simplicity here)
    # A_d = I + A_c * T_ss_dt
    # B_d = B_c_paper * T_ss_dt
    # For better accuracy, use matrix exponential:
    from scipy.linalg import expm, block_diag
    M_cont = np.block([
        [A_c, B_c_paper],
        [np.zeros((B_c_paper.shape[1], A_c.shape[1])), np.zeros((B_c_paper.shape[1], B_c_paper.shape[1]))]
    ])
    M_disc = expm(M_cont * T_ss_dt)
    A_d = M_disc[0:A_c.shape[0], 0:A_c.shape[1]]
    B_d = M_disc[0:A_c.shape[0], A_c.shape[1]:]

    return A_d, B_d

def get_alip_reset_map_matrices(A_d_double_stance, B_cop_double_stance, T_ds, H_com, mass):
    """
    Calculates ALIP reset map matrices A_r, B_r (from Acosta2023, Eq. 5-8).
    This involves discretizing double stance dynamics and coordinate transform.
    9C
    """
    A_c_reset = np.array([ # ALIP dynamics for reset map (similar to A_c)
        [0, 0, 0, 1 / (mass * H_com)],
        [0, 0, -1 / (mass * H_com), 0],
        [0, -mass * g, 0, 0],
        [mass * g, 0, 0, 0]
    ])
    B_c_reset = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, mass * g, 0],
        [-mass * g, 0, 0]
    ])
    from scipy.linalg import expm, block_diag
    M_cont = np.block([
        [A_c_reset, B_c_reset],
        [np.zeros((B_c_reset.shape[1], A_c_reset.shape[1])), np.zeros((B_c_reset.shape[1], B_c_reset.shape[1]))]
    ])
    M_disc = expm(M_cont * T_ds)
    Ar_step = M_disc[0:A_c_reset.shape[0], 0:A_c_reset.shape[1]]
    Br_step = M_disc[0:A_c_reset.shape[0], A_c_reset.shape[1]:]
    tmp = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,0],
        [0,0,0]
    ])
    Br_step = Br_step + tmp;
    return Ar_step, Br_step


# --- Main MPC Setup ---
try:
    # --- Model Parameters (Example Values) ---
    N_horizon = 3  # Number of stance periods in MPC horizon (Acosta uses 3)
    K_knots = 10   # Number of knot points per stance period
    T_ss = 0.4     # Duration of single stance phase (seconds)
    T_ds = 0.1     # Duration of double stance phase (seconds)
    T_ss_dt = T_ss / (K_knots -1) # Time step for discretizing single stance
 
    mass = 30.0    # kg
    g = 9.81       # m/s^2
    H_com_nominal = 0.8 # m (nominal CoM height)

    # Get ALIP dynamics matrices
    A_d, B_d = get_alip_matrices(H_com_nominal, mass, g, T_ss_dt)
    Ar_step, Br_step = get_alip_reset_map_matrices(A_d, B_d, T_ds, H_com_nominal, mass) # Placeholder B_d for B_cop_double_stance

    # Desired velocity (example)
    vx_desired = 0.2 # m/s
    vy_desired = 0.0 # m/s
    # Reference gait parameters (for x_d)
    nominal_stance_width = 0.2 # m
    nominal_step_length = vx_desired * (T_ss + T_ds)

    # --- Create Gurobi Model ---
    m = Model("acosta_mpc")

    # --- Create Variables ---
    # ALIP state: x_alip = [x_com, y_com, L_x, L_y]
    # Input: u_ankle = [sagittal_ankle_torque]

    x_alip_vars = [] # List of lists of Gurobi vars: x_alip_vars[n][k] for n-th stance, k-th knot
    u_ankle_vars = []# List of lists of Gurobi vars: u_ankle_vars[n][k]

    # 第一层 (最外层列表 x_alip_vars)：长度为 N_horizon。每个元素代表一个“步态周期”或“规划阶段” n。
    # x_alip_vars[n] 指向第 n 个阶段的所有状态变量。
    # 第二层 (内层列表 x_n)：长度为 K_knots。每个元素代表在阶段 n 中的第 k 个时间节点的状态。
    # x_alip_vars[n][k] 指向第 n 个阶段的第 k 个时间节点的状态变量。
    # 第三层 (Gurobi变量元组)：长度为 4。这是 m.addVars(4, ...) 返回的结果。每个元素是一个单独的 Gurobi 决策变量，代表状态向量的一个分量（例如 x_com, y_com, Lx, Ly）。
    # x_alip_vars[n][k][j] 指向第 n 个阶段、第 k 个时间节点的第 j 个状态分量 (其中 j 的范围是 0 到 3)。

    for n in range(N_horizon):
        x_n = []
        u_n = []
        # 0到K_knots-1是K_knots个knot点，第0个是单脚支撑的起点（双脚支撑的终点），第k-1是单脚支撑的终点（即双脚支撑的起点）
        for k in range(K_knots):
            # 4维变量
            x_n.append(m.addVars(4, lb=-GRB.INFINITY, name=f"x_n{n}_k{k}"))
            if k < K_knots - 1: # Input is applied between knots
                # 一维变量
                # u_n 是一个（K_knots-1）维的向量，表示每个knot点之间的输入
                u_n.append(m.addVar(lb=-5, ub=5, name=f"u_n{n}_k{k}")) # Ankle torque limit from paper
        # 每一次增加K_knots个变量，x_n为K_knots x 4 矩阵
        x_alip_vars.append(x_n)
        if u_n: # Only add if there are inputs for this stance phase
            u_ankle_vars.append(u_n)

    # Footstep positions (relative to world or previous footstep)
    # p_n = [px, py, pz] for n-th PLANNED footstep (p0 is current stance foot)
    # We need N_horizon+1 footstep positions (p0 to p_N_horizon)
    # p0 is known (current stance foot). p1 to p_N_horizon are decision variables.
    # shiji yinggai shi x y yaw geng heshi, yinwei z,roll, pitch doushi you huanjing yueshu jueding de
    p_foot_vars = [m.addVars(3, lb=-GRB.INFINITY, name=f"p_n{n+1}") for n in range(N_horizon)]

    # Binary variables for foothold selection
    # mu_vars[n][i] = 1 if n-th footstep is in region i
    # Assume N_REGIONS convex polygons are provided by perception
    N_REGIONS = 2 # Example: two steppable regions
    # N_horizon x N_REGIONS
    mu_vars = [m.addVars(N_REGIONS, vtype=GRB.BINARY, name=f"mu_n{n+1}") for n in range(N_horizon)]

    # m.update() # Important after adding variables

    # --- Define Perception Data (Example Steppable Polygons) ---
    # Each region P_i is defined by F_i * p <= c_i (and a plane equation f_i^T p = b_i)
    # For simplicity, assume 2D polygons in XY plane, pz is part of region definition.
    # Region 1: x in [0,1], y in [0,1], z = 0
    F_region1 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    c_region1 = np.array([1, 0, 1, 0, 0.01, 0.01]) # x<=1, x>=0, y<=1, y>=0, z approx 0
    # Region 2: x in [1,2], y in [0.5,1.5], z = 0.1
    F_region2 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    c_region2 = np.array([2, -1, 1.5, -0.5, 0.1+0.01, -(0.1-0.01)]) # x<=2, x>=1, y<=1.5, y>=0.5, z approx 0.1
    
    regions_F = [F_region1, F_region2]
    regions_c = [c_region1, c_region2]
    
    # --- Set Initial Conditions ---
    # Current ALIP state x_current_alip (e.g., from state estimator)
    x_current_alip_val = np.array([0, 0.1, 0, 0]) # Example: [x_com, y_com, L_x, L_y]
    # Current stance foot p_current_foot (p0)
    p_current_foot_val = np.array([0, 0, 0]) # Example: [px, py, pz]

    for i in range(4):
        m.addConstr(x_alip_vars[0][0][i] == x_current_alip_val[i], name=f"init_x_alip_{i}")

    # --- Add Constraints ---
    # 1. ALIP Continuous Dynamics (Eq. 9b)
    for n in range(N_horizon):
        for k in range(K_knots - 1): # 以前一个单脚支撑期为起点
            # x_{k+1} = A_d * x_k + B_d * u_k
            for row in range(4): # For each state in ALIP model
                m.addConstr(
                    x_alip_vars[n][k+1][row] == \
                    quicksum(A_d[row,col] * x_alip_vars[n][k][col] for col in range(4)) + \
                    B_d[row,0] * u_ankle_vars[n][k], # Assuming u_ankle_vars[n][k] is a single var
                    name=f"alip_dyn_n{n}_k{k}_row{row}"
                )

    # 2. ALIP Reset Map (Eq. 9c)
    # x_{n+1,1} = Ar_step * x_{n,K} + Br_step * (p_{n+1} - p_n)
    # p_n for the reset map is the stance foot of phase n.
    # p_{n+1} is the next planned footstep, which becomes stance foot for phase n+1.
    for n in range(N_horizon - 1):
        p_n_eff = p_current_foot_val if n == 0 else [p_foot_vars[n-1][j] for j in range(3)]
        p_np1_eff = [p_foot_vars[n][j] for j in range(3)] # p_foot_vars is 0-indexed for planned steps

        # dp = [p_np1_eff[0] - p_n_eff[0], p_np1_eff[1] - p_n_eff[1], p_np1_eff[2] - p_n_eff[2]]
        # Assuming Br_step is for 2D difference (dx, dy) for simplicity now based on typical ALIP reset maps
        dp_x = p_np1_eff[0] - (p_n_eff[0] if isinstance(p_n_eff[0], Var) else p_n_eff[0])
        dp_y = p_np1_eff[1] - (p_n_eff[1] if isinstance(p_n_eff[1], Var) else p_n_eff[1])
        # dp_z for Br_step would typically adjust effective CoM height or be zero if terrain is planar

        for row in range(4):
            # Br_step for [dp_x, dp_y] if it's 4x2
            # If Br_step is 4x3 (for dp_x, dp_y, dp_z), adjust accordingly
            # Let's assume Br_step is 4x2 expecting [dp_x, dp_y]
            br_term = Br_step[row,0] * dp_x + Br_step[row,1] * dp_y
            # If Br_step is 4x3: br_term += Br_step[row,2] * (p_np1_eff[2] - p_n_eff[2])

            m.addConstr(
                x_alip_vars[n+1][0][row] == \
                quicksum(Ar_step[row,col] * x_alip_vars[n][K_knots-1][col] for col in range(4)) + \
                br_term,
                name=f"alip_reset_n{n}_row{row}"
            )

    # 3. Foothold Constraints (Eq. 9d, implemented via Big-M from Eq. 12)
    M_big = 100 # Big-M constant
    for n in range(N_horizon): # For each planned footstep p_foot_vars[n]
        current_foot_p = p_foot_vars[n]
        for i_region in range(N_REGIONS): # For each possible region
            F_mat = regions_F[i_region]
            c_vec = regions_c[i_region]
            # F_mat * p_n <= c_vec IF mu_vars[n][i_region] == 1
            for row_idx in range(F_mat.shape[0]):
                m.addConstr(
                    quicksum(F_mat[row_idx, col_idx] * current_foot_p[col_idx] for col_idx in range(3)) \
                    <= c_vec[row_idx] + M_big * (1 - mu_vars[n][i_region]),
                    name=f"foothold_n{n}_region{i_region}_row{row_idx}"
                )
        # Each footstep must be in exactly one region (Eq. 9e)
        m.addConstr(quicksum(mu_vars[n][i_reg] for i_reg in range(N_REGIONS)) == 1, name=f"sum_mu_n{n}")

    # 4. State, Input, and Footstep Limits (Eq. 9f and text)
    # - Ankle torque limits are already in var definition.
    # - CoM limits (e.g., y_com to avoid hip roll limits)
    y_com_max = 0.1 # example
    for n in range(N_horizon):
        for k in range(K_knots):
            m.addConstr(x_alip_vars[n][k][1] <= y_com_max, name=f"y_com_max_n{n}_k{k}")
            m.addConstr(x_alip_vars[n][k][1] >= -y_com_max, name=f"y_com_min_n{n}_k{k}")

    # - Footstep kinematic limits (max step length, width, height change)
    # Example: max step length in x, y
    max_dx = 0.5; max_dy = 0.3; max_dz = 0.2
    prev_p = p_current_foot_val
    for n in range(N_horizon):
        current_p_vars = p_foot_vars[n]
        dx = current_p_vars[0] - (prev_p[0] if not isinstance(prev_p[0],Var) else prev_p[0])
        dy = current_p_vars[1] - (prev_p[1] if not isinstance(prev_p[1],Var) else prev_p[1])
        # dz = current_p_vars[2] - (prev_p[2] if not isinstance(prev_p[2],Var) else prev_p[2]) # if pz is variable
        m.addConstr(dx <= max_dx, name=f"dx_max_n{n+1}")
        m.addConstr(dx >= -max_dx, name=f"dx_min_n{n+1}") # Allow backward steps
        m.addConstr(dy <= max_dy, name=f"dy_max_n{n+1}")
        m.addConstr(dy >= -max_dy, name=f"dy_min_n{n+1}")
        # m.addConstr(dz <= max_dz, name=f"dz_max_n{n+1}")
        # m.addConstr(dz >= -max_dz, name=f"dz_min_n{n+1}")
        prev_p = current_p_vars
# 


    # --- Define Reference Trajectory x_d (Conceptual) ---
    # This needs to be computed based on v_desired, nominal_stance_width etc.
    # For each x_alip_vars[n][k], there's a corresponding x_d_nk
    # Here, for simplicity, let's assume a target for the final state of the horizon
    x_d_final_target = np.array([
        x_current_alip_val[0] + N_horizon * nominal_step_length, # Target x_com
        0.0, # Target y_com (center between feet)
        mass * H_com_nominal * vy_desired, # Target L_x (for y_vel)
        mass * H_com_nominal * vx_desired  # Target L_y (for x_vel)
    ])
    
    # Or, more correctly, a reference for each knot point x_d[n][k]
    # This is a complex part involving gait generation.
    # For now, let's use a simple reference: try to keep L_y at a desired value for vx_desired
    # and L_x at zero for vy_desired=0, and y_com at zero.
    x_d_val_simple = np.array([0, 0, mass*H_com_nominal*vy_desired, mass*H_com_nominal*vx_desired])
    # The x_com part of x_d would be advancing with vx_desired.

    # --- Set Objective Function (Eq. 9a) ---
    # minimize sum( (x_nk - x_d_nk)^T Q (x_nk - x_d_nk) + u_nk^T R u_nk ) + terminal_cost
    Q_state = np.diag([1.0, 10.0, 1.0, 10.0]) # Weights for x_com_err, y_com_err, Lx_err, Ly_err
    R_input = np.array([[0.1]])             # Weight for ankle torque


    # 定义目标函数
    objective = 0

    for n in range(N_horizon):
        for k in range(K_knots):
            # State error term: (x_nk - x_d_nk)
            # For simplicity, let x_d_nk be x_d_val_simple, ignoring x_com progression
            x_curr = x_alip_vars[n][k]
            x_err = [x_curr[j] - x_d_val_simple[j] for j in range(4)] # x_d_val_simple[0] should be advancing

            for r in range(4):
                for c_ in range(4):
                    objective += x_err[r] * Q_state[r,c_] * x_err[c_]
            
            # Input cost term (if applicable)
            if k < K_knots - 1:
                u_curr = u_ankle_vars[n][k] # This is a single Gurobi Var
                objective += u_curr * R_input[0,0] * u_curr
    
    # Terminal cost (for x_alip_vars[N_horizon-1][K_knots-1])
    # x_final_err = [x_alip_vars[N_horizon-1][K_knots-1][j] - x_d_final_target[j] for j in range(4)]
    # Q_f_state = Q_state * 10 # Heavier weight on terminal state
    # for r in range(4):
    #     for c_ in range(4):
    #         objective += x_final_err[r] * Q_f_state[r,c_] * x_final_err[c_]

    m.setObjective(objective, GRB.MINIMIZE)

    # --- Optimize ---
    m.Params.MIPGap = 0.05 # Allow a 5% gap for faster solve in MIQP
    m.Params.TimeLimit = 0.1 # Max 100ms solve time for MPC (adjust as needed)
    m.optimize()

    # --- Extract Results (The first step of the plan) ---
    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        print("MPC solution found.")
        # Optimal next footstep p_foot_vars[0]
        optimal_next_foot = [p_foot_vars[0][j].X for j in range(3)]
        print(f"Optimal next footstep (p1): {optimal_next_foot}")

        # Optimal ALIP trajectory for the first stance phase
        # optimal_x0_traj = [[x_alip_vars[0][k][j].X for j in range(4)] for k in range(K_knots)]
        # optimal_u0_traj = [u_ankle_vars[0][k].X for k in range(K_knots-1)]
        # print(f"Optimal x_alip traj (n=0): {optimal_x0_traj}")
        # print(f"Optimal u_ankle traj (n=0): {optimal_u0_traj}")

        # Which region was selected for the first planned step:
        for i_reg in range(N_REGIONS):
            if mu_vars[0][i_reg].X > 0.5:
                print(f"Footstep p1 planned for region {i_reg+1}")
                break
    elif m.status == GRB.INFEASIBLE:
        print("MPC problem is infeasible.")
    else:
        print(f"Optimization ended with status: {m.status}")


except GurobiError as e:
    print(f"Gurobi Error code {e.errno}: {e}")
except AttributeError as e:
    print(f"AttributeError: {e} (Likely Gurobi variable not properly accessed or model not solved)")