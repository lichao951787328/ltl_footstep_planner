from gurobipy import *
import numpy as np
import math
from scipy.linalg import expm, block_diag # Ensure expm is imported

# --- Helper Functions ---
def get_autonomous_alip_matrix_A(H_com, mass, g):
    """
    Returns the continuous-time autonomous ALIP dynamics matrix A (input u=0).
    x_dot = A * x
    x = [x_com, y_com, L_x, L_y]^T
    """
    A_c_autonomous = np.array([
        [0, 0, 0, 1 / (mass * H_com)],
        [0, 0, -1 / (mass * H_com), 0],
        [0, -mass * g, 0, 0],
        [mass * g, 0, 0, 0]
    ])
    return A_c_autonomous

def get_alip_matrices_with_input(H_com, mass, g, T_ss_dt):
    """
    Calculates discretized ALIP dynamics matrices A_d, B_d including input.
    Used for MPC state propagation constraints.
    """
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g)
    # B_c_paper is for sagittal ankle torque affecting L_y
    B_c_input_effect = np.array([
        [0],
        [0],
        [0],
        [1]
    ])
    M_cont = np.block([
        [A_c_autonomous, B_c_input_effect],
        [np.zeros((B_c_input_effect.shape[1], A_c_autonomous.shape[1])), np.zeros((B_c_input_effect.shape[1], B_c_input_effect.shape[1]))]
    ])
    M_disc = expm(M_cont * T_ss_dt)
    A_d_for_mpc = M_disc[0:A_c_autonomous.shape[0], 0:A_c_autonomous.shape[1]]
    B_d_for_mpc = M_disc[0:A_c_autonomous.shape[0], A_c_autonomous.shape[1]:]
    return A_d_for_mpc, B_d_for_mpc


def get_alip_reset_map_matrices_detailed(T_ds, H_com, mass, g):
    """
    Calculates ALIP reset map matrices Ar_ds and Br_combined_delta_p using Eq.6 for Bds.
    x_plus_wrt_p_plus = Ar_ds * x_minus_wrt_p_minus + Br_combined_delta_p * (p_plus_world - p_minus_world)
    Br_combined_delta_p = Bds + Bfp_implicit
    """
    A_c = get_autonomous_alip_matrix_A(H_com, mass, g) # Continuous time A
    Ar_ds = expm(A_c * T_ds) # Discrete time A for double stance duration

    # B_CoP maps CoP offset [dcop_x, dcop_y] relative to p_ to state change.
    B_CoP = np.array([ # 4x2
        [0, 0, 0], [0, 0, 0], [0, mass * g, 0], [-mass * g, 0, 0]
    ])
    Ar_ds_inv =  np.linalg.inv(Ar_ds)  # Inverse of Ar_ds
    A_c_inv = np.linalg.inv(A_c)
    B_ds = Ar_ds @ A_c_inv @ ( (1.0/T_ds) * ( A_c_inv @ (np.eye(A_c.shape[0]) - Ar_ds_inv) ) - Ar_ds_inv) @   B_CoP  
    B_fp =  np.array([ # 4x3
        [1,  0,  0], [ 0, 1,  0], [ 0,  0,  0], [ 0, 0,  0]
    ])
    B_r = B_ds + B_fp
    
    assert Ar_ds.shape == (4,4)
    assert B_r.shape == (4,3)

    return Ar_ds, B_r


def calculate_periodic_alip_reference_states(vx_d, vy_d, stance_width_l,
                                             T_s2s, A_s2s_autonomous_cycle, Br_map_for_cycle,
                                             initial_stance_is_left):
    """
    Calculates the periodic ALIP states for a two-step cycle.
    A_s2s_autonomous_cycle: exp(A_c_autonomous * T_s2s)
    Br_map_for_cycle: The matrix that maps (delta_p_world_xy) to ALIP state change for Eq.14 type dynamics.
                      This should be (Bds+Bfp) from Eq.8, projected to 2D foot displacement.
                      Shape (4x2).
    Returns two states: ref_state_phase0, ref_state_phase1, which are the ALIP states
    at the beginning of each of the two single stance phases in the cycle.
    """
    v_d_vec = np.array([vx_d, vy_d])

    if initial_stance_is_left:
        sigma_1 = +1  # Step 1 (p2-p1): Right foot swings (relative to p1)
        sigma_2 = -1  # Step 2 (p3-p2): Left foot swings (relative to p2)
    else:
        sigma_1 = -1
        sigma_2 = +1

    delta_p_A_x = v_d_vec[0] * T_s2s
    delta_p_A_y = v_d_vec[1] * T_s2s + sigma_1 * stance_width_l
    delta_p_A_2d = np.array([delta_p_A_x, delta_p_A_y]) # World XY displacement

    delta_p_B_x = v_d_vec[0] * T_s2s
    delta_p_B_y = v_d_vec[1] * T_s2s + sigma_2 * stance_width_l
    delta_p_B_2d = np.array([delta_p_B_x, delta_p_B_y])

    # Ensure Br_map_for_cycle is 4x2
    if Br_map_for_cycle.shape[1] != 2:
        print(f"Warning: Br_map_for_cycle expected 2 columns, got {Br_map_for_cycle.shape[1]}. Using first 2.")
        Br_map_2d = Br_map_for_cycle[:, 0:2]
    else:
        Br_map_2d = Br_map_for_cycle

    I_mat = np.eye(4)
    M_cycle = A_s2s_autonomous_cycle
    M_sq_cycle = np.dot(M_cycle, M_cycle)
    
    # (I - M^2) * x_start_cycle = M * Br * delta_p_A + Br * delta_p_B
    lhs_matrix = I_mat - M_sq_cycle
    rhs_vector = np.dot(M_cycle, np.dot(Br_map_2d, delta_p_A_2d)) + np.dot(Br_map_2d, delta_p_B_2d)
    
    try:
        # This x_start_of_2_step_cycle_ref is the ALIP state at the very beginning of the
        # two-step cycle, *before* taking the first step (delta_p_A).
        # It's the state that is periodic over two full step-to-step transitions.
        x_start_of_2_step_cycle_ref = np.linalg.solve(lhs_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        print("Singular matrix in calculating periodic reference. Using zeros.")
        return np.zeros(4), np.zeros(4)

    # ref_state_phase0: ALIP state at the beginning of the FIRST single stance phase in the cycle.
    # This IS x_start_of_2_step_cycle_ref.
    ref_state_phase0 = x_start_of_2_step_cycle_ref
    
    # ref_state_phase1: ALIP state at the beginning of the SECOND single stance phase in the cycle.
    # This is after the first step (delta_p_A) has been taken from ref_state_phase0.
    ref_state_phase1 = np.dot(M_cycle, ref_state_phase0) + np.dot(Br_map_2d, delta_p_A_2d)
        
    return ref_state_phase0, ref_state_phase1

# --- Main MPC Setup ---
try:
    # --- Model Parameters ---
    N_horizon = 3
    K_knots = 10
    T_ss = 0.4
    T_ds = 0.1
    T_ss_dt = T_ss / (K_knots - 1)
    T_s2s_cycle = T_ss + T_ds # Duration of one full step-to-step for periodic ref

    mass = 30.0
    g = 9.81
    H_com_nominal = 0.8

    # --- Get Dynamics Matrices ---
    # For MPC constraints (includes input effect)
    # 摆动
    A_d_mpc, B_d_mpc = get_alip_matrices_with_input(H_com_nominal, mass, g, T_ss_dt)
    
    # For ALIP Reset Map (Eq. 9c)
    # Ar_reset links x_{n,K} to x_{n+1,0}. Br_reset_delta_p links (p_{n+1}-p_n) to x_{n+1,0}.
    # 双脚
    Ar_reset, Br_reset_delta_p = get_alip_reset_map_matrices_detailed(T_ds, H_com_nominal, mass, g)
    # CRITICAL: Br_reset_delta_p (4x3) needs to be correctly derived from (Bds+Bfp) terms of Eq.8.

    # For Autonomous Reference Trajectory Generation
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com_nominal, mass, g)
    A_d_autonomous_knot = expm(A_c_autonomous * T_ss_dt) # For within-stance reference
    A_s2s_autonomous_cycle = expm(A_c_autonomous * T_s2s_cycle) # For periodic cycle calculation

    # Desired velocity & gait parameters
    vx_desired = 0.2
    vy_desired = 0.0
    nominal_stance_width = 0.2

    # --- Create Gurobi Model ---
    m = Model("acosta_mpc_revised")

    # --- Create Variables (same as before) ---
    x_alip_vars = []
    u_ankle_vars = []
    for n in range(N_horizon):
        x_n = []
        u_n = []
        for k in range(K_knots):
            x_n.append(m.addVars(4, lb=-GRB.INFINITY, name=f"x_n{n}_k{k}"))
            if k < K_knots - 1: # 摆动脚落地之后就没有力矩了吗
                u_n.append(m.addVar(lb=-5, ub=5, name=f"u_n{n}_k{k}"))
        x_alip_vars.append(x_n)
        if u_n:
            u_ankle_vars.append(u_n)
    p_foot_vars = [m.addVars(3, lb=-GRB.INFINITY, name=f"p_n{n+1}") for n in range(N_horizon)]
    N_REGIONS = 2
    mu_vars = [m.addVars(N_REGIONS, vtype=GRB.BINARY, name=f"mu_n{n+1}") for n in range(N_horizon)]

    # --- Define Perception Data (same as before) ---
    F_region1 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    c_region1 = np.array([1, 0.5, 1, 1, 0.01, -0.01]) 
    F_region2 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    c_region2 = np.array([5, -1.1, 1, 1, 0.1, -0.1])
    regions_F = [F_region1, F_region2]
    regions_c = [c_region1, c_region2]

    # --- Set Initial Conditions ---
    x_current_alip_val = np.array([0, 0, 0, 0]) 
    p_current_foot_val = np.array([0, 0.1, 0])
    current_stance_is_left = True # IMPORTANT: Define the current stance leg

    for i in range(4):
        m.addConstr(x_alip_vars[0][0][i] == x_current_alip_val[i])

    # --- Generate Reference Trajectory x_d_horizon ---
    # The Br_map_for_cycle for calculate_periodic_alip_reference_states should be (Bds+Bfp) from Eq.8,
    # specifically the part that multiplies p_plus (the next footstep), adapted for 2D displacement.
    # Using Br_reset_delta_p as a placeholder (shape 4x3), we take its 2D part.
    # This IS A SIMPLIFICATION and needs rigorous derivation from Eq.8.
    Br_map_placeholder_2D = Br_reset_delta_p[:, 0:2] 
    
    ref_state_cycle_phase0, ref_state_cycle_phase1 = calculate_periodic_alip_reference_states(
        vx_desired, vy_desired, nominal_stance_width,
        T_s2s_cycle, A_s2s_autonomous_cycle, Br_map_placeholder_2D, # Pass the 2D Br_map
        current_stance_is_left # Pass current stance to correctly start the cycle
    )

    # 开始时刻不应该是初始状态吗？，而且这个为啥没有考虑z呢。当有第一个落脚点推第二个落脚点时，实际上公式13点的z应该是受地形约束
    x_d_horizon = []
    for n_mpc in range(N_horizon):
        x_d_stage_n = []
        # Select the correct starting reference state for MPC stage n
        # If current_stance_is_left, the first MPC phase (n=0) will be an R-swing,
        # which corresponds to the start of the 2-step cycle (ref_state_cycle_phase0).
        # The next MPC phase (n=1) will be an L-swing, corresponding to ref_state_cycle_phase1.
        if n_mpc % 2 == 0:
            x_d_nk_start_of_stage = np.copy(ref_state_cycle_phase0)
        else:
            x_d_nk_start_of_stage = np.copy(ref_state_cycle_phase1)
        
        x_d_nk = np.copy(x_d_nk_start_of_stage)
        x_d_stage_n.append(np.copy(x_d_nk)) # x_d_n,0

        for k_knot in range(1, K_knots): # k from 1 to K_knots-1
            x_d_nk = np.dot(A_d_autonomous_knot, x_d_nk) # Evolve using A_d for T_ss_dt
            x_d_stage_n.append(np.copy(x_d_nk))
        x_d_horizon.append(x_d_stage_n)

    # --- Add Constraints ---
    # 1. ALIP Continuous Dynamics (Using A_d_mpc, B_d_mpc)
    for n in range(N_horizon):
        for k in range(K_knots - 1):
            for row in range(4):
                m.addConstr(
                    x_alip_vars[n][k+1][row] == \
                    quicksum(A_d_mpc[row,col] * x_alip_vars[n][k][col] for col in range(4)) + \
                    B_d_mpc[row,0] * u_ankle_vars[n][k],
                    name=f"alip_dyn_n{n}_k{k}_row{row}"
                )

    # 2. ALIP Reset Map (Using Ar_reset, Br_reset_delta_p)
    for n in range(N_horizon - 1):
        p_n_stance_foot = p_current_foot_val if n == 0 else [p_foot_vars[n-1][j] for j in range(3)]
        p_np1_next_stance_foot = [p_foot_vars[n][j] for j in range(3)]

        dp_x = p_np1_next_stance_foot[0] - (p_n_stance_foot[0] if isinstance(p_n_stance_foot[0], Var) else p_n_stance_foot[0])
        dp_y = p_np1_next_stance_foot[1] - (p_n_stance_foot[1] if isinstance(p_n_stance_foot[1], Var) else p_n_stance_foot[1])
        # Assuming pz is constant or handled if Br_reset_delta_p has a z component
        dp_z = p_np1_next_stance_foot[2] - (p_n_stance_foot[2] if isinstance(p_n_stance_foot[2], Var) else p_n_stance_foot[2])


        for row in range(4):
            # Br_reset_delta_p is (4x3), for [dp_x, dp_y, dp_z]
            br_term = Br_reset_delta_p[row,0] * dp_x + \
                        Br_reset_delta_p[row,1] * dp_y + \
                        Br_reset_delta_p[row,2] * dp_z
            m.addConstr(
                x_alip_vars[n+1][0][row] == \
                quicksum(Ar_reset[row,col] * x_alip_vars[n][K_knots-1][col] for col in range(4)) + \
                br_term,
                name=f"alip_reset_n{n}_row{row}"
            )

# F_region1 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
#     c_region1 = np.array([1, -0.5, 1, -1, 0.01, -0.01]) 
# F_region2 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
#     c_region2 = np.array([5, 1.1, 1, -1, 0.1, -0.1])
    # 3. Foothold Constraints (Big-M, same as before)
    M_big = 100 
    for n in range(N_horizon):
        current_foot_p = p_foot_vars[n]
        for i_region in range(N_REGIONS):
            F_mat = regions_F[i_region]
            c_vec = regions_c[i_region]
            for row_idx in range(F_mat.shape[0]):
                m.addConstr(
                    quicksum(F_mat[row_idx, col_idx] * current_foot_p[col_idx] for col_idx in range(3)) \
                    <= c_vec[row_idx] + M_big * (1 - mu_vars[n][i_region]),
                    name=f"foothold_n{n}_region{i_region}_row{row_idx}"
                )
        m.addConstr(quicksum(mu_vars[n][i_reg] for i_reg in range(N_REGIONS)) == 1, name=f"sum_mu_n{n}")

    # 4. State, Input, and Footstep Limits (same as before, conceptual)
    y_com_max = 0.1 
    for n in range(N_horizon):
        for k in range(K_knots):
            m.addConstr(x_alip_vars[n][k][1] <= y_com_max)
            m.addConstr(x_alip_vars[n][k][1] >= -y_com_max)
    max_dx = 0.5; max_dy = 0.3; # max_dz = 0.2
    prev_p_for_limit = p_current_foot_val
    for n in range(N_horizon):
        # ... (kinematic limits constraints, same logic as before) ...
        dx = p_foot_vars[n][0] - (prev_p_for_limit[0] if not isinstance(prev_p_for_limit[0],Var) else prev_p_for_limit[0])
        dy = p_foot_vars[n][1] - (prev_p_for_limit[1] if not isinstance(prev_p_for_limit[1],Var) else prev_p_for_limit[1])
        m.addConstr(dx <= max_dx); m.addConstr(dx >= -max_dx)
        m.addConstr(dy <= max_dy); m.addConstr(dy >= -max_dy)
        prev_p_for_limit = p_foot_vars[n]


    # --- Set Objective Function (Using x_d_horizon) ---
    Q_state = np.diag([1.0, 10.0, 1.0, 10.0]) 
    R_input_val = 0.1 # Scalar, as u is scalar

    objective = 0
    for n in range(N_horizon):
        for k in range(K_knots):
            x_curr_gurobi_vars = x_alip_vars[n][k]
            x_d_target_vals = x_d_horizon[n][k] # This is now a time-varying reference

            # State error: (x_nk - x_d_nk)^T Q (x_nk - x_d_nk)
            for r_idx in range(4): # Since Q_state is diagonal
                err_r = x_curr_gurobi_vars[r_idx] - x_d_target_vals[r_idx]
                objective += err_r * Q_state[r_idx,r_idx] * err_r
            
            if k < K_knots - 1:
                u_curr_gurobi_var = u_ankle_vars[n][k]
                objective += u_curr_gurobi_var * R_input_val * u_curr_gurobi_var
    
    # Terminal cost (optional, comparing to the final reference state)
    Q_f_state = Q_state * 10 
    x_final_actual_vars = x_alip_vars[N_horizon-1][K_knots-1]
    x_final_desired_vals = x_d_horizon[N_horizon-1][K_knots-1]
    for r_idx in range(4):
        err_f_r = x_final_actual_vars[r_idx] - x_final_desired_vals[r_idx]
        objective += err_f_r * Q_f_state[r_idx,r_idx] * err_f_r
        
    m.setObjective(objective, GRB.MINIMIZE)

    # --- Optimize & Extract Results (same as before) ---
    m.Params.MIPGap = 0.05 
    m.Params.TimeLimit = 0.2 # Increased slightly for potentially more complex reference
    m.setParam('DualReductions', 0)
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        print("MPC solution found.")
        optimal_next_foot = [p_foot_vars[0][j].X for j in range(3)]
        print(f"Optimal next footstep (p1): {optimal_next_foot}")
        for i_reg in range(N_REGIONS):
            if mu_vars[0][i_reg].X > 0.5:
                print(f"Footstep p1 planned for region {i_reg+1}")
                break
    elif m.status == GRB.INFEASIBLE:
        print("MPC problem is infeasible.")
        m.computeIIS()
        m.write("mpfc_infeasible.ilp")
        print("IIS written to mpfc_infeasible.ilp")
    else:
        print(f"Optimization ended with status: {m.status}")

except GurobiError as e:
    print(f"Gurobi Error code {e.errno}: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")