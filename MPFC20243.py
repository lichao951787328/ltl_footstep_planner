from gurobipy import *
import numpy as np
import math
from scipy.linalg import expm, block_diag # Ensure expm is imported
import matplotlib.pyplot as plt # 新增导入
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 新增导入 for 3D polygons
from scipy.spatial import ConvexHull, HalfspaceIntersection # 新增导入 for region plotting

# --- Helper Functions ---
# (你所有的辅助函数保持不变)
def get_autonomous_alip_matrix_A(H_com, mass, g):
    A_c_autonomous = np.array([
        [0, 0, 0, 1 / (mass * H_com)],
        [0, 0, -1 / (mass * H_com), 0],
        [0, -mass * g, 0, 0],
        [mass * g, 0, 0, 0]
    ])
    return A_c_autonomous

def get_alip_matrices_with_input(H_com, mass, g, T_ss_dt):
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g)
    B_c_input_effect = np.array([
        [0],
        [0],
        [0],
        [1]
    ])
    # M_cont = np.block([
    #     [A_c_autonomous, B_c_input_effect],
    #     [np.zeros((B_c_input_effect.shape[1], A_c_autonomous.shape[1])), np.zeros((B_c_input_effect.shape[1], B_c_input_effect.shape[1]))]
    # ])
    # M_disc = expm(M_cont * T_ss_dt)
    # A_d_for_mpc = M_disc[0:A_c_autonomous.shape[0], 0:A_c_autonomous.shape[1]]
    # B_d_for_mpc = M_disc[0:A_c_autonomous.shape[0], A_c_autonomous.shape[1]:]
    A_d_for_mpc =  expm(A_c_autonomous * T_ss_dt)
    B_d_for_mpc = np.linalg.inv(A_c_autonomous) @ (A_d_for_mpc - np.eye(A_c_autonomous.shape[0])) @ B_c_input_effect
    return A_d_for_mpc, B_d_for_mpc

def get_alip_reset_map_matrices_detailed(T_ds, H_com, mass, g):
    A_c = get_autonomous_alip_matrix_A(H_com, mass, g) 
    Ar_ds = expm(A_c * T_ds) 

    B_CoP_paper_formulation = np.array([ # This is B_CoP from the paper, it has 2 columns for dcop_x, dcop_y
        [0,0], [0,0], [0, mass*g], [-mass*g, 0]
    ])
    
    # Derivation of Bds from Eq. (6) in the paper:
    # Bds = Ar * A_c^-1 * ( (1/Tds) * (A_c^-1 * (I - Ar^-1)) - Ar^-1 ) * B_cop
    # Note: The paper's B_cop has 2 columns.
    # The code's B_CoP has 3 columns, which seems to imply CoP can also be offset in Z.
    # Let's align with the paper's B_cop (2 columns) for this calculation for Bds first.
    # If the intent is for B_CoP to be 3 columns, the interpretation of Eq. (4) and (6) needs care.
    # For now, assuming the B_CoP in Eq (4) is relative to p_ in XY, and Z is handled by p_ itself.
    
    # If B_CoP is (4x3) as in the original code, it means it maps [dcop_x, dcop_y, dcop_z]
    # Let's stick to the original code's B_CoP (4x3) and see where it leads.
    # The paper's B_COP (page 3, below eq 4) is (4x2) implicitly:
    # d_L = [0, 0, m*g*dcop_y, -m*g*dcop_x]^T
    # So B_CoP should be:
    # [[0, 0],
    #  [0, 0],
    #  [0, m*g],
    #  [-m*g, 0]]
    # However, the provided code B_CoP is:
    B_CoP_from_code = np.array([ # 4x3
         [0, 0, 0], [0, 0, 0], [0, mass * g, 0], [-mass * g, 0, 0]
    ])
    # This B_CoP_from_code implies the 3rd column is for d_cop_z, but it's all zeros, so it doesn't affect L.
    # Effectively, it's like using the first two columns.
    # Let's use the paper's formulation for B_CoP which is (4x2)
    B_CoP_for_Bds = np.array([
        [0,0],
        [0,0],
        [0, mass*g],
        [-mass*g, 0]
    ]) # 4x2, maps [dcop_x, dcop_y]

    try:
        A_c_inv = np.linalg.inv(A_c)
        Ar_ds_inv = np.linalg.inv(Ar_ds)
        
        # Term1 = (1.0/T_ds) * (A_c_inv @ (np.eye(A_c.shape[0]) - Ar_ds_inv))
        # Term2 = Ar_ds_inv
        # B_ds = Ar_ds @ A_c_inv @ (Term1 - Term2) @ B_CoP_for_Bds # Bds will be 4x2
        
        # Simpler form from other sources / interpretations for first-order hold on CoP:
        # B_ds = Ar_ds @ integral_0^Tds (expm(-A_c*s) ds) @ B_CoP / Tds
        # integral_0^T (expm(A*s)ds) = A_inv (expm(A*T)-I)
        # integral_0^T (expm(-A*s)ds) = A_inv (I - expm(-A*T)) = A_inv (I - Ar_ds_inv)
        # B_ds = Ar_ds @ (A_c_inv @ (np.eye(A_c.shape[0]) - Ar_ds_inv)) @ B_CoP_for_Bds * (1/T_ds) # Bds will be 4x2
        B_ds = Ar_ds @ A_c_inv @ ((1/T_ds) * A_c_inv @ (np.eye(A_c.shape[0]) - Ar_ds_inv) - Ar_ds_inv) @ B_CoP_for_Bds

    except np.linalg.LinAlgError:
        print("Warning: A_c or Ar_ds is singular. B_ds might be incorrect.")
        B_ds = np.zeros((4,2)) # Fallback

    # B_fp from Eq. (7) maps (p+ - p-) to x+
    # x_plus_wrt_p_plus = x_tds_wrt_p_minus + Bfp * (p_plus - p_minus)
    # Bfp has 3 columns to multiply (dp_x, dp_y, dp_z)
    B_fp =  np.array([ # 4x3
        [1,  0,  0], [ 0, 1,  0], [ 0,  0,  0], [ 0, 0,  0]
    ])

    # Br_combined_delta_p from Eq. (8) is [Ar*(-Bds-Bfp) (Bds+Bfp)]
    # We need the Br that multiplies (p_plus - p_minus) to get x_plus_wrt_p_plus.
    # From Eq (8): x_plus = Ar_ds * x_minus + Br_reset_map_term * [p_minus^T, p_plus^T]^T
    # The paper's Br in Eq(8) is actually a block matrix: [Ar*(-Bds-Bfp_implicit_from_Bds), (Bds+Bfp_actual)]
    # Bds from eq 6 is (4x2) and relates to (p+ - p-) in XY.
    # Bfp from eq 7 is (4x3) and relates to (p+ - p-) in XYZ.

    # Reconciling dimensions:
    # If p+ - p- is 3D, then Bds needs to effectively act on the XY part.
    # The formulation in the paper (Eq 8) is:
    # x_plus = Ar * x_minus + Ar * (-Bds_xy - Bfp_xyz_implicit_from_Bds_xy) * p_minus_xy_offset_term
    #                        + (Bds_xy + Bfp_xyz) * p_plus_xyz_offset_term
    # This is complex. Let's simplify based on common practice or what get_alip_reset_map_matrices_detailed *was* doing.
    # The original implementation had: B_r = B_ds + B_fp
    # If B_ds was (4x3) and B_fp was (4x3), this would be fine.
    # But B_ds from Eq.6 is (4x2).
    # This implies B_ds should only contribute to the XY components of delta_p.
    
    # Let's assume B_r maps a 3D delta_p = (p_plus - p_minus)
    # B_r = [Bds_col_x, Bds_col_y, 0_col_z] + Bfp_xyz
    B_ds_padded = np.zeros((4,3))
    B_ds_padded[:, 0:2] = B_ds # Bds contributes to dp_x, dp_y effects

    B_r = B_ds_padded + B_fp # This is Br = (Bds+Bfp) from paper Eq.8, second block part.
                            # This Br multiplies (p_plus - p_minus) as a 3D vector.

    assert Ar_ds.shape == (4,4)
    assert B_r.shape == (4,3)

    return Ar_ds, B_r

def calculate_periodic_alip_reference_states(vx_d, vy_d, stance_width_l,
                                             T_s2s, A_s2s_autonomous_cycle, Br_map_for_cycle,
                                             initial_stance_is_left):
    v_d_vec = np.array([vx_d, vy_d])
    if initial_stance_is_left:
        sigma_1 = -1 
        sigma_2 = 1 
    else:
        sigma_1 = 1
        sigma_2 = -1

    # delta_p_A_2d 和 delta_p_B_2d 不是绝对的脚点位置，而是描述了一个理想两步周期中每一步的“目标腿长和方向”。它们被用作参数来计算能够维持这种周期性行走的初始ALIP状态。
    delta_p_A_x = v_d_vec[0] * T_s2s
    delta_p_A_y = v_d_vec[1] * T_s2s + sigma_1 * stance_width_l
    delta_p_A_2d = np.array([delta_p_A_x, delta_p_A_y]) 

    delta_p_B_x = v_d_vec[0] * T_s2s
    delta_p_B_y = v_d_vec[1] * T_s2s + sigma_2 * stance_width_l
    delta_p_B_2d = np.array([delta_p_B_x, delta_p_B_y])

    if Br_map_for_cycle.shape[1] != 2:
        # print(f"Warning: Br_map_for_cycle expected 2 columns, got {Br_map_for_cycle.shape[1]}. Using first 2.")
        Br_map_2d = Br_map_for_cycle[:, 0:2]
    else:
        Br_map_2d = Br_map_for_cycle

    I_mat = np.eye(4)
    M_cycle = A_s2s_autonomous_cycle
    M_sq_cycle = np.dot(M_cycle, M_cycle)
    
    lhs_matrix = I_mat - M_sq_cycle
    rhs_vector = np.dot(M_cycle, np.dot(Br_map_2d, delta_p_A_2d)) + np.dot(Br_map_2d, delta_p_B_2d)
    
    try:
        x_start_of_2_step_cycle_ref = np.linalg.solve(lhs_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        print("Singular matrix in calculating periodic reference. Using zeros.")
        return np.zeros(4), np.zeros(4)

    ref_state_phase0 = x_start_of_2_step_cycle_ref
    ref_state_phase1 = np.dot(M_cycle, ref_state_phase0) + np.dot(Br_map_2d, delta_p_A_2d)
        
    return ref_state_phase0, ref_state_phase1



# --- Plotting Function ---
def plot_results(p_current_val, p_planned_vals, mu_planned_vals, regions_F_list, regions_c_list, N_horizon_plot, N_REGIONS_plot):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot initial footstep
    ax.scatter(p_current_val[0], p_current_val[1], p_current_val[2], c='black', marker='x', s=100, label='Current Stance Foot (p0)')

    # 2. Plot planned footsteps and connecting lines
    all_ps = [p_current_val] + p_planned_vals
    for n in range(N_horizon_plot):
        p_n = all_ps[n]
        p_np1 = all_ps[n+1]
        
        # Determine color based on chosen region for p_np1 (which is p_foot_vars[n])
        chosen_region_idx = -1
        for i_reg in range(N_REGIONS_plot):
            if mu_planned_vals[n][i_reg] > 0.5: # If this mu is active
                chosen_region_idx = i_reg
                break
        
        color = plt.cm.viridis(chosen_region_idx / max(1, N_REGIONS_plot -1 )) if chosen_region_idx != -1 else 'gray'

        ax.scatter(p_np1[0], p_np1[1], p_np1[2], c=[color], marker='o', s=80, label=f'Planned p{n+1} (Region {chosen_region_idx+1})' if n==0 else None)
        ax.plot([p_n[0], p_np1[0]], [p_n[1], p_np1[1]], [p_n[2], p_np1[2]], c='gray', linestyle='--')

    # 3. Plot foothold regions
    region_colors = [plt.cm.cool(i / max(1,N_REGIONS_plot-1)) for i in range(N_REGIONS_plot)]

    for i_reg in range(N_REGIONS_plot):
        F_mat = regions_F_list[i_reg]
        c_vec = regions_c_list[i_reg]

        # Define halfspaces: F_mat * p <= c_vec -->  F_mat * p - c_vec <= 0
        # For HalfspaceIntersection, Ax + b <= 0, so A = F_mat, b = -c_vec
        # Or, Ax <= b --> A=F_mat, b=c_vec. scipy expects normal vectors pointing inwards.
        # F_mat already defines normals. We need an interior point.
        
        # For box constraints [x_max, -x_min, y_max, -y_min, z_max, -z_min]
        # x <= c[0], -x <= c[1] => x >= -c[1]
        # y <= c[2], -y <= c[3] => y >= -c[3]
        # z <= c[4], -z <= c[5] => z >= -c[5]
        x_max, neg_x_min = c_vec[0], c_vec[1]
        y_max, neg_y_min = c_vec[2], c_vec[3]
        z_max, neg_z_min = c_vec[4], c_vec[5]

        x_min, x_max_val = -neg_x_min, x_max
        y_min, y_max_val = -neg_y_min, y_max
        z_min, z_max_val = -neg_z_min, z_max
        
        # Create an interior point (careful if min == max)
        interior_pt = [
            (x_min + x_max_val) / 2,
            (y_min + y_max_val) / 2,
            (z_min + z_max_val) / 2
        ]
        if x_min == x_max_val: interior_pt[0] = x_min
        if y_min == y_max_val: interior_pt[1] = y_min
        if z_min == z_max_val: interior_pt[2] = z_min


        try:
            # For HalfspaceIntersection: normals pointing outward, so F_mat p - c_vec <= 0 is not it.
            # It's Ax <= b. `halfspaces` should be [normal_x, normal_y, normal_z, offset]
            # where normal.p <= offset. So normal = F_mat[i], offset = c_vec[i]
            # No, scipy HalfspaceIntersection: A*x + b_hs <= 0.
            # Our F*p <= c  is F*p - c <= 0. So A_hs = F_mat, b_hs = -c_vec
            halfspaces = np.hstack((F_mat, -c_vec.reshape(-1,1)))

            hs_isect = HalfspaceIntersection(halfspaces, np.array(interior_pt))
            
            # Get vertices of the intersection
            vertices = hs_isect.intersections
            if vertices.shape[0] < 3: # Need at least 3 points for a 2D face, 4 for 3D
                print(f"Region {i_reg+1} has too few vertices ({vertices.shape[0]}) to form a polyhedron.")
                continue

            # Get convex hull of these vertices to find faces
            hull = ConvexHull(vertices) #
            
            # Plot faces
            # hull.simplices gives indices of vertices forming each face
            faces = []
            for s in hull.simplices:
                faces.append([vertices[i] for i in s])
            
            poly_collection = Poly3DCollection(faces, alpha=0.25, facecolors=[region_colors[i_reg]], linewidths=1, edgecolors='k')
            ax.add_collection3d(poly_collection)
            # Add a label for the region patch (for legend) - a bit hacky for Poly3DCollection
            ax.plot([],[],[], color=region_colors[i_reg],label=f'Region {i_reg+1}', alpha=0.5)


        except Exception as e:
            print(f"Could not plot region {i_reg+1}: {e}")
            print(f"  F_mat:\n{F_mat}")
            print(f"  c_vec:\n{c_vec}")
            print(f"  Interior point guess: {interior_pt}")


    ax.set_xlabel('X world (m)')
    ax.set_ylabel('Y world (m)')
    ax.set_zlabel('Z world (m)')
    ax.set_title('MPC Planned Footholds and Regions')
    
    # Auto-scaling for axis limits can be tricky with Poly3DCollection
    # Collect all points to set reasonable limits
    all_plot_points = np.array(all_ps)
    min_coords = all_plot_points.min(axis=0)
    max_coords = all_plot_points.max(axis=0)

    # If regions were plotted, consider their extents too (approximate)
    for i_reg in range(N_REGIONS_plot):
        c_vec = regions_c_list[i_reg]
        x_max_r, neg_x_min_r = c_vec[0], c_vec[1]
        y_max_r, neg_y_min_r = c_vec[2], c_vec[3]
        z_max_r, neg_z_min_r = c_vec[4], c_vec[5]
        min_coords = np.minimum(min_coords, [-neg_x_min_r, -neg_y_min_r, -neg_z_min_r])
        max_coords = np.maximum(max_coords, [x_max_r, y_max_r, z_max_r])
        
    center = (min_coords + max_coords) / 2
    plot_range = np.max(max_coords - min_coords) * 0.6 # Add some padding
    if plot_range < 0.1: plot_range = 1.0 # Ensure some minimal range

    ax.set_xlim(center[0] - plot_range, center[0] + plot_range)
    ax.set_ylim(center[1] - plot_range, center[1] + plot_range)
    ax.set_zlim(center[2] - plot_range, center[2] + plot_range)

    # Enforce somewhat equal aspect ratio in 3D (pyplot makes this hard)
    ax.set_box_aspect([1,1,1]) # This is a newer matplotlib feature for somewhat equal scaling
    # ax.axis('equal') # Often doesn't work well for 3D

    # Create a unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()

# --- Main MPC Setup ---
try:
    # --- Model Parameters ---
    N_horizon = 9
    K_knots = 10
    T_ss = 0.4
    T_ds = 0.1
    T_ss_dt = T_ss / (K_knots - 1)
    T_s2s_cycle = T_ss + T_ds 

    mass = 30.0
    g = 9.81
    H_com_nominal = 0.8

    # --- Get Dynamics Matrices ---
    A_d_mpc, B_d_mpc = get_alip_matrices_with_input(H_com_nominal, mass, g, T_ss_dt)
    Ar_reset, Br_reset_delta_p = get_alip_reset_map_matrices_detailed(T_ds, H_com_nominal, mass, g)
    
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com_nominal, mass, g)
    A_d_autonomous_knot = expm(A_c_autonomous * T_ss_dt) 
    A_s2s_autonomous_cycle = expm(A_c_autonomous * T_s2s_cycle) 

    vx_desired = 0.2
    vy_desired = 0.0
    nominal_stance_width = 0.2

    m = Model("acosta_mpc_revised")

    x_alip_vars = []
    u_ankle_vars = []
    for n in range(N_horizon):
        x_n = []
        u_n = []
        for k in range(K_knots):
            x_n.append(m.addVars(4, lb=-GRB.INFINITY, name=f"x_n{n}_k{k}"))
            if k < K_knots - 1: 
                u_n.append(m.addVar(lb=-5, ub=5, name=f"u_n{n}_k{k}"))
        x_alip_vars.append(x_n)
        if u_n:
            u_ankle_vars.append(u_n)
    p_foot_vars = [m.addVars(3, lb=-GRB.INFINITY, name=f"p_n{n+1}") for n in range(N_horizon)]
    N_REGIONS = 2
    mu_vars = [m.addVars(N_REGIONS, vtype=GRB.BINARY, name=f"mu_n{n+1}") for n in range(N_horizon)]

    # --- Define Perception Data ---
    # F_region @ p <= c_region
    # Format of c_region: [x_max, -x_min, y_max, -y_min, z_max, -z_min]
    F_region_common = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    
    # Region 1: -0.5 <= x <= 1,  -1 <= y <= 1, z = 0.01
    c_region1 = np.array([1, 0.5, 1, 1, 0.01, -0.01]) 
    # Region 2: 1.1 <= x <= 5, -1 <= y <= 1, z = 0.1
    c_region2 = np.array([5, -1.1, 1, 1, 0.1, -0.1])
    
    regions_F = [F_region_common, F_region_common]
    regions_c = [c_region1, c_region2]


    x_current_alip_val = np.array([0, 0, 0, -0.2]) 
    p_current_foot_val = np.array([0, 0.1, 0.01]) # Initial stance foot at (0, 0.1, 0)
    current_stance_is_left = True 

    

    Br_map_placeholder_2D = Br_reset_delta_p[:, 0:2] 
    
    ref_state_cycle_phase0, ref_state_cycle_phase1 = calculate_periodic_alip_reference_states(
        vx_desired, vy_desired, nominal_stance_width,
        T_s2s_cycle, A_s2s_autonomous_cycle, Br_map_placeholder_2D, 
        current_stance_is_left 
    )
    
    if current_stance_is_left: # 或者根据您的具体初始相位定义
        x_current_alip_val = np.copy(ref_state_cycle_phase0)
    else:
        x_current_alip_val = np.copy(ref_state_cycle_phase1)
        
    for i in range(4):
        m.addConstr(x_alip_vars[0][0][i] == x_current_alip_val[i])
    
    x_d_horizon = []
    for n_mpc in range(N_horizon):
        x_d_stage_n = []
        if n_mpc % 2 == 0:
            x_d_nk_start_of_stage = np.copy(ref_state_cycle_phase0)
        else:
            x_d_nk_start_of_stage = np.copy(ref_state_cycle_phase1)
        
        x_d_nk = np.copy(x_d_nk_start_of_stage)
        x_d_stage_n.append(np.copy(x_d_nk)) 

        for k_knot in range(1, K_knots): 
            x_d_nk = np.dot(A_d_autonomous_knot, x_d_nk) 
            x_d_stage_n.append(np.copy(x_d_nk))
        x_d_horizon.append(x_d_stage_n)

    # 约束9b
    for n in range(N_horizon):
        for k in range(K_knots - 1):
            for row in range(4):
                m.addConstr(
                    x_alip_vars[n][k+1][row] == \
                    quicksum(A_d_mpc[row,col] * x_alip_vars[n][k][col] for col in range(4)) + \
                    B_d_mpc[row,0] * u_ankle_vars[n][k],
                    name=f"alip_dyn_n{n}_k{k}_row{row}"
                )
    # 约束9c
    for n in range(N_horizon - 1):
        # p_n_stance_foot is the foot location for ALIP state x_alip_vars[n]
        # For n=0, x_alip_vars[0] is relative to p_current_foot_val
        # For n>0, x_alip_vars[n] is relative to p_foot_vars[n-1] (which is p_n)
        if n == 0:
            p_n_val_for_dp = p_current_foot_val # p0
        else:
            # p_foot_vars[n-1] is p_n (e.g., if n=1, p_foot_vars[0] is p1)
            p_n_val_for_dp = [p_foot_vars[n-1][j] for j in range(3)] 
        
        # p_np1_next_stance_foot is p_foot_vars[n] (e.g. if n=0, p_foot_vars[0] is p1)
        p_np1_val_for_dp = [p_foot_vars[n][j] for j in range(3)]

        # dp = p_{n+1} - p_n
        dp_x = p_np1_val_for_dp[0] - (p_n_val_for_dp[0] if isinstance(p_n_val_for_dp[0], Var) else p_n_val_for_dp[0])
        dp_y = p_np1_val_for_dp[1] - (p_n_val_for_dp[1] if isinstance(p_n_val_for_dp[1], Var) else p_n_val_for_dp[1])
        dp_z = p_np1_val_for_dp[2] - (p_n_val_for_dp[2] if isinstance(p_n_val_for_dp[2], Var) else p_n_val_for_dp[2])

        for row in range(4):
            br_term = Br_reset_delta_p[row,0] * dp_x + \
                        Br_reset_delta_p[row,1] * dp_y + \
                        Br_reset_delta_p[row,2] * dp_z
            m.addConstr(
                x_alip_vars[n+1][0][row] == \
                quicksum(Ar_reset[row,col] * x_alip_vars[n][K_knots-1][col] for col in range(4)) + \
                br_term,
                name=f"alip_reset_n{n}_row{row}"
            )
    # 约束9d
    M_big = 100 
    for n in range(N_horizon):
        current_foot_p_var = p_foot_vars[n] # This is p_{n+1} in math notation (p1, p2, p3 for N=3)
        for i_region in range(N_REGIONS):
            F_mat = regions_F[i_region]
            c_vec = regions_c[i_region]
            for row_idx in range(F_mat.shape[0]):
                m.addConstr(
                    quicksum(F_mat[row_idx, col_idx] * current_foot_p_var[col_idx] for col_idx in range(3)) \
                    <= c_vec[row_idx] + M_big * (1 - mu_vars[n][i_region]),
                    name=f"foothold_n{n}_region{i_region}_row{row_idx}"
                )
        m.addConstr(quicksum(mu_vars[n][i_reg] for i_reg in range(N_REGIONS)) == 1, name=f"sum_mu_n{n}")

    y_com_max = 0.15 
    for n in range(N_horizon):
        for k in range(K_knots):
            m.addConstr(x_alip_vars[n][k][1] <= y_com_max)
            m.addConstr(x_alip_vars[n][k][1] >= -y_com_max)
            
    max_dx = 0.4; max_dy = 0.3; 
    
    # Kinematic limits: (p_{n+1} - p_n)
    prev_p_for_kin_limit = p_current_foot_val # This is p0
    for n in range(N_horizon):
        # p_foot_vars[n] is p_{n+1} (e.g. p1, p2, p3 for N=3)
        current_p_for_kin_limit = p_foot_vars[n]
        dx = current_p_for_kin_limit[0] - (prev_p_for_kin_limit[0] if not isinstance(prev_p_for_kin_limit[0],Var) else prev_p_for_kin_limit[0])
        dy = current_p_for_kin_limit[1] - (prev_p_for_kin_limit[1] if not isinstance(prev_p_for_kin_limit[1],Var) else prev_p_for_kin_limit[1])
        # dz = current_p_for_kin_limit[2] - (prev_p_for_kin_limit[2] if not isinstance(prev_p_for_kin_limit[2],Var) else prev_p_for_kin_limit[2]) # If dz limits needed

        m.addConstr(dx <= max_dx, name=f"max_dx_n{n}")
        m.addConstr(dx >= -max_dx, name=f"min_dx_n{n}")
        m.addConstr(dy <= max_dy, name=f"max_dy_n{n}")
        m.addConstr(dy >= -max_dy, name=f"min_dy_n{n}")
        
        prev_p_for_kin_limit = current_p_for_kin_limit # Update for next iteration

    Q_state = np.diag([1.0, 1.0, 1.0, 1.0]) 
    R_input_val = 0.1 
    objective = 0
    for n in range(N_horizon):
        for k in range(K_knots):
            x_curr_gurobi_vars = x_alip_vars[n][k]
            x_d_target_vals = x_d_horizon[n][k] 
            for r_idx in range(4): 
                err_r = x_curr_gurobi_vars[r_idx] - x_d_target_vals[r_idx]
                objective += err_r * Q_state[r_idx,r_idx] * err_r
            
            if k < K_knots - 1:
                u_curr_gurobi_var = u_ankle_vars[n][k]
                objective += u_curr_gurobi_var * R_input_val * u_curr_gurobi_var
    
    Q_f_state = Q_state * 0.1
    x_final_actual_vars = x_alip_vars[N_horizon-1][K_knots-1]
    x_final_desired_vals = x_d_horizon[N_horizon-1][K_knots-1]
    for r_idx in range(4):
        err_f_r = x_final_actual_vars[r_idx] - x_final_desired_vals[r_idx]
        objective += err_f_r * Q_f_state[r_idx,r_idx] * err_f_r
        
    m.setObjective(objective, GRB.MINIMIZE)

    m.Params.MIPGap = 0.05 
    m.Params.TimeLimit = 0.5 # Increased slightly for plotting overhead if run in same script
    # m.setParam('DualReductions', 0)
    # m.Params.NonConvex = 2 # If you have quadratic constraints (not here)
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        print("MPC solution found.")
        print(f"Gurobi optimization time: {m.Runtime:.4f} seconds") # 打印求解时间
        # Extract results for plotting
        p_planned_vals_optimized = []
        print("\n--- Planned Footstep Sequence (p1, p2, ...) ---") # 添加标题
        for n_opt in range(N_horizon):
            current_planned_foot = [p_foot_vars[n_opt][j].X for j in range(3)] # 提取脚点
            p_planned_vals_optimized.append(current_planned_foot)
            print(f"p{n_opt+1}: x={current_planned_foot[0]:.3f}, y={current_planned_foot[1]:.3f}, z={current_planned_foot[2]:.3f}") # 打印脚点
        
        mu_planned_vals_optimized = []
        for n_opt in range(N_horizon):
            mu_planned_vals_optimized.append([mu_vars[n_opt][i_reg].X for i_reg in range(N_REGIONS)])

        print(f"Optimal first planned footstep (p1): {p_planned_vals_optimized[0]}")
        for i_reg_check in range(N_REGIONS):
            if mu_planned_vals_optimized[0][i_reg_check] > 0.5:
                print(f"Footstep p1 planned for region {i_reg_check+1}")
                break
        
        # Call plotting function
        plot_results(p_current_foot_val, 
                     p_planned_vals_optimized, 
                     mu_planned_vals_optimized, 
                     regions_F, 
                     regions_c,
                     N_horizon, N_REGIONS)

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
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()