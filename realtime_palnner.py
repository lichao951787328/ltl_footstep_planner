from gurobipy import *
import math
import numpy as np
from scipy.linalg import expm, block_diag # Ensure expm is imported
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = patches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p


def get_autonomous_alip_matrix_A(H_com, mass, g):
    """
    Returns the continuous-time state matrix A_c for an autonomous ALIP system.
    ALIP state: [x_com_rel, y_com_rel, L_x_world, L_y_world]
    where L_x_world, L_y_world are angular momentum components about the stance foot in world frame.
    This definition of A_c implies that L_x_dot = -m*g*y_com_rel and L_y_dot = m*g*x_com_rel
    """
    A_c_autonomous = np.array([
        [0, 0, 0, 1 / (mass * H_com)],  # dx_com_rel/dt = (1/mH) * L_y_world
        [0, 0, -1 / (mass * H_com), 0], # dy_com_rel/dt = -(1/mH) * L_x_world
        [0, -mass * g, 0, 0],          # dL_x_world/dt = -m*g*y_com_rel
        [mass * g, 0, 0, 0]            # dL_y_world/dt = m*g*x_com_rel
    ])
    return A_c_autonomous


def get_alip_matrices_with_input(H_com, mass, g, T_dt): # Renamed T_ss_dt to T_dt for clarity
    """
    Returns discrete-time ALIP matrices A_d and B_d for MPC.
    Assumes input u affects L_y_world (e.g., sagittal ankle torque scaled appropriately).
    """
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g)
    # B_c_input_effect: How input u (e.g., a scaled ankle torque tau_y) affects ALIP state derivatives
    # If u is an external CoP offset dCoP_y that creates torque: dL_y_world/dt = m*g*dCoP_y
    # If u is directly an ankle torque tau_y, then dL_y_world/dt = tau_y.
    # The paper's B matrix for MPC (eq 9a) suggests input directly affects L_dot.
    # Let's assume u is a generalized force/torque term that directly adds to dL_y/dt
    B_c_input_effect = np.array([
        [0], # No direct effect on x_com_rel_dot from u
        [0], # No direct effect on y_com_rel_dot from u
        [0], # No direct effect on L_x_dot from u (assuming u is sagittal plane input)
        [1]  # u directly affects L_y_dot
    ])

    A_d = expm(A_c_autonomous * T_dt)
    # Zero-Order Hold for B_d: B_d = A_c_inv @ (A_d - I) @ B_c
    try:
        A_c_inv = np.linalg.inv(A_c_autonomous)
        B_d = A_c_inv @ (A_d - np.eye(A_c_autonomous.shape[0])) @ B_c_input_effect
    except np.linalg.LinAlgError:
        print("Warning: A_c_autonomous is singular during B_d calculation. Using an approximation for B_d.")
        # Fallback or simple forward Euler for B_d if A_c is singular (should not happen for ALIP A_c)
        B_d = B_c_input_effect * T_dt # This is a rough approximation
    return A_d, B_d

# def get_alip_matrices_autonomous_discrete(H_com, mass, g, T_dt): # Renamed from get_alip_matrices
#     """Returns discrete-time A_d for autonomous ALIP evolution over T_dt."""
#     A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g)
#     A_d_autonomous =  expm(A_c_autonomous * T_dt)
#     return A_d_autonomous

def get_alip_reset_map_matrices(T_ds, H_com, mass, g): # Simplified name
    """
    Calculates ALIP reset map matrices Ar_ds and Br_delta_p.
    x_new_initial_wrt_p_new = Ar_ds @ x_old_final_wrt_p_old + Br_delta_p @ (p_new_world - p_old_world)

    Ar_ds: Evolution of ALIP state (wrt old foot) during double support T_ds.
    Br_delta_p: Change in ALIP state (now wrt new foot) due to the shift of reference frame.
                This simplified Br_delta_p assumes CoP is at the new stance foot during DS,
                or T_ds is very short, so no complex B_ds term from CoP modulation.
    """
    A_c = get_autonomous_alip_matrix_A(H_com, mass, g)
    Ar_ds = expm(A_c * T_ds) # Evolution of x_old_final_wrt_p_old over T_ds

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
    B_ds_padded = np.zeros((4,3))
    B_ds_padded[:, 0:2] = B_ds # Bds contributes to dp_x, dp_y effects
    B_r = B_ds_padded + B_fp 
    assert Ar_ds.shape == (4,4)
    assert B_r.shape == (4,3)

    return Ar_ds, B_r


if __name__ == "__main__":
    try:
        # --- Part 1: High-level Footstep Planning (MIQCQP result placeholder) ---
        m_global = Model("footstep_global")
        m_global.Params.OutputFlag = 0 # Suppress Gurobi output for this part
# Create N footstep variables (x,y,theta,s,c)       
        N_global = 18 # Number of footsteps in the global plan
        # Decision variables for global plan (x,y,theta) + (sin_theta, cos_theta)
        footsteps_global_vars = [m_global.addVars(5, lb=-5, ub=5, name="F"+str(i)) for i in range(N_global)]
        S_global = [m_global.addVars(5, vtype=GRB.BINARY, name="S_global"+str(i)) for i in range(N_global)]
        C_global = [m_global.addVars(5, vtype=GRB.BINARY, name="C_global"+str(i)) for i in range(N_global)]
        N_REG_global = 5
        H_global = [m_global.addVars(N_REG_global, vtype=GRB.BINARY, name="H_global"+str(i)) for i in range(N_global)]

        # --- Define Regions (Scenario 3 from original code) ---
        # R1_xmax, R1_xmin, R1_ymax, R1_ymin = 1, 0, 1, 0
        # R2_xmax, R2_xmin, R2_ymax, R2_ymin = 1.6, 1.1, 2, 0
        # R3_xmax, R3_xmin, R3_ymax, R3_ymin = 2, 1.1, 2.5, 2.1
        # R4_xmax, R4_xmin, R4_ymax, R4_ymin = 1, -0.5, 2.7, 2.1
        # R5_xmax, R5_xmin, R5_ymax, R5_ymin = 2, 1.5, 3, 2.55
        # regions_geom_global = [
        #     (R1_xmax, R1_xmin, R1_ymax, R1_ymin), (R2_xmax, R2_xmin, R2_ymax, R2_ymin),
        #     (R3_xmax, R3_xmin, R3_ymax, R3_ymin), (R4_xmax, R4_xmin, R4_ymax, R4_ymin),
        #     (R5_xmax, R5_xmin, R5_ymax, R5_ymin)
        # ]
        # region_midpts_global = [((R[0]+R[1])/2, (R[2]+R[3])/2) for R in regions_geom_global]

        # # Simplified region constraints for global planner (Ax <= b for foot [x,y,theta])
        # # This needs careful formulation if A_coeffs_global and b_coeffs_global are from original code
        # # Original A_i were for [x,y,theta] being inside a box for x,y and interval for theta
        # M_global_region = 50
        # for c in range(N_global):
        #     for i_reg, R_geom in enumerate(regions_geom_global):
        #         Rx_max, Rx_min, Ry_max, Ry_min = R_geom
        #         # Constraints for footstep_global_vars[c][0] (x) and [1] (y) to be in region i_reg
        #         m_global.addConstr(footsteps_global_vars[c][0] <= Rx_max + M_global_region * (1 - H_global[c][i_reg]))
        #         m_global.addConstr(footsteps_global_vars[c][0] >= Rx_min - M_global_region * (1 - H_global[c][i_reg]))
        #         m_global.addConstr(footsteps_global_vars[c][1] <= Ry_max + M_global_region * (1 - H_global[c][i_reg]))
        #         m_global.addConstr(footsteps_global_vars[c][1] >= Ry_min - M_global_region * (1 - H_global[c][i_reg]))
        #         # Theta constraints (example, can be refined)
        #         m_global.addConstr(footsteps_global_vars[c][2] <= math.pi + M_global_region * (1 - H_global[c][i_reg]))
        #         m_global.addConstr(footsteps_global_vars[c][2] >= -math.pi/2 - M_global_region * (1 - H_global[c][i_reg])) # Example
        #     m_global.addConstr(quicksum(H_global[c][j] for j in range(N_REG_global)) == 1)


        R1_xmax = 1
        R1_xmin = 0
        R1_ymax = 1
        R1_ymin = 0
        R1_midpt = [(R1_xmax + R1_xmin)/2 , (R1_ymax + R1_ymin)/2]

        R2_xmax = 1.6
        R2_xmin = 1.1
        R2_ymax = 2
        R2_ymin = 0
        R2_midpt = [(R2_xmax + R2_xmin)/2 , (R2_ymax + R2_ymin)/2]

        R3_xmax = 2
        R3_xmin = 1.1
        R3_ymax = 2.5
        R3_ymin = 2.1
        R3_midpt = [(R3_xmax + R3_xmin)/2 , (R3_ymax + R3_ymin)/2]

        R4_xmax = 1
        R4_xmin = -0.5
        R4_ymax = 2.7
        R4_ymin = 2.1
        R4_midpt = [(R4_xmax + R4_xmin)/2 , (R4_ymax + R4_ymin)/2]

        R5_xmax = 2
        R5_xmin = 1.5
        R5_ymax = 3
        R5_ymin = 2.55
        R5_midpt = [(R5_xmax + R5_xmin)/2 , (R5_ymax + R5_ymin)/2]
        
        A_1 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
        b_1 = [R1_xmax,-R1_xmin,R1_ymax,-R1_ymin,math.pi,math.pi/2]
		# print(A_1[1][2])
		
        A_2 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
        b_2 = [R2_xmax,-R2_xmin,R2_ymax,-R2_ymin,math.pi,math.pi/2]
		
        A_3 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
        b_3 = [R3_xmax,-R3_xmin,R3_ymax,-R3_ymin,math.pi,math.pi/2]

        A_4 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
        b_4 = [R4_xmax,-R4_xmin,R4_ymax,-R4_ymin,math.pi,math.pi/2]

        A_5 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
        b_5 = [R5_xmax,-R5_xmin,R5_ymax,-R5_ymin,math.pi,math.pi/2]

        # All footsteps must be in the regions
        for c in range(0,N_global):
            for i in range(0,len(A_1)):
                M_global = 20
                # Region 1
                m_global.addConstr(-M_global*(1-H_global[c][0]) + quicksum(A_1[i][j]*footsteps_global_vars[c][j] for j in range(0,3)) - b_1[i] <= 0)
                # Region 2
                m_global.addConstr(-M_global*(1-H_global[c][1]) + quicksum(A_2[i][j]*footsteps_global_vars[c][j] for j in range(0,3)) - b_2[i] <= 0)
                # Region 3
                m_global.addConstr(-M_global*(1-H_global[c][2]) + quicksum(A_3[i][j]*footsteps_global_vars[c][j] for j in range(0,3)) - b_3[i] <= 0)
                # Region 4
                m_global.addConstr(-M_global*(1-H_global[c][3]) + quicksum(A_4[i][j]*footsteps_global_vars[c][j] for j in range(0,3)) - b_4[i] <= 0)
                # Region 5
                m_global.addConstr(-M_global*(1-H_global[c][4]) + quicksum(A_5[i][j]*footsteps_global_vars[c][j] for j in range(0,3)) - b_5[i] <= 0)
                # # Region 6
                # m.addConstr(-M*(1-H[c][5]) + quicksum(A_6[i][j]*footsteps[c][j] for j in range(0,3)) - b_6[i] <= 0)
                # # Region 7
                # m.addConstr(-M*(1-H[c][6]) + quicksum(A_7[i][j]*footsteps[c][j] for j in range(0,3)) - b_7[i] <= 0)
                # # Region 8
                # m.addConstr(-M*(1-H[c][7]) + quicksum(A_8[i][j]*footsteps[c][j] for j in range(0,3)) - b_8[i] <= 0)

			# Constraint that the sum of H must be 1 for every foothold
            m_global.addConstr(quicksum(H_global[c][j] for j in range(0,N_REG_global)) == 1 )

        for c in range(2, N_global):
            if (c % 2 == 0): # 约束右脚
                p1 = [0,0.1]
                p2 = [0,-0.8]
                d1 = 0.55
                d2 = 0.55
                xn = footsteps_global_vars[c][0]
                yn = footsteps_global_vars[c][1]
                xc = footsteps_global_vars[c-1][0]
                yc = footsteps_global_vars[c-1][1]
                thetac = footsteps_global_vars[c-1][2]
                term1_a = xn - (xc + p1[0]*footsteps_global_vars[c-1][4] - p1[1]*footsteps_global_vars[c-1][3])
                term2_a = yn - (yc + p1[0]*footsteps_global_vars[c-1][3] + p1[1]*footsteps_global_vars[c-1][4])
                # term1_a = xn - (xc + p1[0])
                # term2_a = yn - (yc + p1[1])
                m_global.addQConstr(term1_a*term1_a + term2_a*term2_a <= d1*d1)

				# term1_b = xn - (xc + p2[0])
				# term2_b = yn - (yc + p2[1])
                term1_b = xn - (xc + p2[0]*footsteps_global_vars[c-1][4] - p2[1]*footsteps_global_vars[c-1][3])
                term2_b = yn - (yc + p2[0]*footsteps_global_vars[c-1][3] + p2[1]*footsteps_global_vars[c-1][4])
                m_global.addQConstr(term1_b*term1_b + term2_b*term2_b <= d2*d2)
            else: # 约束左脚
                p1 = [0, -0.1]
                p2 = [0,0.8]
                d1 = 0.55
                d2 = 0.55
                xn = footsteps_global_vars[c][0]
                yn = footsteps_global_vars[c][1]
                xc = footsteps_global_vars[c-1][0]
                yc = footsteps_global_vars[c-1][1]
                thetac = footsteps_global_vars[c-1][2]
                term1 = xn - (xc + p1[0]*footsteps_global_vars[c-1][4] - p1[1]*footsteps_global_vars[c-1][3])
                term2 = yn - (yc + p1[0]*footsteps_global_vars[c-1][3] + p1[1]*footsteps_global_vars[c-1][4])
				# term1 = xn - (xc + p1[0])
				# term2 = yn - (yc + p1[1])
                m_global.addQConstr(term1*term1 + term2*term2 <= d1*d1)

				# term1 = xn - (xc + p2[0])
				# term2 = yn - (yc + p2[1])
                term1 = xn - (xc + p2[0]*footsteps_global_vars[c-1][4] - p2[1]*footsteps_global_vars[c-1][3])
                term2 = yn - (yc + p2[0]*footsteps_global_vars[c-1][3] + p2[1]*footsteps_global_vars[c-1][4])
                m_global.addQConstr(term1*term1 + term2*term2 <= d2*d2)
                

        # Add constraints for sin
        for c in range(0,N_global):
            for i in range(0,5):
                M = 20
                if i == 0:
                    phi_l = -math.pi
                    phi_lp1 = 1-math.pi
                    g_l = -1
                    h_l = -math.pi
                    m_global.addConstr(-(1-S_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M - footsteps_global_vars[c][3] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + footsteps_global_vars[c][3] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 1:
                    phi_l = 1-math.pi
                    phi_lp1 = -1
                    g_l = 0
                    h_l = -1
                    m_global.addConstr(-(1-S_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M - footsteps_global_vars[c][3] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + footsteps_global_vars[c][3] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 2:
                    phi_l = -1
                    phi_lp1 = 1
                    g_l = 1
                    h_l = 0
                    m_global.addConstr(-(1-S_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M - footsteps_global_vars[c][3] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + footsteps_global_vars[c][3] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 3:
                    phi_l = 1
                    phi_lp1 = math.pi-1
                    g_l = 0
                    h_l = 1
                    m_global.addConstr(-(1-S_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M - footsteps_global_vars[c][3] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + footsteps_global_vars[c][3] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 4:
                    phi_l = math.pi-1
                    phi_lp1 = math.pi
                    g_l = -1
                    h_l = math.pi
                    m_global.addConstr(-(1-S_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M - footsteps_global_vars[c][3] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-S_global[c][i])*M + footsteps_global_vars[c][3] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
            m_global.addConstr(quicksum(S_global[c][j] for j in range(0,5)) == 1 )

		# Add constraints for cos
        for c in range(0,N_global):
            for i in range(0,5):
                M = 50
                if i == 0:
                    phi_l = -math.pi
                    phi_lp1 = -1-math.pi/2
                    g_l = 0
                    h_l = -1
                    m_global.addConstr(-(1-C_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M - footsteps_global_vars[c][4] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + footsteps_global_vars[c][4] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 1:
                    phi_l = -1-math.pi/2
                    phi_lp1 = 1-math.pi/2
                    g_l = 1
                    h_l = math.pi/2
                    m_global.addConstr(-(1-C_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M - footsteps_global_vars[c][4] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + footsteps_global_vars[c][4] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 2:
                    phi_l = 1-math.pi/2
                    phi_lp1 = math.pi/2-1
                    g_l = 0
                    h_l = 1
                    m_global.addConstr(-(1-C_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M - footsteps_global_vars[c][4] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + footsteps_global_vars[c][4] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 3:
                    phi_l = math.pi/2-1
                    phi_lp1 = math.pi/2+1
                    g_l = -1
                    h_l = math.pi/2
                    m_global.addConstr(-(1-C_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M - footsteps_global_vars[c][4] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + footsteps_global_vars[c][4] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
                elif i == 4:
                    phi_l = math.pi/2+1
                    phi_lp1 = math.pi
                    g_l = 0
                    h_l = -1
                    m_global.addConstr(-(1-C_global[c][i])*M - phi_lp1 + footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + phi_l - footsteps_global_vars[c][2] <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M - footsteps_global_vars[c][4] + g_l*footsteps_global_vars[c][2] + h_l <= 0)
                    m_global.addConstr(-(1-C_global[c][i])*M + footsteps_global_vars[c][4] - g_l*footsteps_global_vars[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
            m_global.addConstr(quicksum(C_global[c][j] for j in range(0,5)) == 1 )


        # Initial footstep positions (fixed)
        init_theta_global = 0.0
        f1_s_global = math.sin(init_theta_global)
        f1_c_global = math.cos(init_theta_global)
        m_global.addConstr(footsteps_global_vars[0][0] == 0)
        m_global.addConstr(footsteps_global_vars[0][1] == 0.4) # Adjusted initial y
        # m_global.addConstr(footsteps_global_vars[0][2] == init_theta_global)
        m_global.addConstr(footsteps_global_vars[0][3] == f1_s_global)
        m_global.addConstr(footsteps_global_vars[0][4] == f1_c_global)
        m_global.addConstr(S_global[0][2] == 1) # Initial sin(theta) = sin(0)
        m_global.addConstr(C_global[0][2] == 1) # Initial cos(theta) = cos(0)

        m_global.addConstr(footsteps_global_vars[1][0] == 0)
        m_global.addConstr(footsteps_global_vars[1][1] == 0) # Adjusted initial y
        # m_global.addConstr(footsteps_global_vars[1][2] == init_theta_global)
        m_global.addConstr(footsteps_global_vars[1][3] == f1_s_global)
        m_global.addConstr(footsteps_global_vars[1][4] == f1_c_global)
        m_global.addConstr(S_global[1][2] == 1) # Initial sin(theta) = sin(0)
        m_global.addConstr(C_global[1][2] == 1) # Initial cos

        # Initial/Final "velocity" (displacement) constraints for global plan
        small_disp_thresh_global = 0.15
        T_step_global = 1.0 # Nominal time per step for global plan context
        dx_init_g = footsteps_global_vars[2][0] - footsteps_global_vars[0][0] # From F0 to F2
        dy_init_g = footsteps_global_vars[2][1] - footsteps_global_vars[0][1]
        m_global.addQConstr(dx_init_g*dx_init_g + dy_init_g*dy_init_g <= (small_disp_thresh_global * T_step_global)**2)

        dx_final_g = footsteps_global_vars[N_global-1][0] - footsteps_global_vars[N_global-3][0]
        dy_final_g = footsteps_global_vars[N_global-1][1] - footsteps_global_vars[N_global-3][1]
        m_global.addQConstr(dx_final_g*dx_final_g + dy_final_g*dy_final_g <= (small_disp_thresh_global * T_step_global)**2)



        # Max rotation per step
        for c in range(1, N_global):
            del_theta_max_g = math.pi/8
            m_global.addConstr((footsteps_global_vars[c][2] - footsteps_global_vars[c-1][2]) <= del_theta_max_g)
            m_global.addConstr((footsteps_global_vars[c][2] - footsteps_global_vars[c-1][2]) >= -del_theta_max_g)


        # Always be in regions 2 or 3 before entering region 4
        # reg1 = 0 # region 1
        # reg2 = 1 # region 2
        # phi2_reg = 2 # region 43

        # T = [m_global.addVar(vtype=GRB.BINARY, name="T"+str(i)) for i in range(0,N_global)]

		# # Base case
        # m_global.addConstr(T[N_global-1] == H_global[N_global-1][phi2_reg])

		# # Satisfiability constraint
		# # m.addConstr(quicksum(T[j] for j in range(0,N)) == N)
        # m_global.addConstr(T[0] == 1)

        # Pphi1 = [m_global.addVar(vtype=GRB.BINARY, name="Pphi1"+str(i)) for i in range(0,N_global-1)]
        # B = [m_global.addVar(vtype=GRB.BINARY, name="B"+str(i)) for i in range(0,N_global-1)]

		# # Recursive constraints
        # for i in range(0,N_global-1):
        #     M = 20
        #     delta = 0.001
			
        #     m_global.addConstr(H_global[i][reg1] + H_global[i][reg2] - 1 >= -M*(1-Pphi1[i]))
        #     m_global.addConstr(H_global[i][reg1] + H_global[i][reg2] - 1 + delta <=  M*(Pphi1[i]))

		# 	# Term in parenthesis
        #     m_global.addConstr(Pphi1[i] + T[i+1] - 2 >= -M*(1-B[i]))
        #     m_global.addConstr(Pphi1[i] + T[i+1] - 2 + delta <= M*(B[i]))

		# 	# Final constraint
        #     m_global.addConstr(H_global[i][phi2_reg] + B[i] - 1 >= -M*(1-T[i]))
        #     m_global.addConstr(H_global[i][phi2_reg] + B[i] - 1 + delta <= M*(T[i]))

        # Global Objective
        g_target_global = [1.5, 2.2, 3*math.pi/4] # Target [x,y,theta]
        e0_g = footsteps_global_vars[N_global-1][0] - g_target_global[0]
        e1_g = footsteps_global_vars[N_global-1][1] - g_target_global[1]
        e2_g = footsteps_global_vars[N_global-1][2] - g_target_global[2]
        Q_g_term = np.diag([300, 300, 100]) # Terminal cost weights
        term_cost_g = e0_g*e0_g*Q_g_term[0,0] + e1_g*e1_g*Q_g_term[1,1] + e2_g*e2_g*Q_g_term[2,2]

        R_g_inc = np.diag([0.5, 0.5, 0.5]) # Incremental cost weights (x,y,theta diff)
        inc_cost_g = QuadExpr(0)
        for j in range(1, N_global): # Corrected range from 1
            dx_inc = footsteps_global_vars[j][0] - footsteps_global_vars[j-1][0]
            dy_inc = footsteps_global_vars[j][1] - footsteps_global_vars[j-1][1]
            dt_inc = footsteps_global_vars[j][2] - footsteps_global_vars[j-1][2]
            inc_cost_g += (dx_inc*dx_inc*R_g_inc[0,0] +
                           dy_inc*dy_inc*R_g_inc[1,1] +
                           dt_inc*dt_inc*R_g_inc[2,2])

        # Velocity smoothness (midpoint based) for global plan
        midpoints_xy_g = []
        for j in range(1, N_global):
            mid_x_g = (footsteps_global_vars[j][0] + footsteps_global_vars[j-1][0]) / 2.0
            mid_y_g = (footsteps_global_vars[j][1] + footsteps_global_vars[j-1][1]) / 2.0
            midpoints_xy_g.append((mid_x_g, mid_y_g))

        velocities_xy_g = []
        if N_global > 2:
            for j in range(1, len(midpoints_xy_g)):
                vel_x_g = (midpoints_xy_g[j][0] - midpoints_xy_g[j-1][0]) / T_step_global # Using T_step_global
                vel_y_g = (midpoints_xy_g[j][1] - midpoints_xy_g[j-1][1]) / T_step_global
                velocities_xy_g.append((vel_x_g, vel_y_g))

        vel_smooth_cost_g = QuadExpr(0)
        G_vel_smooth_g = 5.0
        if len(velocities_xy_g) > 1:
            for j in range(1, len(velocities_xy_g)):
                accel_x_g = velocities_xy_g[j][0] - velocities_xy_g[j-1][0]
                accel_y_g = velocities_xy_g[j][1] - velocities_xy_g[j-1][1]
                vel_smooth_cost_g += accel_x_g*accel_x_g + accel_y_g*accel_y_g

        m_global.setObjective(term_cost_g + inc_cost_g + G_vel_smooth_g * vel_smooth_cost_g, GRB.MINIMIZE)
        m_global.Params.MIPFocus = 1 # Focus on finding feasible solutions faster
        m_global.Params.Heuristics = 0.2
        m_global.optimize()
        print(f"Global planner optimization status: {m_global.Status}")
        if m_global.Status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS to find conflicting constraints...")
            m_global.computeIIS()
            m_global.write("model_iis.ilp") # Writes the Irreducible Inconsistent Subsystem to a file
            print("IIS written to model_iis.ilp. Please check this file.")
            # You might want to exit or handle this case rather than trying to access .X
        elif m_global.Status == GRB.OPTIMAL or m_global.Status == GRB.SUBOPTIMAL:
            footsteps_x_global = [m_global.getVarByName(f"F{c}[0]").X for c in range(N_global)]
            # ... rest of your result extraction ...
        else:
            print(f"Optimization ended with status {m_global.Status}. Solution may not be available.")
        print(f"Global footstep planner Gurobi optimization time: {m_global.Runtime:.4f} seconds")


        # footsteps_x_global = [m_global.getVarByName(f"F{c}[0]").X for c in range(N_global)]
        # footsteps_y_global = [m_global.getVarByName(f"F{c}[1]").X for c in range(N_global)]
        # footsteps_theta_global = [m_global.getVarByName(f"F{c}[2]").X for c in range(N_global)]
        # # footsteps_z_global is needed! For now, assume flat terrain.
        # footsteps_z_global = [0.0] * N_global # Placeholder: ASSUME FLAT TERRAIN at z=0

        # print("Global Footsteps X:", footsteps_x_global)
        # print("Global Footsteps Y:", footsteps_y_global)
        # print("Global Footsteps Theta:", footsteps_theta_global)

        # # --- Part 2: Mid-level: Generate Ideal ALIP Reference Trajectory ---
        # K_knots_mpc = 10    # Number of discretization knots within one single support phase for MPC
        # T_ss_mpc = 0.7      # Single support time for MPC reference
        # T_ds_mpc = 0.1      # Double support time for MPC reference
        # T_ss_dt_mpc = T_ss_mpc / (K_knots_mpc - 1) # Time step between knots

        # mass_robot = 30.0
        # g_gravity = 9.81
        # H_com_robot = 0.8   # Nominal CoM height

        # A_c_auton_alip = get_autonomous_alip_matrix_A(H_com_robot, mass_robot, g_gravity)
        # A_d_knot_auton_alip = expm(A_c_auton_alip * T_ss_dt_mpc)
        # Ar_ds_reset, Br_delta_p_reset = get_alip_reset_map_matrices(T_ds_mpc, H_com_robot, mass_robot, g_gravity)

        # # Initial ALIP state for the *reference trajectory generation*
        # # This should be based on the first *two* global footsteps to infer initial CoM state/velocity
        # # For simplicity, let's start with a common heuristic: CoM slightly ahead, some forward momentum
        # # relative to the foot that is about to become the stance foot (footsteps_global_vars[1])
        # # The ALIP state is relative to the *current* stance foot.
        # # If robot starts on footsteps_global_vars[0] and steps to footsteps_global_vars[1],
        # # then the first ALIP trajectory is for SS on footsteps_global_vars[1].
        # # x_alip_current_ref_gen is ALIP state relative to footsteps_global_vars[1] at the start of its SS phase.
        # # A simple starting ALIP state (relative to current stance foot):
        # # [x_com_rel_to_stance, y_com_rel_to_stance, Lx_world_about_stance, Ly_world_about_stance]
        # # Example: Small forward offset from stance foot, and corresponding Ly for forward motion.
        # # This needs careful initialization based on actual robot state or desired startup.
        # # Let's assume a starting state where CoM is at (0,0) relative to footsteps_x_global[1], y_global[1]
        # # and has some initial "push" or desired velocity that translates to L.
        # # If starting from rest on two feet, then stepping to f_g[1]:
        # # CoM might be at (f_g[0]+f_g[1])/2. Then it lands on f_g[1].
        # # CoM pos rel to f_g[1] = (f_g[0]-f_g[1])/2.
        # # This is complex. Let's use a placeholder and acknowledge it needs proper init.
        # initial_com_x_rel_to_f1 = 0.05 # Slightly ahead of f_g[1]
        # initial_com_y_rel_to_f1 = 0.0
        # # For a forward velocity v_fwd, Ly_approx = m * v_fwd * H_com (simplified from L = r x p)
        # # Or, using DCM: Ly_approx = m * g * x_com_rel (if DCM is at origin)
        # # Let's use a common initialization for steady walking: DCM at origin relative to stance.
        # # ξ_x = x_com + Ly / (m * H_com * ω0) = 0 => Ly = - x_com * m * H_com * ω0
        # # ξ_y = y_com - Lx / (m * H_com * ω0) = 0 => Lx =   y_com * m * H_com * ω0
        # # ω0 = sqrt(g_gravity / H_com_robot)
        # omega0_alip = math.sqrt(g_gravity / H_com_robot)
        # # initial_Lx_world = initial_com_y_rel_to_f1 * mass_robot * H_com_robot * omega0_alip # Should be mass_robot * g_gravity * y_com_rel
        # # initial_Ly_world = -initial_com_x_rel_to_f1 * mass_robot * H_com_robot * omega0_alip # Should be mass_robot * g_gravity * x_com_rel
        # initial_Lx_world = initial_com_y_rel_to_f1 * mass_robot * g_gravity # From L_x_dot = -mgy
        # initial_Ly_world = initial_com_x_rel_to_f1 * mass_robot * g_gravity # From L_y_dot = mgx
        
        # x_alip_start_of_ss_ref_gen = np.array([
        #     initial_com_x_rel_to_f1, initial_com_y_rel_to_f1,
        #     initial_Lx_world, initial_Ly_world
        # ])

        # mid_level_ref_alip_states_per_ss = [] # List to store ALIP states for each SS phase of MPC horizon

        # # The mid-level reference generation will use the footsteps from the global plan
        # # It plans for N_global - 2 "moves" if the first two global steps are fixed stance.
        # # current_stance_foot_global_idx = 1 (i.e., footsteps_global[1] is the first active stance foot)
        
        # # We need to generate enough reference for the MPC horizon (N_horizon_mpc)
        # # Let's assume N_horizon_mpc is, for example, 3 steps.
        # N_horizon_mpc_val = 3 # Example MPC horizon
        # if N_global -1 < N_horizon_mpc_val: # Need at least N_horizon_mpc footsteps *after* the initial one
        #      print(f"Warning: Global plan too short ({N_global-1} steps available) for MPC horizon ({N_horizon_mpc_val}). Adjusting MPC horizon.")
        #      N_horizon_mpc_val = max(1, N_global - 2)


        # active_stance_foot_global_idx = 1 # Start with global_footsteps[1] as the active stance foot
        #                                   # for the first segment of the reference trajectory.
        #                                   # The ALIP state x_alip_start_of_ss_ref_gen is relative to this foot.

        # for i_mpc_step in range(N_horizon_mpc_val):
        #     if active_stance_foot_global_idx + 1 >= N_global:
        #         print(f"Mid-level ref gen: Ran out of global footsteps to define delta_p. Stopping ref gen at MPC step {i_mpc_step}.")
        #         break # Not enough global footsteps to define the next step's delta_p

        #     current_stance_foot_world = np.array([
        #         footsteps_x_global[active_stance_foot_global_idx],
        #         footsteps_y_global[active_stance_foot_global_idx],
        #         footsteps_z_global[active_stance_foot_global_idx]
        #     ])
        #     next_stance_foot_world = np.array([
        #         footsteps_x_global[active_stance_foot_global_idx + 1],
        #         footsteps_y_global[active_stance_foot_global_idx + 1],
        #         footsteps_z_global[active_stance_foot_global_idx + 1]
        #     ])
        #     delta_p_world_ref_gen = next_stance_foot_world - current_stance_foot_world

        #     knots_for_this_ss_phase = []
        #     x_alip_knot_current = np.copy(x_alip_start_of_ss_ref_gen) # Relative to current_stance_foot_world
        #     knots_for_this_ss_phase.append(np.copy(x_alip_knot_current))

        #     for _ in range(K_knots_mpc - 1):
        #         x_alip_knot_current = A_d_knot_auton_alip @ x_alip_knot_current
        #         knots_for_this_ss_phase.append(np.copy(x_alip_knot_current))
            
        #     mid_level_ref_alip_states_per_ss.append(knots_for_this_ss_phase)

        #     # Prepare for next iteration: Reset ALIP state to be relative to next_stance_foot_world
        #     x_alip_at_end_of_ss_wrt_current_stance = x_alip_knot_current
        #     x_alip_start_of_ss_ref_gen = Ar_ds_reset @ x_alip_at_end_of_ss_wrt_current_stance + \
        #                                  Br_delta_p_reset @ delta_p_world_ref_gen
            
        #     active_stance_foot_global_idx += 1


        # # --- Part 3: Low-level MPC (Setup only, not a full MPC loop) ---
        # if not mid_level_ref_alip_states_per_ss:
        #     print("Error: No reference ALIP states generated by mid-level. MPC cannot be set up.")
        # else:
        #     m_mpc = Model("footstep_mpc")
        #     m_mpc.Params.OutputFlag = 0 # Suppress Gurobi output for MPC

        #     # MPC plans N_horizon_mpc_val steps.
        #     # x_alip_vars_mpc[i_step][k_knot]
        #     x_alip_vars_mpc = []
        #     u_ankle_vars_mpc = [] # u_ankle_vars_mpc[i_step][k_knot_interval]

        #     for i_step in range(N_horizon_mpc_val):
        #         knots_vars_this_step = []
        #         controls_vars_this_step = []
        #         for k_knot in range(K_knots_mpc):
        #             knots_vars_this_step.append(m_mpc.addVars(4, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_mpc_s{i_step}_k{k_knot}"))
        #             if k_knot < K_knots_mpc - 1:
        #                 # Define bounds for ankle torque based on your robot's capabilities
        #                 max_ankle_torque_abs = 80 # Example N*m, needs scaling if u is not raw torque
        #                 controls_vars_this_step.append(m_mpc.addVar(lb=-max_ankle_torque_abs, ub=max_ankle_torque_abs, name=f"u_mpc_s{i_step}_k{k_knot}"))
        #         x_alip_vars_mpc.append(knots_vars_this_step)
        #         if controls_vars_this_step:
        #             u_ankle_vars_mpc.append(controls_vars_this_step)
            
        #     # p_foot_vars_mpc[i_step] is the (i_step)-th *planned* footstep by MPC (p1_mpc, p2_mpc, ...)
        #     # This is the location of the *next* stance foot for step i_step.
        #     p_foot_vars_mpc = [m_mpc.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_mpc_s{i_step}") for i_step in range(N_horizon_mpc_val)]

        #     # Binary variables for foothold region selection in MPC
        #     H_vars_mpc = [m_mpc.addVars(N_REG_global, vtype=GRB.BINARY, name=f"H_mpc_s{i_step}") for i_step in range(N_horizon_mpc_val)]


        #     # MPC Initial State Constraint
        #     # This x_robot_current_alip should be the *actual current* ALIP state of the robot.
        #     # For this script, we use the start of the reference trajectory as the initial state for the first MPC solve.
        #     x_robot_current_alip_val = mid_level_ref_alip_states_per_ss[0][0] # First state of first SS phase
        #     for i_dim in range(4):
        #         m_mpc.addConstr(x_alip_vars_mpc[0][0][i_dim] == x_robot_current_alip_val[i_dim])

        #     # MPC Dynamics Constraints (ALIP evolution within each SS phase)
        #     A_d_mpc_dyn, B_d_mpc_dyn = get_alip_matrices_with_input(H_com_robot, mass_robot, g_gravity, T_ss_dt_mpc)
        #     for i_step in range(N_horizon_mpc_val):
        #         for k_knot in range(K_knots_mpc - 1):
        #             for i_row in range(4):
        #                 m_mpc.addConstr(
        #                     x_alip_vars_mpc[i_step][k_knot+1][i_row] ==
        #                     quicksum(A_d_mpc_dyn[i_row, i_col] * x_alip_vars_mpc[i_step][k_knot][i_col] for i_col in range(4)) +
        #                     B_d_mpc_dyn[i_row, 0] * u_ankle_vars_mpc[i_step][k_knot]
        #                 )
            
        #     # MPC Reset Map Constraints (between SS phases)
        #     # p_robot_current_stance_foot_world: The foot the robot is *currently* standing on for the *first* MPC step.
        #     # For this script, it's footsteps_global[active_stance_foot_global_idx_at_mpc_start]
        #     # active_stance_foot_global_idx_at_mpc_start was 1 if the ref gen started from f_g[1].
        #     p_robot_current_stance_foot_world = np.array([
        #         footsteps_x_global[1], footsteps_y_global[1], footsteps_z_global[1]
        #     ])

        #     for i_step in range(N_horizon_mpc_val - 1): # N_horizon_mpc_val - 1 resets
        #         # Stance foot for ALIP states in x_alip_vars_mpc[i_step]
        #         if i_step == 0:
        #             p_old_stance_mpc_x = p_robot_current_stance_foot_world[0]
        #             p_old_stance_mpc_y = p_robot_current_stance_foot_world[1]
        #             p_old_stance_mpc_z = p_robot_current_stance_foot_world[2]
        #         else:
        #             p_old_stance_mpc_x = p_foot_vars_mpc[i_step-1][0] # Previous MPC planned foot
        #             p_old_stance_mpc_y = p_foot_vars_mpc[i_step-1][1]
        #             p_old_stance_mpc_z = p_foot_vars_mpc[i_step-1][2]
                
        #         # New stance foot that MPC is planning for this reset
        #         p_new_stance_mpc_x = p_foot_vars_mpc[i_step][0]
        #         p_new_stance_mpc_y = p_foot_vars_mpc[i_step][1]
        #         p_new_stance_mpc_z = p_foot_vars_mpc[i_step][2]

        #         dp_x_mpc = p_new_stance_mpc_x - p_old_stance_mpc_x
        #         dp_y_mpc = p_new_stance_mpc_y - p_old_stance_mpc_y
        #         dp_z_mpc = p_new_stance_mpc_z - p_old_stance_mpc_z

        #         x_alip_end_of_ss_mpc = x_alip_vars_mpc[i_step][K_knots_mpc-1] # ALIP state at end of SS, wrt p_old_stance_mpc
        #         x_alip_start_of_next_ss_mpc = x_alip_vars_mpc[i_step+1][0]   # ALIP state at start of next SS, wrt p_new_stance_mpc

        #         for i_row in range(4):
        #             br_term_mpc = Br_delta_p_reset[i_row,0] * dp_x_mpc + \
        #                           Br_delta_p_reset[i_row,1] * dp_y_mpc + \
        #                           Br_delta_p_reset[i_row,2] * dp_z_mpc
        #             m_mpc.addConstr(
        #                 x_alip_start_of_next_ss_mpc[i_row] ==
        #                 quicksum(Ar_ds_reset[i_row, i_col] * x_alip_end_of_ss_mpc[i_col] for i_col in range(4)) +
        #                 br_term_mpc
        #             )

        #     # MPC Foothold Region Constraints (simplified, adapt A_coeffs, b_coeffs for 3D p_foot_vars_mpc)
        #     M_mpc_region = M_global_region # Reuse, but might need different M for MPC
        #     for i_step in range(N_horizon_mpc_val):
        #         p_foot_var_this_step = p_foot_vars_mpc[i_step]
        #         for i_reg, R_geom_mpc in enumerate(regions_geom_global): # Using global regions for MPC too
        #             Rx_max, Rx_min, Ry_max, Ry_min = R_geom_mpc
        #             # Simplified for X, Y only. Z needs terrain model.
        #             m_mpc.addConstr(p_foot_var_this_step[0] <= Rx_max + M_mpc_region * (1 - H_vars_mpc[i_step][i_reg]))
        #             m_mpc.addConstr(p_foot_var_this_step[0] >= Rx_min - M_mpc_region * (1 - H_vars_mpc[i_step][i_reg]))
        #             m_mpc.addConstr(p_foot_var_this_step[1] <= Ry_max + M_mpc_region * (1 - H_vars_mpc[i_step][i_reg]))
        #             m_mpc.addConstr(p_foot_var_this_step[1] >= Ry_min - M_mpc_region * (1 - H_vars_mpc[i_step][i_reg]))
        #             # m_mpc.addConstr(p_foot_var_this_step[2] == 0.0) # Example: force Z=0 if flat terrain
        #         m_mpc.addConstr(quicksum(H_vars_mpc[i_step][j] for j in range(N_REG_global)) == 1)


        #     # MPC Kinematic Reachability Constraints
        #     max_dx_mpc, max_dy_mpc, max_dz_mpc = 0.5, 0.4, 0.15 # Example limits for MPC steps
            
        #     # First MPC step is from p_robot_current_stance_foot_world to p_foot_vars_mpc[0]
        #     dp_kin_x0 = p_foot_vars_mpc[0][0] - p_robot_current_stance_foot_world[0]
        #     dp_kin_y0 = p_foot_vars_mpc[0][1] - p_robot_current_stance_foot_world[1]
        #     dp_kin_z0 = p_foot_vars_mpc[0][2] - p_robot_current_stance_foot_world[2]
        #     m_mpc.addConstr(dp_kin_x0 <= max_dx_mpc); m_mpc.addConstr(dp_kin_x0 >= -max_dx_mpc)
        #     m_mpc.addConstr(dp_kin_y0 <= max_dy_mpc); m_mpc.addConstr(dp_kin_y0 >= -max_dy_mpc)
        #     m_mpc.addConstr(dp_kin_z0 <= max_dz_mpc); m_mpc.addConstr(dp_kin_z0 >= -max_dz_mpc)

        #     for i_step in range(N_horizon_mpc_val - 1):
        #         dp_kin_x = p_foot_vars_mpc[i_step+1][0] - p_foot_vars_mpc[i_step][0]
        #         dp_kin_y = p_foot_vars_mpc[i_step+1][1] - p_foot_vars_mpc[i_step][1]
        #         dp_kin_z = p_foot_vars_mpc[i_step+1][2] - p_foot_vars_mpc[i_step][2]
        #         m_mpc.addConstr(dp_kin_x <= max_dx_mpc); m_mpc.addConstr(dp_kin_x >= -max_dx_mpc)
        #         m_mpc.addConstr(dp_kin_y <= max_dy_mpc); m_mpc.addConstr(dp_kin_y >= -max_dy_mpc)
        #         m_mpc.addConstr(dp_kin_z <= max_dz_mpc); m_mpc.addConstr(dp_kin_z >= -max_dz_mpc)


        #     # MPC Objective Function
        #     Q_state_mpc = np.diag([10.0, 10.0, 1.0, 1.0]) # Weights for [x_com_r, y_com_r, Lx_r, Ly_r]
        #     R_input_mpc = 0.001                            # Weight for ankle torque
        #     Q_foot_ref_track = np.diag([0.1, 0.1, 0.01])   # Weight for tracking global footsteps (optional)

        #     objective_mpc = QuadExpr(0)

        #     for i_step in range(N_horizon_mpc_val):
        #         # ALIP state tracking cost
        #         for k_knot in range(K_knots_mpc):
        #             x_ref_knot = mid_level_ref_alip_states_per_ss[i_step][k_knot] # This is a numpy array
        #             for i_dim in range(4):
        #                 error_state = x_alip_vars_mpc[i_step][k_knot][i_dim] - x_ref_knot[i_dim]
        #                 objective_mpc += error_state * Q_state_mpc[i_dim, i_dim] * error_state
        #         # Ankle torque cost
        #         if i_step < len(u_ankle_vars_mpc): # Check if u_ankle_vars_mpc has this step
        #             for k_interval in range(K_knots_mpc - 1):
        #                 u_var = u_ankle_vars_mpc[i_step][k_interval]
        #                 objective_mpc += u_var * R_input_mpc * u_var
                
        #         # Optional: Cost for p_foot_vars_mpc[i_step] to be close to global plan footsteps
        #         # The corresponding global footstep for p_foot_vars_mpc[i_step] (which is p_i_step+1_mpc)
        #         # would be footsteps_global[current_stance_idx_for_mpc_start + 1 + i_step]
        #         # current_stance_idx_for_mpc_start is 1 in this setup
        #         global_foot_target_idx = 1 + 1 + i_step
        #         if global_foot_target_idx < N_global:
        #             err_px = p_foot_vars_mpc[i_step][0] - footsteps_x_global[global_foot_target_idx]
        #             err_py = p_foot_vars_mpc[i_step][1] - footsteps_y_global[global_foot_target_idx]
        #             err_pz = p_foot_vars_mpc[i_step][2] - footsteps_z_global[global_foot_target_idx] # Assumes z=0
        #             objective_mpc += (err_px*err_px * Q_foot_ref_track[0,0] +
        #                               err_py*err_py * Q_foot_ref_track[1,1] +
        #                               err_pz*err_pz * Q_foot_ref_track[2,2])


        #     # MPC Terminal state cost (optional)
        #     if N_horizon_mpc_val > 0 and mid_level_ref_alip_states_per_ss:
        #          Q_f_state_mpc = Q_state_mpc * 1.0 # Terminal state weight
        #          x_final_actual_mpc = x_alip_vars_mpc[N_horizon_mpc_val-1][K_knots_mpc-1]
        #          x_final_desired_mpc = mid_level_ref_alip_states_per_ss[N_horizon_mpc_val-1][K_knots_mpc-1]
        #          for i_dim in range(4):
        #              error_final_state = x_final_actual_mpc[i_dim] - x_final_desired_mpc[i_dim]
        #              objective_mpc += error_final_state * Q_f_state_mpc[i_dim, i_dim] * error_final_state

        #     m_mpc.setObjective(objective_mpc, GRB.MINIMIZE)
        #     m_mpc.Params.TimeLimit = 0.05 # MPC needs to be very fast
        #     # m_mpc.optimize()
        #     # print(f"MPC Gurobi optimization time: {m_mpc.Runtime:.4f} seconds")
        #     # MPC results would be extracted here in a real loop.


        # # --- Plotting (uses results from the *first* global planner) ---
        # fig = plt.figure(figsize=(8, 8))
        # ax1 = fig.add_subplot(1,1,1)
        # ax1.set(xlim=(-1, 3.5), ylim=(-1, 3.5)) # Adjusted limits

        # # Plot global footsteps
        # ax1.plot(footsteps_x_global[0], footsteps_y_global[0], 'bo', markersize=8, label="Start L Foot (Global)")
        # ax1.plot(footsteps_x_global[1], footsteps_y_global[1], 'ro', markersize=8, label="Start R Foot (Global)") # Changed to 'ro'
        # ax1.arrow(footsteps_x_global[0], footsteps_y_global[0], 0.25*math.cos(footsteps_theta_global[0]), 0.25*math.sin(footsteps_theta_global[0]), head_width=0.05, fc='blue', ec='blue')
        # ax1.arrow(footsteps_x_global[1], footsteps_y_global[1], 0.25*math.cos(footsteps_theta_global[1]), 0.25*math.sin(footsteps_theta_global[1]), head_width=0.05, fc='red', ec='red')

        # for c in range(2, N_global):
        #     color = 'bo' if c % 2 == 0 else 'ro' # Assume F0, F2... are Left (blue), F1, F3... are Right (red)
        #                                         # Given F0, F1 are fixed, F2 is first planned by global.
        #                                         # If F0=L, F1=R, then F2=L, F3=R
        #     marker = 'o' if c%2 == 0 else '*' # Example different marker for R
        #     ax1.plot(footsteps_x_global[c], footsteps_y_global[c], color=color[0], marker=marker, markersize=6)
        #     ax1.arrow(footsteps_x_global[c], footsteps_y_global[c], 0.15*math.cos(footsteps_theta_global[c]), 0.15*math.sin(footsteps_theta_global[c]), head_width=0.04, fc=color[0], ec=color[0])

        # # Plot CoM reference from mid-level (if generated)
        # if mid_level_ref_alip_states_per_ss:
        #     com_x_ref_plot = []
        #     com_y_ref_plot = []
        #     plot_stance_foot_idx = 1 # The mid_level_ref_alip_states_per_ss[0] is relative to global_footsteps[1]
        #     for i_ss_phase, knots_data in enumerate(mid_level_ref_alip_states_per_ss):
        #         stance_foot_x = footsteps_x_global[plot_stance_foot_idx + i_ss_phase]
        #         stance_foot_y = footsteps_y_global[plot_stance_foot_idx + i_ss_phase]
        #         for alip_state_rel in knots_data:
        #             com_x_ref_plot.append(alip_state_rel[0] + stance_foot_x)
        #             com_y_ref_plot.append(alip_state_rel[1] + stance_foot_y)
        #     ax1.plot(com_x_ref_plot, com_y_ref_plot, 'g--', linewidth=1.5, label="Mid-Level CoM Ref (Ideal ALIP)")


        # # Plot safe regions
        # for i_reg, R_geom_plot in enumerate(regions_geom_global):
        #     Rx_max, Rx_min, Ry_max, Ry_min = R_geom_plot
        #     rect = patches.Rectangle((Rx_min, Ry_min), Rx_max-Rx_min, Ry_max-Ry_min, linewidth=1, edgecolor='gray', facecolor='lightgreen', alpha=0.3)
        #     ax1.add_patch(rect)
        #     ax1.text(region_midpts_global[i_reg][0], region_midpts_global[i_reg][1], f"R{i_reg+1}", fontsize=9, ha='center', va='center')

        # ax1.set_xlabel("X coordinate")
        # ax1.set_ylabel("Y coordinate")
        # ax1.set_title("Footstep Planning Results")
        # ax1.legend()
        # ax1.grid(True)
        # ax1.axis('equal')
        # plt.show()

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ":" + str(e))
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()