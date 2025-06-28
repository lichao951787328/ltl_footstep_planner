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
    A_d_for_mpc =  expm(A_c_autonomous * T_ss_dt)
    B_d_for_mpc = np.linalg.inv(A_c_autonomous) @ (A_d_for_mpc - np.eye(A_c_autonomous.shape[0])) @ B_c_input_effect
    return A_d_for_mpc, B_d_for_mpc

def get_alip_matrices(H_com, mass, g, T_ss_dt):
    A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g)
    A_d_for_mpc =  expm(A_c_autonomous * T_ss_dt)
    return A_d_for_mpc

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

# 使用向量差的模L2 Norm of the Difference
# Sum of squared L2 norms of first-order differences (一阶差分L2范数平方和):

if __name__ == "__main__":
	try:
		# Create a new model
		m = Model("footstep")

		# Create variables
		# Create N footstep variables (x,y,theta,s,c)
		N = 18
		footsteps = [m.addVars(5,lb=-5,name="F"+str(i)) for i in range(0,N)]

		# Trig approx functions
		S = [m.addVars(5,vtype=GRB.BINARY, name="S"+str(i)) for i in range(0,N)]
		C = [m.addVars(5,vtype=GRB.BINARY, name="C"+str(i)) for i in range(0,N)]

		# Safe Regions
		N_REG = 5
		H = [m.addVars(N_REG,vtype=GRB.BINARY, name="H"+str(i)) for i in range(0,N)]

		# Set constraints
		
		# DEFINE REGIONS
		## SCENARIOS 1,2
		# R1_xmax = 1
		# R1_xmin = 0
		# R1_ymax = 1
		# R1_ymin = 0
		# R1_midpt = [(R1_xmax + R1_xmin)/2 , (R1_ymax + R1_ymin)/2]

		# R2_xmax = 1.6
		# R2_xmin = 1.1
		# R2_ymax = 0.85
		# R2_ymin = -0.5
		# R2_midpt = [(R2_xmax + R2_xmin)/2 , (R2_ymax + R2_ymin)/2]

		# R3_xmax = 2.2
		# R3_xmin = 1.65
		# R3_ymax = 1.8
		# R3_ymin = 0
		# R3_midpt = [(R3_xmax + R3_xmin)/2 , (R3_ymax + R3_ymin)/2]

		# R4_xmax = 2.35
		# R4_xmin = 1.2
		# R4_ymax = 2.5
		# R4_ymin = 1.85
		# R4_midpt = [(R4_xmax + R4_xmin)/2 , (R4_ymax + R4_ymin)/2]

		# R5_xmax = 1
		# R5_xmin = -0.5
		# R5_ymax = 2
		# R5_ymin = 1.1
		# R5_midpt = [(R5_xmax + R5_xmin)/2 , (R5_ymax + R5_ymin)/2]

		## SCENARIO 3
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

		### EXTRA REGIONS ###

		# R6_xmax = 0
		# R6_xmin = -1
		# R6_ymax = 4
		# R6_ymin = 3.1

		# R7_xmax = 2.5
		# R7_xmin = 1
		# R7_ymax = 3.5
		# R7_ymin = 2.6

		# R8_xmax = 2.5
		# R8_xmin = 2
		# R8_ymax = 2.6
		# R8_ymin = 1.5


		A_1 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_1 = [R1_xmax,-R1_xmin,R1_ymax,-R1_ymin,math.pi,math.pi/2]
		print(A_1[1][2])
		
		A_2 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_2 = [R2_xmax,-R2_xmin,R2_ymax,-R2_ymin,math.pi,math.pi/2]
		
		A_3 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_3 = [R3_xmax,-R3_xmin,R3_ymax,-R3_ymin,math.pi,math.pi/2]

		A_4 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_4 = [R4_xmax,-R4_xmin,R4_ymax,-R4_ymin,math.pi,math.pi/2]

		A_5 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_5 = [R5_xmax,-R5_xmin,R5_ymax,-R5_ymin,math.pi,math.pi/2]

		# A_6 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_6 = [R6_xmax,-R6_xmin,R6_ymax,-R6_ymin,math.pi,math.pi/2]

		# A_7 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_7 = [R7_xmax,-R7_xmin,R7_ymax,-R7_ymin,math.pi,math.pi/2]

		# A_8 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		# b_8 = [R8_xmax,-R8_xmin,R8_ymax,-R8_ymin,math.pi,math.pi/2]

		# All footsteps must be in the regions
		for c in range(0,N):
			for i in range(0,len(A_1)):
				M = 20
				# Region 1
				m.addConstr(-M*(1-H[c][0]) + quicksum(A_1[i][j]*footsteps[c][j] for j in range(0,3)) - b_1[i] <= 0)
				# Region 2
				m.addConstr(-M*(1-H[c][1]) + quicksum(A_2[i][j]*footsteps[c][j] for j in range(0,3)) - b_2[i] <= 0)
				# Region 3
				m.addConstr(-M*(1-H[c][2]) + quicksum(A_3[i][j]*footsteps[c][j] for j in range(0,3)) - b_3[i] <= 0)
				# Region 4
				m.addConstr(-M*(1-H[c][3]) + quicksum(A_4[i][j]*footsteps[c][j] for j in range(0,3)) - b_4[i] <= 0)
				# Region 5
				m.addConstr(-M*(1-H[c][4]) + quicksum(A_5[i][j]*footsteps[c][j] for j in range(0,3)) - b_5[i] <= 0)
				# # Region 6
				# m.addConstr(-M*(1-H[c][5]) + quicksum(A_6[i][j]*footsteps[c][j] for j in range(0,3)) - b_6[i] <= 0)
				# # Region 7
				# m.addConstr(-M*(1-H[c][6]) + quicksum(A_7[i][j]*footsteps[c][j] for j in range(0,3)) - b_7[i] <= 0)
				# # Region 8
				# m.addConstr(-M*(1-H[c][7]) + quicksum(A_8[i][j]*footsteps[c][j] for j in range(0,3)) - b_8[i] <= 0)

			# Constraint that the sum of H must be 1 for every foothold
			m.addConstr(quicksum(H[c][j] for j in range(0,N_REG)) == 1 )

		#Reachability constraint
		for c in range(2,N):
			# if odd after f1,f2 (fixed), so f3, f5, f7, ...
			# Let's say odd is finding a step for right leg
			if (c % 2 != 0):
				p1 = [0,0.1]
				p2 = [0,-0.8]
				d1 = 0.55
				d2 = 0.55
				xn = footsteps[c][0]
				yn = footsteps[c][1]
				xc = footsteps[c-1][0]
				yc = footsteps[c-1][1]
				thetac = footsteps[c-1][2]
				term1_a = xn - (xc + p1[0]*footsteps[c-1][4] - p1[1]*footsteps[c-1][3])
				term2_a = yn - (yc + p1[0]*footsteps[c-1][3] + p1[1]*footsteps[c-1][4])
				# term1_a = xn - (xc + p1[0])
				# term2_a = yn - (yc + p1[1])
				m.addQConstr(term1_a*term1_a + term2_a*term2_a <= d1*d1)

				# term1_b = xn - (xc + p2[0])
				# term2_b = yn - (yc + p2[1])
				term1_b = xn - (xc + p2[0]*footsteps[c-1][4] - p2[1]*footsteps[c-1][3])
				term2_b = yn - (yc + p2[0]*footsteps[c-1][3] + p2[1]*footsteps[c-1][4])
				m.addQConstr(term1_b*term1_b + term2_b*term2_b <= d2*d2)
			else:
				# finding step for left leg
				print("OTHER LEG")
				p1 = [0, -0.1]
				p2 = [0,0.8]
				d1 = 0.55
				d2 = 0.55
				xn = footsteps[c][0]
				yn = footsteps[c][1]
				xc = footsteps[c-1][0]
				yc = footsteps[c-1][1]
				thetac = footsteps[c-1][2]
				term1 = xn - (xc + p1[0]*footsteps[c-1][4] - p1[1]*footsteps[c-1][3])
				term2 = yn - (yc + p1[0]*footsteps[c-1][3] + p1[1]*footsteps[c-1][4])
				# term1 = xn - (xc + p1[0])
				# term2 = yn - (yc + p1[1])
				m.addQConstr(term1*term1 + term2*term2 <= d1*d1)

				# term1 = xn - (xc + p2[0])
				# term2 = yn - (yc + p2[1])
				term1 = xn - (xc + p2[0]*footsteps[c-1][4] - p2[1]*footsteps[c-1][3])
				term2 = yn - (yc + p2[0]*footsteps[c-1][3] + p2[1]*footsteps[c-1][4])
				m.addQConstr(term1*term1 + term2*term2 <= d2*d2)

		# Add constraints for sin
		for c in range(0,N):
			for i in range(0,5):
				M = 20
				if i == 0:
					phi_l = -math.pi
					phi_lp1 = 1-math.pi
					g_l = -1
					h_l = -math.pi
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 1:
					phi_l = 1-math.pi
					phi_lp1 = -1
					g_l = 0
					h_l = -1
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 2:
					phi_l = -1
					phi_lp1 = 1
					g_l = 1
					h_l = 0
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 3:
					phi_l = 1
					phi_lp1 = math.pi-1
					g_l = 0
					h_l = 1
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 4:
					phi_l = math.pi-1
					phi_lp1 = math.pi
					g_l = -1
					h_l = math.pi
					m.addConstr(-(1-S[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-S[c][i])*M - footsteps[c][3] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-S[c][i])*M + footsteps[c][3] - g_l*footsteps[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
			m.addConstr(quicksum(S[c][j] for j in range(0,5)) == 1 )

		# Add constraints for cos
		for c in range(0,N):
			for i in range(0,5):
				M = 20
				if i == 0:
					phi_l = -math.pi
					phi_lp1 = -1-math.pi/2
					g_l = 0
					h_l = -1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 1:
					phi_l = -1-math.pi/2
					phi_lp1 = 1-math.pi/2
					g_l = 1
					h_l = math.pi/2
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 2:
					phi_l = 1-math.pi/2
					phi_lp1 = math.pi/2-1
					g_l = 0
					h_l = 1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 3:
					phi_l = math.pi/2-1
					phi_lp1 = math.pi/2+1
					g_l = -1
					h_l = math.pi/2
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
				elif i == 4:
					phi_l = math.pi/2+1
					phi_lp1 = math.pi
					g_l = 0
					h_l = -1
					m.addConstr(-(1-C[c][i])*M - phi_lp1 + footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M + phi_l - footsteps[c][2] <= 0)
					m.addConstr(-(1-C[c][i])*M - footsteps[c][4] + g_l*footsteps[c][2] + h_l <= 0)
					m.addConstr(-(1-C[c][i])*M + footsteps[c][4] - g_l*footsteps[c][2] - h_l <= 0)
			
			# Constraint that the sum of S must be 1 for every foothold
			m.addConstr(quicksum(C[c][j] for j in range(0,5)) == 1 )

		# # Set first two footholds
		init_theta = 0
		f1_s = math.sin(init_theta)
		f1_c = math.cos(init_theta)
		m.addConstr(footsteps[0][0] == 0)
		m.addConstr(footsteps[0][1] == 0.4)
		m.addConstr(footsteps[0][3] == f1_s)
		m.addConstr(footsteps[0][4] == f1_c)
		m.addConstr(S[0][2] == 1)
		m.addConstr(C[0][2] == 1)
		
		m.addConstr(footsteps[1][0] == 0)
		m.addConstr(footsteps[1][1] == 0)
		m.addConstr(footsteps[1][2] == init_theta)
		m.addConstr(footsteps[1][3] == f1_s)
		m.addConstr(footsteps[1][4] == f1_c)
		m.addConstr(S[1][2] == 1)
		m.addConstr(C[1][2] == 1)
  		
		small_displacement_threshold = 0.15
		T_step = 1 # Time step duration
		dx_init = footsteps[2][0] - footsteps[0][0]
		dy_init = footsteps[2][1] - footsteps[0][1]
		m.addQConstr(dx_init*dx_init + dy_init*dy_init <= (small_displacement_threshold * T_step)**2, name="init_step_displacement_sq")

		dx_final = footsteps[N-1][0] - footsteps[N-3][0]
		dy_final = footsteps[N-1][1] - footsteps[N-3][1]
		m.addQConstr(dx_final*dx_final + dy_final*dy_final <= (small_displacement_threshold * T_step)**2, name="final_step_displacement_sq")


		# Add constraint of how much foot can rotate in one step
		for c in range(1,N):
			del_theta_max = math.pi/8
			m.addConstr((footsteps[c][2] - footsteps[c-1][2]) <= del_theta_max)
			m.addConstr( (footsteps[c][2] - footsteps[c-1][2]) >= -del_theta_max)

		#########################################################
		####################### SPECS ###########################
		#########################################################

		########## SCENARIO 1 ##############
		# # Visit region 3 or 4 eventually
		# m.addConstr(quicksum(H[j][2] for j in range(0,N)) + quicksum(H[j][3] for j in range(0,N)) >= 1)


		########## SCENARIO 2 ############### 
		# Until
		# Always be in regions 2 or 3 before entering region 4
		# reg1 = 0 # region 1
		# reg2 = 1 # region 2
		# phi2_reg = 2 # region 43

		# T = [m.addVar(vtype=GRB.BINARY, name="T"+str(i)) for i in range(0,N)]

		# # Base case
		# m.addConstr(T[N-1] == H[N-1][phi2_reg])

		# # Satisfiability constraint
		# # m.addConstr(quicksum(T[j] for j in range(0,N)) == N)
		# m.addConstr(T[0] == 1)

		# Pphi1 = [m.addVar(vtype=GRB.BINARY, name="Pphi1"+str(i)) for i in range(0,N-1)]
		# B = [m.addVar(vtype=GRB.BINARY, name="B"+str(i)) for i in range(0,N-1)]

		# # Recursive constraints
		# for i in range(0,N-1):
		# 	M = 20
		# 	delta = 0.001
			
		# 	m.addConstr(H[i][reg1] + H[i][reg2] - 1 >= -M*(1-Pphi1[i]))
		# 	m.addConstr(H[i][reg1] + H[i][reg2] - 1 + delta <=  M*(Pphi1[i]))

		# 	# Term in parenthesis
		# 	m.addConstr(Pphi1[i] + T[i+1] - 2 >= -M*(1-B[i]))
		# 	m.addConstr(Pphi1[i] + T[i+1] - 2 + delta <= M*(B[i]))

		# 	# Final constraint
		# 	m.addConstr(H[i][phi2_reg] + B[i] - 1 >= -M*(1-T[i]))
		# 	m.addConstr(H[i][phi2_reg] + B[i] - 1 + delta <= M*(T[i]))

		########## SCENARIO 3 ##############
		# ni = 7
		# nf = 15
		# m.addConstr(quicksum(H[i][1] for i in range(ni-1,nf)) >= nf-ni+1)

		# Set objective
		# g = [2,1.5,math.pi/2]
		g = [1.5,2.2,3*math.pi/4] # Scenario 3
		e0 = footsteps[N-1][0]-g[0] 
		e1 = footsteps[N-1][1]-g[1] 
		e2 = footsteps[N-1][2]-g[2] 
		Q = [[300,0,0],[0,300,0],[0,0,300]]

		term_cost = (e0)*(e0)*Q[0][0] + (e0)*(e1)*Q[1][0] + (e2)*(e0)*Q[2][0] + (e0)*(e1)*Q[0][1]\
			+(e1)*(e1)*Q[1][1]+(e2)*(e1)*Q[1][2]+(e0)*(e2)*Q[2][0]+(e1)*(e2)*Q[2][1]+(e2)*(e2)*Q[2][2]


		# Calculate incremental costs
		R = [[0.5,0,0],[0,0.5,0],[0,0,0.5]]
		inc_cost = quicksum((footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][0]-footsteps[j-1][0])*R[0][0] + (footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][1]-footsteps[j-1][1])*R[1][0]\
			+(footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][0]-footsteps[j-1][0])*R[2][0] + (footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][1]-footsteps[j-1][1])*R[0][1]\
			+(footsteps[j][1]-footsteps[j-1][1])*(footsteps[j][1]-footsteps[j-1][1])*R[1][1] + (footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][1]-footsteps[j-1][1])*R[1][2]\
			+(footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][2]-footsteps[j-1][2])*R[2][0] + (footsteps[j][1]-footsteps[j-1][1])*(footsteps[j][2]-footsteps[j-1][2])*R[2][1]\
			+(footsteps[j][2]-footsteps[j-1][2])*(footsteps[j][2]-footsteps[j-1][2])*R[2][2] for j in range(0,N))


		# 增加速度平方和的代价，保证速度的平滑性。同时设置初始步态和最终步态的速度
		# N步中有N-1个速度变量
		# 第一个速度是由上层控制给定
		# 最后一个速度为0
		# 中间速度为位移/周期
  
		T_cycle = 1.0 # 假设两个中点之间的周期为1，这代表了脚印之间的时间

		# 1. 计算相邻脚印的中点 (N-1 个中点)
		# midpoints[k] 是第 k 和 k+1 个脚印之间的中点 (k from 0 to N-2)
		# footsteps[j] 和 footsteps[j-1] 之间的中点
		midpoints_xy = [] # List of tuples (mid_x_expr, mid_y_expr)
		for j in range(1, N): # j from 1 to N-1, so N-1 midpoints
			# midpoints_xy[j-1] 对应 footsteps[j-1] 和 footsteps[j] 的中点
			mid_x = (footsteps[j][0] + footsteps[j-1][0]) / 2.0
			mid_y = (footsteps[j][1] + footsteps[j-1][1]) / 2.0
			midpoints_xy.append((mid_x, mid_y))

		# 2. 计算中点之间的速度 (N-2 个速度向量)
		# velocities_xy[k] 是 midpoints_xy[k]到midpoints_xy[k+1]的平均速度
		# (k from 0 to N-3)
		velocities_xy = [] # List of tuples (vel_x_expr, vel_y_expr)
		if N > 2: # 至少需要3个脚印才能定义一个中点间的速度
			for j in range(1, len(midpoints_xy)): # j from 1 to N-2 (index for midpoints_xy)
				# velocity between midpoints_xy[j-1] and midpoints_xy[j]
				vel_x = (midpoints_xy[j][0] - midpoints_xy[j-1][0]) / T_cycle
				vel_y = (midpoints_xy[j][1] - midpoints_xy[j-1][1]) / T_cycle
				velocities_xy.append((vel_x, vel_y))

		# 3. 计算速度平滑性的代价 (即加速度的平方和)
		# (N-3 个加速度项，如果 N > 3)
		velocity_smoothness_cost = QuadExpr(0) # Initialize as an empty quadratic expression
		G_vel_smoothness = 5.0 # Weight for velocity smoothness

		if len(velocities_xy) > 1: # 至少需要2个速度向量才能计算它们之间的差
			for j in range(1, len(velocities_xy)): # j from 1 to N-3 (index for velocities_xy)
				# Acceleration_x = velocities_xy[j][0] - velocities_xy[j-1][0]
				# Acceleration_y = velocities_xy[j][1] - velocities_xy[j-1][1]
				# (我们这里假设 T_cycle_accel = 1, 即速度变化也发生在一个周期内)
				accel_x = velocities_xy[j][0] - velocities_xy[j-1][0]
				accel_y = velocities_xy[j][1] - velocities_xy[j-1][1]
				velocity_smoothness_cost += accel_x*accel_x + accel_y*accel_y

		# 更新目标函数
		# m.setObjective(term_cost + G_vel_smoothness * velocity_smoothness_cost, GRB.MINIMIZE)
		m.setObjective(term_cost + inc_cost + G_vel_smoothness * velocity_smoothness_cost, GRB.MINIMIZE)
		# m.setObjective(term_cost + inc_cost + G * velocity_cost, GRB.MINIMIZE)
			  
		# Print the velocities
		# print("Velocities at midpoints:")
		# for idx, velocity in enumerate(velocities):
    	# 		print(f"Step {idx+1}-{idx+2}: {velocity}") / 2
  
		# Print the midpoints
		# print("Midpoints of adjacent footsteps:")
		# for idx, midpoint in enumerate(midpoints):
    	# 		print(f"Step {idx+1}-{idx+2}: {midpoint}") / 2
  
		#inc_cost = quicksum((footsteps[j][0]-footsteps[j-1][0])*(footsteps[j][0]-footsteps[j-1][0])*R[0][0] for j in range(0,N))
		#inc_cost = 0
		# m.setObjective( term_cost + inc_cost + G * velocity_cost
				# \
				# , GRB.MINIMIZE )

		#print(f.values())
		m.Params.MIPFocus = 0
		m.optimize()
		print(f"Gurobi optimization time: {m.Runtime:.4f} seconds") # 打印求解时间
		footsteps_x = []
		footsteps_y = []
		footsteps_theta = []

		# Get x values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[0]")
			footsteps_x.append(v.X)

		# Get y values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[1]")
			footsteps_y.append(v.X)

		# Get theta values
		for c in range(0,N):
			v = m.getVarByName("F"+str(c)+"[2]")
			footsteps_theta.append(v.X)

		# 打印所有点的坐标
		for i in range(0,N):
			print(f"Footstep {i}: x={footsteps_x[i]:.2f}, y={footsteps_y[i]:.2f}, theta={footsteps_theta[i]:.2f}")
        
		###### PLOT ######

		

		# Plot x_idea_all_data first and second arrays
		# if len(x_idea_all_data) > 1:
    	# 		x1 = x_idea_all_data[0][0
		# 	y1 = x_idea_all_data[0][0][1]
		# 	x2 = x_idea_all_data[0][1][0]
		# 	y2 = x_idea_all_data[0][1][1]
		# 	ax1.plot([x1, x2], [y1, y2], 'go-', label="ALIP trajectory (first two points)")

		
		###### PLOT ######

		fig = plt.figure(figsize=(8, 8))
		ax1 = fig.add_subplot(1,1,1)
		ax1.set(xlim=(-2,5), ylim=(-2,5))

		# Plot initial foot stance
		ax1.plot(footsteps_x[0],footsteps_y[0], 'bo')
		# ax1.text(footsteps_x[0]-0.05,footsteps_y[0]+0.02, str(1), fontsize=8, color='blue' )
		ax1.plot(footsteps_x[1],footsteps_y[1], 'r*')
		#ax1.text(footsteps_x[1],footsteps_x[1]-0.02, str(2), fontsize=8, color='red' )
		ax1.arrow(footsteps_x[0],footsteps_y[0],0.25*math.cos(footsteps_theta[0]),0.25*math.sin(footsteps_theta[0]))
		ax1.arrow(footsteps_x[1],footsteps_y[1],0.25*math.cos(footsteps_theta[1]),0.25*math.sin(footsteps_theta[1]))

		# Plot safe region 1
		rect = patches.Rectangle((R1_xmin,R1_ymin),R1_xmax-R1_xmin,R1_ymax-R1_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 2
		rect = patches.Rectangle((R2_xmin,R2_ymin),R2_xmax-R2_xmin,R2_ymax-R2_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 3
		rect = patches.Rectangle((R3_xmin,R3_ymin),R3_xmax-R3_xmin,R3_ymax-R3_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 4
		rect = patches.Rectangle((R4_xmin,R4_ymin),R4_xmax-R4_xmin,R4_ymax-R4_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# Plot safe region 5
		rect = patches.Rectangle((R5_xmin,R5_ymin),R5_xmax-R5_xmin,R5_ymax-R5_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		ax1.add_patch(rect)
		# # Plot safe region 6
		# rect = patches.Rectangle((R6_xmin,R6_ymin),R6_xmax-R6_xmin,R6_ymax-R6_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)
		# # Plot safe region 7
		# rect = patches.Rectangle((R7_xmin,R7_ymin),R7_xmax-R7_xmin,R7_ymax-R7_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)
		# # Plot safe region 8
		# rect = patches.Rectangle((R8_xmin,R8_ymin),R8_xmax-R8_xmin,R8_ymax-R8_ymin,linewidth=1, edgecolor='b',facecolor='green', alpha=0.4)
		# ax1.add_patch(rect)

		# Plot x_idea_all_data
		

		# Plot x_idea_all_data first and second arrays
		# print(f"x_idea_all_data length: {len(x_idea_all_data)}")
		# print("x_idea_all_data size:", np.array(x_idea_all_data).shape)
		# if len(x_idea_all_data) > 1:
		# 	for i in range(1):
		# 		for j in range(K_knots_mpc):
		# 			x1 = x_idea_all_data[i][j][0]
		# 			y1 = x_idea_all_data[i][j][1]
		# 			x2 = x_idea_all_data[i][j][0]
		# 			y2 = x_idea_all_data[i][j][1]
					# ax1.plot([x1, x2], [y1, y2], 'go-', label="ALIP trajectory (between steps)")
    				
		def animate(i):
			if (i % 2 == 0) & (i < len(footsteps_x)-2):
				# It is a left footstep
				cur_x = footsteps_x[i+2]
				cur_y = footsteps_y[i+2]
				cur_theta = footsteps_theta[i+2]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				#bl_x = cur_x
				#bl_y = cur_y
				ax1.plot(cur_x,cur_y,'bo')
				# ax1.text(cur_x-0.18,cur_y-0.1, str(i+3), fontsize=8, color='blue' )
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))

				p1 = [0,0.1]
				p2 = [0,-0.8]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

				# circ1 = patches.Circle((center_x1,center_y1),0.55,linewidth=1, edgecolor='r',facecolor='none')
				# circ2 = patches.Circle((center_x2,center_y2),0.55,linewidth=1, edgecolor='r',facecolor='none')
				# ax1.add_patch(circ1)
				# ax1.add_patch(circ2)
				
			elif (i % 2 != 0) & (i < len(footsteps_x)-2):
				cur_x = footsteps_x[i+2]
				cur_y = footsteps_y[i+2]
				cur_theta = footsteps_theta[i+2]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				# It is a right footstep
				ax1.plot(cur_x,cur_y,'r*')
				#ax1.text(cur_x-0.18,cur_y-0.1, str(i+3), fontsize=8, color='red' )
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				arrow = ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))
				p1 = [0,-0.1]
				p2 = [0,0.8]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

		ani = animation.FuncAnimation(fig, animate, interval=1000)
		ax1.legend(["Left foot", "Right foot"])
		offset = 0.1
		ax1.text(R1_midpt[0]-offset,R1_midpt[1]-offset,"R1")
		ax1.text(R2_midpt[0]-offset,R2_midpt[1]-offset,"R2")
		ax1.text(R3_midpt[0]-offset,R3_midpt[1]-offset,"R3")
		ax1.text(R4_midpt[0]-offset,R4_midpt[1]-offset,"R4")
		ax1.text(R5_midpt[0]-offset,R5_midpt[1]-offset,"R5")

		plt.show()

	except GurobiError as e:
		print('Error code' + str(e.errno)+":"+str(e))