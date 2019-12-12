from gurobipy import *
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

if __name__ == "__main__":
	try:
		# Create a new model
		m = Model("footstep")

		# Create variables
		# Create N footstep variables (x,y,theta,s,c)
		N = 10
		footsteps = [m.addVars(5,lb=-5,name="F"+str(i)) for i in range(0,N)]

		# Trig approx functions
		S = [m.addVars(5,vtype=GRB.BINARY, name="S"+str(i)) for i in range(0,N)]
		C = [m.addVars(5,vtype=GRB.BINARY, name="C"+str(i)) for i in range(0,N)]

		# Set constraints

		# DEFINE REGION 1
		bound1 = 3
		A_1 = [[1, 0, 0],[-1, 0, 0],[0, 1, 0], [0, -1, 0],[0, 0, 1], [0, 0, -1]]
		b_1 = [3,0,3,0,math.pi,math.pi/2]
		print(A_1[1][2])
		# All footsteps must be in region 1
		for c in range(0,N):
			for i in range(0,len(A_1)):
				m.addConstr(quicksum(A_1[i][j]*footsteps[c][j] for j in range(0,3)) <= b_1[i])

		#Reachability constraint
		for c in range(2,N):
			# if odd after f1,f2 (fixed), so f3, f5, f7, ...
			# Let's say odd is finding a step for right leg
			if (c % 2 != 0):
				p1 = [0,0.2]
				p2 = [0,-0.6]
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
				p1 = [0, -0.2]
				p2 = [0,0.6]
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
				M = 1000
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
				M = 1000
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
		m.addConstr(footsteps[0][1] == 0.6)
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

		# Add constraint of how much can turn foot in one step
		for c in range(1,N):
			del_theta_max = math.pi/8
			m.addConstr((footsteps[c][2] - footsteps[c-1][2]) <= del_theta_max)
			m.addConstr( (footsteps[c][2] - footsteps[c-1][2]) >= -del_theta_max)

		# Set objective
		g = [1,0,math.pi]
		e0 = footsteps[N-1][0]-g[0] 
		e1 = footsteps[N-1][1]-g[1] 
		e2 = footsteps[N-1][2]-g[2] 
		Q = [[10,0,0],[0,10,0],[0,0,10]]

		term_cost = (e0)*(e0)*Q[0][0] + (e0)*(e1)*Q[1][0] + (e2)*(e0)*Q[2][0] + (e0)*(e1)*Q[0][1]\
			+(e1)*(e1)*Q[1][1]+(e2)*(e1)*Q[1][2]+(e0)*(e2)*Q[2][0]+(e1)*(e2)*Q[2][1]+(e2)*(e2)*Q[2][2]

		m.setObjective(term_cost
				\
				, GRB.MINIMIZE )

		#print(f.values())
		m.optimize()

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

		print(footsteps_x)
		print(footsteps_y)
		print(footsteps_theta)

		for v in m.getVars():
			print('%s %g' % (v.varName,v.x))

		###### PLOT ######

		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		ax1.set(xlim=(-1,2.5), ylim=(-1,2.5))
		def animate(i):
			if (i % 2 == 0) & (i < len(footsteps_x)):
				# It is a left footstep
				cur_x = footsteps_x[i]
				cur_y = footsteps_y[i]
				cur_theta = footsteps_theta[i]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				#bl_x = cur_x
				#bl_y = cur_y
				ax1.plot(footsteps_x[i],footsteps_y[i],'bo')
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))

				p1 = [0,0.2]
				p2 = [0,-0.6]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

				#circ1 = patches.Circle((center_x1,center_y1),0.55,linewidth=1, edgecolor='r',facecolor='none')
				#circ2 = patches.Circle((center_x2,center_y2),0.55,linewidth=1, edgecolor='r',facecolor='none')
				# ax1.add_patch(circ1)
				# ax1.add_patch(circ2)
				
			elif (i % 2 != 0) & (i < len(footsteps_x)):
				cur_x = footsteps_x[i]
				cur_y = footsteps_y[i]
				cur_theta = footsteps_theta[i]
				print(cur_theta)
				bl = [-0.05, -0.125]
				bl_x = math.cos(cur_theta)*bl[0] - math.sin(cur_theta)*bl[1] + cur_x
				bl_y = math.sin(cur_theta)*bl[0] + math.cos(cur_theta)*bl[1] + cur_y
				# It is a right footstep
				ax1.plot(footsteps_x[i],footsteps_y[i],'r*')
				#rect = patches.Rectangle((bl_x,bl_y),0.1,0.25,math.degrees(cur_theta),linewidth=1, edgecolor='r',facecolor='none')
				#ax1.add_patch(rect)
				ax1.arrow(cur_x,cur_y,0.25*math.cos(cur_theta),0.25*math.sin(cur_theta))
				p1 = [0,-0.2]
				p2 = [0,0.6]
				center_x1 = cur_x + p1[0]*math.cos(cur_theta) - p1[1]*math.sin(cur_theta)
				center_y1 = cur_y + p1[0]*math.sin(cur_theta) + p1[1]*math.cos(cur_theta)

				center_x2 = cur_x + p2[0]*math.cos(cur_theta) - p2[1]*math.sin(cur_theta)
				center_y2 = cur_y + p2[0]*math.sin(cur_theta) + p2[1]*math.cos(cur_theta)

				#circ1 = patches.Circle((center_x1,center_y1),0.55,linewidth=1, edgecolor='b',facecolor='none')
				#circ2 = patches.Circle((center_x2,center_y2),0.55,linewidth=1, edgecolor='b',facecolor='none')
				# ax1.add_patch(circ1)
				# ax1.add_patch(circ2)

		ani = animation.FuncAnimation(fig, animate, interval=1000)
		plt.show()


		# for i in range(3,N):
		# 	plt.plot(left_footsteps_x[:i],left_footsteps_y[:i],'bo', label="LL")
		# 	plt.plot(right_footsteps_x[:i],right_footsteps_y[:i], 'r*', label="RL")
		# 	plt.legend()
		# 	plt.show()
		# 	plt.pause(0.1)

	except GurobiError as e:
		print('Error code' + str(e.errno)+":"+str(e))