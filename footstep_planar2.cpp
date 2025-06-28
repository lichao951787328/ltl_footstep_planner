#define _USE_MATH_DEFINES // For M_PI
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <array>
#include <numeric> // For std::iota if needed, or quicksum simulation
#include <map>
#include "gurobi_c++.h"

// For ALIP functions (optional if not used in main optimization)
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions> // For matrixExponential

// For plotting
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// Helper to create a GRBLinExpr from a vector of GRBVvars and coefficients
// Gurobi's quicksum is more concise in Python
GRBLinExpr quicksum(const std::vector<GRBVar>& vars, const std::vector<double>& coeffs) {
    GRBLinExpr expr = 0;
    if (vars.size() != coeffs.size()) {
        throw std::runtime_error("quicksum: vars and coeffs size mismatch");
    }
    for (size_t i = 0; i < vars.size(); ++i) {
        expr += vars[i] * coeffs[i];
    }
    return expr;
}

GRBLinExpr quicksum(const std::vector<GRBVar>& vars) {
    GRBLinExpr expr = 0;
    for (const auto& var : vars) {
        expr += var;
    }
    return expr;
}

// --- ALIP Matrix Functions (using Eigen) ---
// Note: These are not used in the Gurobi optimization part of the Python script.
// Provided for completeness based on the input file.

Eigen::Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) {
    Eigen::Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}

std::pair<Eigen::Matrix4d, Eigen::Vector4d> get_alip_matrices_with_input(double H_com, double mass, double g, double T_ss_dt) {
    Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Vector4d B_c_input_effect;
    B_c_input_effect << 0, 0, 0, 1;

    Eigen::Matrix4d A_d_for_mpc = (A_c_autonomous * T_ss_dt).exp();
    Eigen::Vector4d B_d_for_mpc;
    if (A_c_autonomous.determinant() == 0) { // Check for singularity
        // Use series expansion for A_c_inv * (expm(A_c*T) - I) if A_c is singular
        // For simplicity, assuming A_c is invertible here. A more robust solution is needed for singular A_c.
        // B_d = (I*T_ss_dt + A_c*T_ss_dt^2/2! + A_c^2*T_ss_dt^3/3! + ...) B_c
        // This is a common approximation if T_ss_dt is small:
        // B_d_for_mpc = B_c_input_effect * T_ss_dt; // First order approximation
        // Or more accurately:
        Eigen::Matrix4d Phi_int_B = Eigen::Matrix4d::Identity() * T_ss_dt; // Integral of I
        // Add more terms for higher accuracy if needed
        // Phi_int_B += (A_c_autonomous * T_ss_dt * T_ss_dt / 2.0);
        // B_d_for_mpc = Phi_int_B * B_c_input_effect;
        // The Python code uses inv, so let's try to stick to it, warning if singular.
        std::cerr << "Warning: A_c_autonomous might be singular in get_alip_matrices_with_input. Inverse used." << std::endl;
    }
    // This is valid if A_c is invertible
    B_d_for_mpc = A_c_autonomous.inverse() * (A_d_for_mpc - Eigen::Matrix4d::Identity()) * B_c_input_effect;
    
    return {A_d_for_mpc, B_d_for_mpc};
}

Eigen::Matrix4d get_alip_matrices(double H_com, double mass, double g, double T_ss_dt) {
    Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);
    return (A_c_autonomous * T_ss_dt).exp();
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 3>> get_alip_reset_map_matrices_detailed(double T_ds, double H_com, double mass, double g) {
    Eigen::Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix4d Ar_ds = (A_c * T_ds).exp();

    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0,
                     0, 0,
                     0, mass * g,
                     -mass * g, 0;

    Eigen::Matrix<double, 4, 2> B_ds = Eigen::Matrix<double, 4, 2>::Zero();
    try {
        Eigen::Matrix4d A_c_inv = A_c.inverse();
        Eigen::Matrix4d Ar_ds_inv = Ar_ds.inverse();
        
        // Original Python line:
        // B_ds = Ar_ds @ A_c_inv @ ((1/T_ds) * A_c_inv @ (np.eye(A_c.shape[0]) - Ar_ds_inv) - Ar_ds_inv) @ B_CoP_for_Bds
        B_ds = Ar_ds * A_c_inv * ((1.0/T_ds) * A_c_inv * (Eigen::Matrix4d::Identity() - Ar_ds_inv) - Ar_ds_inv) * B_CoP_for_Bds;

    } catch (const std::runtime_error& e) { // Eigen throws on non-invertible for .inverse()
        std::cerr << "Warning: A_c or Ar_ds is singular in get_alip_reset_map_matrices_detailed. B_ds set to zero." << std::endl;
        // B_ds remains zero
    }

    Eigen::Matrix<double, 4, 3> B_fp;
    B_fp << 1, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 0;

    Eigen::Matrix<double, 4, 3> B_ds_padded = Eigen::Matrix<double, 4, 3>::Zero();
    B_ds_padded.block<4,2>(0,0) = B_ds;

    Eigen::Matrix<double, 4, 3> B_r = B_ds_padded + B_fp;

    return {Ar_ds, B_r};
}


int main(int argc, char *argv[]) {
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.set(GRB_StringAttr_ModelName, "footstep_cpp");

        // Parameters
        const int N = 18; // Number of footsteps
        const int N_REG = 5; // Number of safe regions
        const double M_BIG = 20.0; // Big M for indicator constraints

        // Create variables
        // Footsteps: x, y, theta, sin(theta), cos(theta)
        std::vector<std::array<GRBVar, 5>> footsteps(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 5; ++j) {
                footsteps[i][j] = model.addVar(-5.0, 5.0, 0.0, GRB_CONTINUOUS, "F" + std::to_string(i) + "_" + std::to_string(j));
            }
        }

        // Trig approx binary variables
        std::vector<std::array<GRBVar, 5>> S_vars(N); // For sin
        std::vector<std::array<GRBVar, 5>> C_vars(N); // For cos
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 5; ++j) {
                S_vars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "S" + std::to_string(i) + "_" + std::to_string(j));
                C_vars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "C" + std::to_string(i) + "_" + std::to_string(j));
            }
        }

        // Safe regions binary variables
        std::vector<std::array<GRBVar, N_REG>> H_vars(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N_REG; ++j) {
                H_vars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "H" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
        
        model.update(); // Update model to "see" new variables

        // DEFINE REGIONS (SCENARIO 3 from Python)
        double R1_xmax = 1, R1_xmin = 0, R1_ymax = 1, R1_ymin = 0;
        double R2_xmax = 1.6, R2_xmin = 1.1, R2_ymax = 2, R2_ymin = 0;
        double R3_xmax = 2, R3_xmin = 1.1, R3_ymax = 2.5, R3_ymin = 2.1;
        double R4_xmax = 1, R4_xmin = -0.5, R4_ymax = 2.7, R4_ymin = 2.1;
        double R5_xmax = 2, R5_xmin = 1.5, R5_ymax = 3, R5_ymin = 2.55;

        std::array<double, 2> R1_midpt = {(R1_xmax + R1_xmin) / 2, (R1_ymax + R1_ymin) / 2};
        std::array<double, 2> R2_midpt = {(R2_xmax + R2_xmin) / 2, (R2_ymax + R2_ymin) / 2};
        std::array<double, 2> R3_midpt = {(R3_xmax + R3_xmin) / 2, (R3_ymax + R3_ymin) / 2};
        std::array<double, 2> R4_midpt = {(R4_xmax + R4_xmin) / 2, (R4_ymax + R4_ymin) / 2};
        std::array<double, 2> R5_midpt = {(R5_xmax + R5_xmin) / 2, (R5_ymax + R5_ymin) / 2};

        // Region constraint matrices (A) and vectors (b)
        // A_i * [x, y, theta]^T <= b_i
        // Each region is defined by 6 inequalities (xmin, xmax, ymin, ymax, thmin, thmax)
        std::vector<std::array<std::array<double, 3>, 6>> A_regions(N_REG);
        std::vector<std::array<double, 6>> b_regions(N_REG);

        // Common A matrix structure for all regions (axis-aligned box in x,y,theta)
        std::array<std::array<double, 3>, 6> A_template = {{
            {{1, 0, 0}}, {{-1, 0, 0}}, {{0, 1, 0}}, {{0, -1, 0}}, {{0, 0, 1}}, {{0, 0, -1}}
        }};

        A_regions[0] = A_template; b_regions[0] = {R1_xmax, -R1_xmin, R1_ymax, -R1_ymin, M_PI, M_PI/2.0}; // Note: Python was math.pi, math.pi/2 for b[4],b[5]
        A_regions[1] = A_template; b_regions[1] = {R2_xmax, -R2_xmin, R2_ymax, -R2_ymin, M_PI, M_PI/2.0}; // Assuming last two b are upper bounds
        A_regions[2] = A_template; b_regions[2] = {R3_xmax, -R3_xmin, R3_ymax, -R3_ymin, M_PI, M_PI/2.0}; // If they were ranges, one would be -theta_lower
        A_regions[3] = A_template; b_regions[3] = {R4_xmax, -R4_xmin, R4_ymax, -R4_ymin, M_PI, M_PI/2.0};
        A_regions[4] = A_template; b_regions[4] = {R5_xmax, -R5_xmin, R5_ymax, -R5_ymin, M_PI, M_PI/2.0};
        // The Python code for b_1 was: [R1_xmax,-R1_xmin,R1_ymax,-R1_ymin,math.pi,math.pi/2]
        // This means theta <= pi and -theta <= pi/2 (i.e. theta >= -pi/2).
        // So it should be: {R1_xmax, -R1_xmin, R1_ymax, -R1_ymin, M_PI, -(-M_PI/2.0)} if b[5] is for -theta <= bound
        // Or if constraint is A[5][2]*theta <= b[5] meaning -theta <= pi/2 -> theta >= -pi/2
        // Let's adjust b for the last two constraints to match typical Ax <= b for min/max on theta
        // theta <= theta_max  (A_ij = [0,0,1], b_i = theta_max)
        // theta >= theta_min  -> -theta <= -theta_min (A_ij = [0,0,-1], b_i = -theta_min)
        double theta_upper_bound = M_PI;
        double theta_lower_bound = -M_PI/2.0; 
        for(int r=0; r<N_REG; ++r) {
            b_regions[r][4] = theta_upper_bound;
            b_regions[r][5] = -theta_lower_bound; // for -theta <= -theta_min
        }


        // All footsteps must be in one of the regions
        for (int c = 0; c < N; ++c) {
            for (int r = 0; r < N_REG; ++r) {
                for (int i = 0; i < 6; ++i) { // 6 inequalities per region
                    GRBLinExpr expr = 0;
                    for (int j = 0; j < 3; ++j) { // x, y, theta
                        expr += A_regions[r][i][j] * footsteps[c][j];
                    }
                    // Constraint: expr - b_regions[r][i] <= M_BIG * (1 - H_vars[c][r])
                    // which is: expr - b_regions[r][i] - M_BIG * (1 - H_vars[c][r]) <= 0
                    // or:       expr - b_regions[r][i] + M_BIG * H_vars[c][r] <= M_BIG
                    model.addConstr(expr - b_regions[r][i] <= M_BIG * (1.0 - H_vars[c][r]),
                                    "region_constr_c" + std::to_string(c) + "_r" + std::to_string(r) + "_i" + std::to_string(i));
                }
            }
            // Sum of H_vars[c][j] for j must be 1
            GRBLinExpr sum_H = 0;
            for (int j = 0; j < N_REG; ++j) {
                sum_H += H_vars[c][j];
            }
            model.addConstr(sum_H == 1.0, "sum_H_c" + std::to_string(c));
        }

        // Reachability constraint
        for (int c = 2; c < N; ++c) {
            GRBVar xn = footsteps[c][0];
            GRBVar yn = footsteps[c][1];
            GRBVar xc = footsteps[c-1][0];
            GRBVar yc = footsteps[c-1][1];
            // GRBVar thetac = footsteps[c-1][2]; // Not used directly, but sin/cos are
            GRBVar sinc_prev = footsteps[c-1][3];
            GRBVar cosc_prev = footsteps[c-1][4];

            if (c % 2 != 0) { // Odd step (e.g., f3, f5, ...), right leg
                std::array<double, 2> p1 = {0, 0.1};
                std::array<double, 2> p2 = {0, -0.8};
                double d1 = 0.55, d2 = 0.55;

                GRBLinExpr term1_a_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
                GRBLinExpr term2_a_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
                model.addQConstr(term1_a_expr * term1_a_expr + term2_a_expr * term2_a_expr <= d1*d1, "reach_a_c" + std::to_string(c));
                
                GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev);
                GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev);
                model.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));

            } else { // Even step, left leg
                std::array<double, 2> p1 = {0, -0.1};
                std::array<double, 2> p2 = {0, 0.8};
                double d1 = 0.55, d2 = 0.55;

                GRBLinExpr term1_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
                GRBLinExpr term2_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
                model.addQConstr(term1_expr * term1_expr + term2_expr * term2_expr <= d1*d1, "reach_a_c" + std::to_string(c));

                GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev); // Python had this as term1 again
                GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev); // Python had this as term2 again
                model.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));
            }
        }

        // Sin approximation constraints
        std::vector<std::tuple<double, double, double, double>> sin_params = {
            {-M_PI, 1 - M_PI, -1, -M_PI},
            {1 - M_PI, -1, 0, -1},
            {-1, 1, 1, 0},
            {1, M_PI - 1, 0, 1},
            {M_PI - 1, M_PI, -1, M_PI}
        };
        for (int c = 0; c < N; ++c) {
            GRBVar theta_c = footsteps[c][2];
            GRBVar sin_theta_c = footsteps[c][3];
            for (int i = 0; i < 5; ++i) {
                double phi_l = std::get<0>(sin_params[i]);
                double phi_lp1 = std::get<1>(sin_params[i]);
                double g_l = std::get<2>(sin_params[i]);
                double h_l = std::get<3>(sin_params[i]);
                GRBVar S_ci = S_vars[c][i];

                // theta_c in [phi_l, phi_lp1] if S_ci = 1
                model.addConstr(theta_c <= phi_lp1 + M_BIG * (1.0 - S_ci)); // Original: -M*(1-S) - phi_lp1 + theta <= 0 => theta - phi_lp1 <= M(1-S)
                model.addConstr(theta_c >= phi_l - M_BIG * (1.0 - S_ci));   // Original: -M*(1-S) + phi_l - theta <= 0 => phi_l - theta <= M(1-S) => theta - phi_l >= -M(1-S)

                // sin_theta_c = g_l * theta_c + h_l if S_ci = 1
                model.addConstr(sin_theta_c <= g_l * theta_c + h_l + M_BIG * (1.0 - S_ci));
                model.addConstr(sin_theta_c >= g_l * theta_c + h_l - M_BIG * (1.0 - S_ci));
            }
            GRBLinExpr sum_S = 0;
            for (int j = 0; j < 5; ++j) sum_S += S_vars[c][j];
            model.addConstr(sum_S == 1.0, "sum_S_c" + std::to_string(c));
        }

        // Cos approximation constraints
        std::vector<std::tuple<double, double, double, double>> cos_params = {
            {-M_PI, -1 - M_PI/2, 0, -1},
            {-1 - M_PI/2, 1 - M_PI/2, 1, M_PI/2},
            {1 - M_PI/2, M_PI/2 - 1, 0, 1},
            {M_PI/2 - 1, M_PI/2 + 1, -1, M_PI/2},
            {M_PI/2 + 1, M_PI, 0, -1}
        };
        for (int c = 0; c < N; ++c) {
            GRBVar theta_c = footsteps[c][2];
            GRBVar cos_theta_c = footsteps[c][4];
            for (int i = 0; i < 5; ++i) {
                double phi_l = std::get<0>(cos_params[i]);
                double phi_lp1 = std::get<1>(cos_params[i]);
                double g_l = std::get<2>(cos_params[i]);
                double h_l = std::get<3>(cos_params[i]);
                GRBVar C_ci = C_vars[c][i];

                model.addConstr(theta_c <= phi_lp1 + M_BIG * (1.0 - C_ci));
                model.addConstr(theta_c >= phi_l - M_BIG * (1.0 - C_ci));
                model.addConstr(cos_theta_c <= g_l * theta_c + h_l + M_BIG * (1.0 - C_ci));
                model.addConstr(cos_theta_c >= g_l * theta_c + h_l - M_BIG * (1.0 - C_ci));
            }
            GRBLinExpr sum_C = 0;
            for (int j = 0; j < 5; ++j) sum_C += C_vars[c][j];
            model.addConstr(sum_C == 1.0, "sum_C_c" + std::to_string(c));
        }
        
        // Initial footstep constraints
        double init_theta = 0.0;
        double f1_s = std::sin(init_theta);
        double f1_c = std::cos(init_theta);

        model.addConstr(footsteps[0][0] == 0.0, "f0_x");
        model.addConstr(footsteps[0][1] == 0.4, "f0_y");
        // footsteps[0][2] (theta) is not explicitly set here, but its sin/cos are.
        // This implies theta is such that sin(theta) = f1_s and cos(theta) = f1_c
        // We need to pick the S and C vars that correspond to theta=0
        // For sin: theta=0 is in segment 2 (-1 to 1), where S[c][2] = 1. g_l=1, h_l=0. sin(0) = 1*0+0=0
        // For cos: theta=0 is in segment 2 (1-pi/2 to pi/2-1), where C[c][2] = 1. g_l=0, h_l=1. cos(0) = 0*0+1=1
        model.addConstr(footsteps[0][3] == f1_s, "f0_s");
        model.addConstr(footsteps[0][4] == f1_c, "f0_c");
        model.addConstr(S_vars[0][2] == 1.0, "S0_2"); // sin(0) is in segment where g=1,h=0 (-1 to 1)
        model.addConstr(C_vars[0][2] == 1.0, "C0_2"); // cos(0) is in segment where g=0,h=1 (1-pi/2 to pi/2-1, approx -0.57 to 0.57)

        model.addConstr(footsteps[1][0] == 0.0, "f1_x");
        model.addConstr(footsteps[1][1] == 0.0, "f1_y");
        model.addConstr(footsteps[1][2] == init_theta, "f1_theta"); // Explicitly set theta here
        model.addConstr(footsteps[1][3] == f1_s, "f1_s");
        model.addConstr(footsteps[1][4] == f1_c, "f1_c");
        model.addConstr(S_vars[1][2] == 1.0, "S1_2");
        model.addConstr(C_vars[1][2] == 1.0, "C1_2");

        // Small displacement thresholds
        double small_displacement_threshold = 0.15;
        double T_step = 1.0; // Time step duration (not T_cycle for velocity smoothness)

        GRBLinExpr dx_init = footsteps[2][0] - footsteps[0][0];
        GRBLinExpr dy_init = footsteps[2][1] - footsteps[0][1];
        model.addQConstr(dx_init*dx_init + dy_init*dy_init <= std::pow(small_displacement_threshold * T_step, 2), "init_step_displacement_sq");

        GRBLinExpr dx_final = footsteps[N-1][0] - footsteps[N-3][0];
        GRBLinExpr dy_final = footsteps[N-1][1] - footsteps[N-3][1];
        model.addQConstr(dx_final*dx_final + dy_final*dy_final <= std::pow(small_displacement_threshold * T_step, 2), "final_step_displacement_sq");

        // Max rotation per step
        double del_theta_max = M_PI / 8.0;
        for (int c = 1; c < N; ++c) {
            GRBLinExpr delta_theta = footsteps[c][2] - footsteps[c-1][2];
            model.addConstr(delta_theta <= del_theta_max, "max_rot_pos_c" + std::to_string(c));
            model.addConstr(delta_theta >= -del_theta_max, "max_rot_neg_c" + std::to_string(c));
        }
        
        // --- Objective Function ---
        GRBQuadExpr total_objective = 0;

        // Terminal cost (Scenario 3 goal)
        std::array<double, 3> g_target = {1.5, 2.2, 3.0 * M_PI / 4.0};
        GRBLinExpr e0 = footsteps[N-1][0] - g_target[0];
        GRBLinExpr e1 = footsteps[N-1][1] - g_target[1];
        GRBLinExpr e2 = footsteps[N-1][2] - g_target[2];
        
        std::array<std::array<double,3>,3> Q_mat = {{ {300,0,0}, {0,300,0}, {0,0,300} }};
        GRBQuadExpr term_cost = 0;
        term_cost += e0*e0*Q_mat[0][0] + e0*e1*Q_mat[0][1] + e0*e2*Q_mat[0][2];
        term_cost += e1*e0*Q_mat[1][0] + e1*e1*Q_mat[1][1] + e1*e2*Q_mat[1][2];
        term_cost += e2*e0*Q_mat[2][0] + e2*e1*Q_mat[2][1] + e2*e2*Q_mat[2][2];
        total_objective += term_cost;

        // Incremental cost
        std::array<std::array<double,3>,3> R_mat = {{ {0.5,0,0}, {0,0.5,0}, {0,0,0.5} }};
        GRBQuadExpr inc_cost = 0;
        // Python: for j in range(0,N). If j=0, footsteps[j-1] is invalid.
        // Assuming it's range(1,N) for differences, or special handling for j=0.
        // The Python code `footsteps[j][0]-footsteps[j-1][0]` will fail for j=0 if footsteps[-1] is not defined.
        // Let's assume it was meant for j from 1 to N-1 (N-1 differences).
        // Or, if it's N terms, then footsteps[-1] is perhaps footsteps[N-1] (cyclic) or 0.
        // Given the context of path smoothness, range(1,N) seems more likely.
        for (int j = 1; j < N; ++j) {
            GRBLinExpr dx = footsteps[j][0] - footsteps[j-1][0];
            GRBLinExpr dy = footsteps[j][1] - footsteps[j-1][1];
            GRBLinExpr dtheta = footsteps[j][2] - footsteps[j-1][2];
            inc_cost += dx*dx*R_mat[0][0] + dy*dy*R_mat[1][1] + dtheta*dtheta*R_mat[2][2];
            // Add off-diagonal terms if R_mat is not diagonal
            inc_cost += dx*dy*R_mat[0][1] + dx*dtheta*R_mat[0][2];
            inc_cost += dy*dx*R_mat[1][0] + dy*dtheta*R_mat[1][2];
            inc_cost += dtheta*dx*R_mat[2][0] + dtheta*dy*R_mat[2][1];
        }
        total_objective += inc_cost;

        // Velocity smoothness cost
        double T_cycle = 1.0;
        double G_vel_smoothness = 5.0;
        GRBQuadExpr velocity_smoothness_cost_expr = 0;

        if (N > 2) { // Need at least 3 footsteps for 2 midpoints -> 1 velocity
            std::vector<std::pair<GRBLinExpr, GRBLinExpr>> midpoints_xy_exprs;
            for (int j = 1; j < N; ++j) { // N-1 midpoints
                GRBLinExpr mid_x = (footsteps[j][0] + footsteps[j-1][0]) / 2.0;
                GRBLinExpr mid_y = (footsteps[j][1] + footsteps[j-1][1]) / 2.0;
                midpoints_xy_exprs.push_back({mid_x, mid_y});
            }

            if (midpoints_xy_exprs.size() > 1) { // Need at least 2 midpoints for 1 velocity
                std::vector<std::pair<GRBLinExpr, GRBLinExpr>> velocities_xy_exprs;
                for (size_t k = 1; k < midpoints_xy_exprs.size(); ++k) { // N-2 velocities
                    GRBLinExpr vel_x = (midpoints_xy_exprs[k].first - midpoints_xy_exprs[k-1].first) / T_cycle;
                    GRBLinExpr vel_y = (midpoints_xy_exprs[k].second - midpoints_xy_exprs[k-1].second) / T_cycle;
                    velocities_xy_exprs.push_back({vel_x, vel_y});
                }
                
                if (velocities_xy_exprs.size() > 1) { // Need at least 2 velocities for 1 acceleration
                    for (size_t l = 1; l < velocities_xy_exprs.size(); ++l) { // N-3 accelerations
                        GRBLinExpr accel_x = velocities_xy_exprs[l].first - velocities_xy_exprs[l-1].first;
                        GRBLinExpr accel_y = velocities_xy_exprs[l].second - velocities_xy_exprs[l-1].second;
                        velocity_smoothness_cost_expr += accel_x*accel_x + accel_y*accel_y;
                    }
                }
            }
        }
        total_objective += G_vel_smoothness * velocity_smoothness_cost_expr;
        
        model.setObjective(total_objective, GRB_MINIMIZE);

        // --- Solve ---
        // model.set(GRB_IntParam_MIPFocus, 0); // Default is 0 (balanced)
        // model.set(GRB_DoubleParam_TimeLimit, 300.0); // Optional: set a time limit
        // model.set(GRB_DoubleParam_MIPGap, 0.01); // Optional: set a MIP gap
        
        model.optimize();

        std::cout << "Gurobi optimization time: " << model.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
        std::cout << "Objective value: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

        // --- Retrieve and Print Results ---
        std::vector<double> footsteps_x_sol(N), footsteps_y_sol(N), footsteps_theta_sol(N);
        for (int i = 0; i < N; ++i) {
            footsteps_x_sol[i] = footsteps[i][0].get(GRB_DoubleAttr_X);
            footsteps_y_sol[i] = footsteps[i][1].get(GRB_DoubleAttr_X);
            footsteps_theta_sol[i] = footsteps[i][2].get(GRB_DoubleAttr_X);
            printf("Footstep %2d: x=%.2f, y=%.2f, theta=%.2f\n", 
                   i, footsteps_x_sol[i], footsteps_y_sol[i], footsteps_theta_sol[i]);
        }

        // --- Plotting with matplotlib-cpp ---
        // plt::figure_size(800, 800);
        // plt::xlim(-2, 5);
        // plt::ylim(-2, 5);

        // // Plot initial foot stance
        // // plt::plot({footsteps_x_sol[0]}, {footsteps_y_sol[0]}, "bo", {{"label", "Left Foot (start)"}});
        // // plt::plot({footsteps_x_sol[1]}, {footsteps_y_sol[1]}, "r*", {{"label", "Right Foot (start)"}});
        // std::map<std::string, std::string> plot_kwargs_left_start;
        // plot_kwargs_left_start["label"] = "Left Foot (start)";
        // plot_kwargs_left_start["marker"] = "o"; // 从 "bo" 分离出标记
        // plot_kwargs_left_start["color"] = "b";  // 从 "bo" 分离出颜色
        // plt::plot({footsteps_x_sol[0]}, {footsteps_y_sol[0]}, plot_kwargs_left_start);
        // std::map<std::string, std::string> plot_kwargs_right_start;
        // plot_kwargs_right_start["label"] = "Right Foot (start)";
        // plot_kwargs_right_start["marker"] = "*"; // 从 "r*" 分离出标记
        // plot_kwargs_right_start["color"] = "r";  // 从 "r*" 分
        // plt::plot({footsteps_x_sol[1]}, {footsteps_y_sol[1]}, plot_kwargs_right_start);

        
        // // Arrows for initial orientation
        // plt::arrow(footsteps_x_sol[0], footsteps_y_sol[0], 0.25 * std::cos(footsteps_theta_sol[0]), 0.25 * std::sin(footsteps_theta_sol[0]));
        // plt::arrow(footsteps_x_sol[1], footsteps_y_sol[1], 0.25 * std::cos(footsteps_theta_sol[1]), 0.25 * std::sin(footsteps_theta_sol[1]));

        // // Plot safe regions
        // auto plot_rect = [&](double xmin, double ymin, double xmax, double ymax, const std::string& label) {
        //     std::vector<double> x_coords = {xmin, xmax, xmax, xmin, xmin};
        //     std::vector<double> y_coords = {ymin, ymin, ymax, ymax, ymin};
        //     // plt::plot(x_coords, y_coords, "g-", {{"alpha", "0.4"}});
        //     plt::fill(x_coords, y_coords, {{"color", "green"}, {"alpha", "0.2"}});
        //     plt::text((xmin+xmax)/2.0, (ymin+ymax)/2.0, label);
        // };
        // plot_rect(R1_xmin, R1_ymin, R1_xmax, R1_ymax, "R1");
        // plot_rect(R2_xmin, R2_ymin, R2_xmax, R2_ymax, "R2");
        // plot_rect(R3_xmin, R3_ymin, R3_xmax, R3_ymax, "R3");
        // plot_rect(R4_xmin, R4_ymin, R4_xmax, R4_ymax, "R4");
        // plot_rect(R5_xmin, R5_ymin, R5_xmax, R5_ymax, "R5");
        
        // // Plot subsequent footsteps
        // std::vector<double> planned_path_x, planned_path_y;
        // for (int i = 2; i < N; ++i) {
        //     planned_path_x.push_back(footsteps_x_sol[i]);
        //     planned_path_y.push_back(footsteps_y_sol[i]);
        //     if (i % 2 == 0) { // Left foot (steps 2, 4, ...)
        //         plt::plot({footsteps_x_sol[i]}, {footsteps_y_sol[i]}, "bo");
        //     } else { // Right foot (steps 3, 5, ...)
        //         plt::plot({footsteps_x_sol[i]}, {footsteps_y_sol[i]}, "r*");
        //     }
        //     plt::arrow(footsteps_x_sol[i], footsteps_y_sol[i], 
        //                0.25 * std::cos(footsteps_theta_sol[i]), 
        //                0.25 * std::sin(footsteps_theta_sol[i]));
        // }
        // // Connect the path (first two steps are fixed, path starts from step 1 to 2, then 2 to 3 etc.)
        // std::vector<double> full_path_x = {footsteps_x_sol[0], footsteps_x_sol[1]};
        // std::vector<double> full_path_y = {footsteps_y_sol[0], footsteps_y_sol[1]};
        // full_path_x.insert(full_path_x.end(), planned_path_x.begin(), planned_path_x.end());
        // full_path_y.insert(full_path_y.end(), planned_path_y.begin(), planned_path_y.end());
        // plt::plot(full_path_x, full_path_y, {{"color", "gray"}, {"linestyle", "--"}, {"label", "Planned Path"}});


        // plt::xlabel("X coordinate");
        // plt::ylabel("Y coordinate");
        // plt::title("Footstep Plan");
        // plt::legend();
        // plt::grid(true);
        // plt::save("footstep_plan_cpp.png");
        // plt::show();

    } catch (GRBException& e) {
        std::cerr << "Gurobi Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}