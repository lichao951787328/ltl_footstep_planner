#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <stdexcept> // For std::runtime_error

// Gurobi
#include "gurobi_c++.h"

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions> // For matrix.exp()

// Constants
const double INF = GRB_INFINITY;

// Type aliases for Eigen matrices/vectors for clarity
using Matrix4d = Eigen::Matrix4d;
using Vector4d = Eigen::Vector4d;
using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;
template<int Rows, int Cols>
using Matrix = Eigen::Matrix<double, Rows, Cols>;


// --- Helper Functions ---

Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) {
    Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1.0 / (mass * H_com),
                      0, 0, -1.0 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}

std::pair<Matrix4d, Vector4d> get_alip_matrices_with_input(
    double H_com, double mass, double g, double T_ss_dt) {
    Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);
    Vector4d B_c_input_effect;
    B_c_input_effect << 0, 0, 0, 1;

    // Using augmented matrix for robust B_d calculation
    Eigen::Matrix<double, 5, 5> M_cont = Eigen::Matrix<double, 5, 5>::Zero();
    M_cont.topLeftCorner(4, 4) = A_c_autonomous;
    M_cont.topRightCorner(4, 1) = B_c_input_effect;

    Eigen::Matrix<double, 5, 5> M_disc = (M_cont * T_ss_dt).exp();
    Matrix4d A_d_for_mpc = M_disc.topLeftCorner(4, 4);
    Vector4d B_d_for_mpc = M_disc.topRightCorner(4, 1);

    return {A_d_for_mpc, B_d_for_mpc};
}





std::pair<Matrix4d, Eigen::Matrix<double, 4, 3>> get_alip_reset_map_matrices_detailed(
    double T_ds, double H_com, double mass, double g) {
    Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Matrix4d Ar_ds = (A_c * T_ds).exp();

    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0,
                     0, 0,
                     0, mass * g,
                     -mass * g, 0;

    Eigen::Matrix<double, 4, 2> B_ds = Eigen::Matrix<double, 4, 2>::Zero();;
    if (A_c.determinant() != 0 && Ar_ds.determinant() != 0) { // Basic check
        try {
            Matrix4d A_c_inv = A_c.inverse();
            Matrix4d Ar_ds_inv = Ar_ds.inverse();
            Matrix4d I4 = Matrix4d::Identity();
            Matrix4d term1 = (1.0 / T_ds) * (A_c_inv * (I4 - Ar_ds_inv));
            Matrix4d term2 = Ar_ds_inv;
            B_ds = Ar_ds * A_c_inv * (term1 - term2) * B_CoP_for_Bds;
        } catch (const std::runtime_error& e) {
             std::cerr << "Eigen Warning: Singularity in get_alip_reset_map_matrices_detailed. B_ds might be incorrect: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Warning: A_c or Ar_ds is singular in reset map. B_ds set to zero." << std::endl;
    }


    Eigen::Matrix<double, 4, 3> B_fp;
    B_fp << 1, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 0;

    Eigen::Matrix<double, 4, 3> B_ds_padded = Eigen::Matrix<double, 4, 3>::Zero();
    B_ds_padded.leftCols(2) = B_ds;

    Eigen::Matrix<double, 4, 3> B_r = B_ds_padded + B_fp;

    return {Ar_ds, B_r};
}

std::pair<Vector4d, Vector4d> calculate_periodic_alip_reference_states(
    double vx_d, double vy_d, double stance_width_l,
    double T_s2s_cycle, const Matrix4d& A_s2s_autonomous_cycle,
    const Eigen::Matrix<double, 4, 2>& Br_map_for_cycle_2d,
    bool initial_stance_is_left) {

    Vector2d v_d_vec(vx_d, vy_d);
    double sigma_1 = initial_stance_is_left ? -1.0 : 1.0;
    double sigma_2 = initial_stance_is_left ? 1.0 : -1.0;

    Vector2d delta_p_A_2d, delta_p_B_2d;
    delta_p_A_2d(0) = v_d_vec(0) * T_s2s_cycle;
    delta_p_A_2d(1) = v_d_vec(1) * T_s2s_cycle + sigma_1 * stance_width_l;
    delta_p_B_2d(0) = v_d_vec(0) * T_s2s_cycle;
    delta_p_B_2d(1) = v_d_vec(1) * T_s2s_cycle + sigma_2 * stance_width_l;

    Matrix4d I_mat = Matrix4d::Identity();
    Matrix4d M_cycle = A_s2s_autonomous_cycle;
    Matrix4d M_sq_cycle = M_cycle * M_cycle;

    Matrix4d lhs_matrix = I_mat - M_sq_cycle;
    Vector4d rhs_vector = M_cycle * (Br_map_for_cycle_2d * delta_p_A_2d) + (Br_map_for_cycle_2d * delta_p_B_2d);

    Vector4d x_start_of_2_step_cycle_ref;
    if (lhs_matrix.determinant() == 0) {
        std::cerr << "Singular matrix in calculating periodic reference. Using pseudo-inverse or zeros." << std::endl;
        // x_start_of_2_step_cycle_ref = lhs_matrix.completeOrthogonalDecomposition().pseudoInverse() * rhs_vector; // More robust
        x_start_of_2_step_cycle_ref = Vector4d::Zero(); // Fallback
    } else {
        x_start_of_2_step_cycle_ref = lhs_matrix.colPivHouseholderQr().solve(rhs_vector);
    }

    Vector4d ref_state_phase0 = x_start_of_2_step_cycle_ref;
    Vector4d ref_state_phase1 = M_cycle * ref_state_phase0 + Br_map_for_cycle_2d * delta_p_A_2d;

    return {ref_state_phase0, ref_state_phase1};
}


// --- Main MPC Setup and Solve ---
int main(int argc, char *argv[]) {
    try {
        GRBEnv env = GRBEnv(true);
        // env.set(GRB_IntParam_OutputFlag, 0); // Disable Gurobi output for cleaner console
        env.set(GRB_StringParam_LogFile, "gurobi_mpfc.log");
        env.start();
        GRBModel model = GRBModel(env);
        model.set(GRB_StringAttr_ModelName, "acosta_mpc_cpp");

        // --- Model Parameters ---
        int N_horizon = 9;
        int K_knots = 10;
        double T_ss = 0.4;
        double T_ds = 0.1;
        double T_ss_dt = T_ss / (K_knots - 1);
        double T_s2s_cycle = T_ss + T_ds;

        double mass = 30.0;
        double g = 9.81;
        double H_com_nominal = 0.8;
        int N_REGIONS = 2;


        // --- Get Dynamics Matrices ---
        Matrix4d A_d_mpc_mat; Vector4d B_d_mpc_vec;
        std::tie(A_d_mpc_mat, B_d_mpc_vec) = get_alip_matrices_with_input(H_com_nominal, mass, g, T_ss_dt);

        Matrix4d Ar_reset_mat; Eigen::Matrix<double, 4, 3> Br_reset_delta_p_mat;
        std::tie(Ar_reset_mat, Br_reset_delta_p_mat) = get_alip_reset_map_matrices_detailed(T_ds, H_com_nominal, mass, g);

        Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com_nominal, mass, g);
        Matrix4d A_d_autonomous_knot = (A_c_autonomous * T_ss_dt).exp();
        Matrix4d A_s2s_autonomous_cycle = (A_c_autonomous * T_s2s_cycle).exp();


        // --- Define Gurobi Variables ---
        std::vector<std::vector<std::vector<GRBVar>>> x_alip_vars(N_horizon,
            std::vector<std::vector<GRBVar>>(K_knots, std::vector<GRBVar>(4)));
        std::vector<std::vector<GRBVar>> u_ankle_vars(N_horizon, std::vector<GRBVar>(K_knots - 1));
        std::vector<std::vector<GRBVar>> p_foot_vars(N_horizon, std::vector<GRBVar>(3));
        std::vector<std::vector<GRBVar>> mu_vars(N_horizon, std::vector<GRBVar>(N_REGIONS));

        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots; ++k) {
                for (int i = 0; i < 4; ++i) {
                    x_alip_vars[n][k][i] = model.addVar(-INF, INF, 0.0, GRB_CONTINUOUS, "x_n" + std::to_string(n) + "_k" + std::to_string(k) + "_i" + std::to_string(i));
                }
                if (k < K_knots - 1) {
                    u_ankle_vars[n][k] = model.addVar(-5.0, 5.0, 0.0, GRB_CONTINUOUS, "u_n" + std::to_string(n) + "_k" + std::to_string(k));
                }
            }
            for (int j = 0; j < 3; ++j) {
                p_foot_vars[n][j] = model.addVar(-INF, INF, 0.0, GRB_CONTINUOUS, "p_n" + std::to_string(n + 1) + "_j" + std::to_string(j));
            }
            for (int i_reg = 0; i_reg < N_REGIONS; ++i_reg) {
                mu_vars[n][i_reg] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "mu_n" + std::to_string(n + 1) + "_reg" + std::to_string(i_reg));
            }
        }
        // model.update(); // Gurobi C++ API often updates implicitly, but can be called.

        // --- Define Perception Data (Foothold Regions) ---
        Eigen::Matrix<double, 6, 3> F_region_common;
        F_region_common << 1,0,0,  -1,0,0,  0,1,0,  0,-1,0,  0,0,1,  0,0,-1;
        Eigen::Matrix<double, 6, 1> c_region1, c_region2;
        c_region1 << 1.0, 0.5, 1.0, 1.0, 0.01, -0.01;
        c_region2 << 5.0, -1.1, 1.0, 1.0, 0.1, -0.1;
        std::vector<Eigen::Matrix<double, 6, 3>> regions_F_list = {F_region_common, F_region_common};
        std::vector<Eigen::Matrix<double, 6, 1>> regions_c_list = {c_region1, c_region2};

        // --- Initial Conditions and Reference Trajectory ---
        double vx_desired = 0.2;
        double vy_desired = 0.0;
        double nominal_stance_width = 0.2;
        bool current_stance_is_left = true;
        Vector3d p_current_foot_val(0.0, 0.1, 0.01); // Initial stance foot

        Eigen::Matrix<double,4,2> Br_map_for_cycle_2d = Br_reset_delta_p_mat.leftCols(2);

        Vector4d ref_state_cycle_phase0, ref_state_cycle_phase1;
        std::tie(ref_state_cycle_phase0, ref_state_cycle_phase1) = calculate_periodic_alip_reference_states(
            vx_desired, vy_desired, nominal_stance_width, T_s2s_cycle, A_s2s_autonomous_cycle, Br_map_for_cycle_2d, current_stance_is_left);

        Vector4d x_current_alip_val = current_stance_is_left ? ref_state_cycle_phase0 : ref_state_cycle_phase1;
        for (int i = 0; i < 4; ++i) {
            model.addConstr(x_alip_vars[0][0][i] == x_current_alip_val(i), "init_x_" + std::to_string(i));
        }

        std::vector<std::vector<Vector4d>> x_d_horizon(N_horizon, std::vector<Vector4d>(K_knots));
        for (int n_mpc = 0; n_mpc < N_horizon; ++n_mpc) {
            Vector4d x_d_nk_start_of_stage = (n_mpc % 2 == (current_stance_is_left ? 0:1) ) ? ref_state_cycle_phase0 : ref_state_cycle_phase1;
            Vector4d x_d_nk = x_d_nk_start_of_stage;
            x_d_horizon[n_mpc][0] = x_d_nk;
            for (int k_knot = 1; k_knot < K_knots; ++k_knot) {
                x_d_nk = A_d_autonomous_knot * x_d_nk;
                x_d_horizon[n_mpc][k_knot] = x_d_nk;
            }
        }

        // --- Add Constraints to Gurobi Model ---
        // Constraint 9b (Dynamics within stance)
        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots - 1; ++k) {
                for (int row = 0; row < 4; ++row) {
                    GRBLinExpr lhs = x_alip_vars[n][k+1][row];
                    GRBLinExpr rhs = 0.0;
                    for (int col = 0; col < 4; ++col) {
                        rhs.addTerm(A_d_mpc_mat(row, col), x_alip_vars[n][k][col]);
                    }
                    rhs.addTerm(B_d_mpc_vec(row), u_ankle_vars[n][k]);
                    model.addConstr(lhs == rhs, "dyn_n" + std::to_string(n) + "_k" + std::to_string(k) + "_r" + std::to_string(row));
                }
            }
        }

        // Constraint 9c (Reset map)
        for (int n = 0; n < N_horizon - 1; ++n) {
            GRBLinExpr dp_x, dp_y, dp_z;
            if (n == 0) {
                dp_x = p_foot_vars[n][0] - p_current_foot_val(0);
                dp_y = p_foot_vars[n][1] - p_current_foot_val(1);
                dp_z = p_foot_vars[n][2] - p_current_foot_val(2);
            } else {
                dp_x = p_foot_vars[n][0] - p_foot_vars[n-1][0];
                dp_y = p_foot_vars[n][1] - p_foot_vars[n-1][1];
                dp_z = p_foot_vars[n][2] - p_foot_vars[n-1][2];
            }
            for (int row = 0; row < 4; ++row) {
                GRBLinExpr lhs = x_alip_vars[n+1][0][row];
                GRBLinExpr rhs_A_term = 0.0;
                for (int col = 0; col < 4; ++col) {
                    rhs_A_term.addTerm(Ar_reset_mat(row, col), x_alip_vars[n][K_knots-1][col]);
                }
                GRBLinExpr rhs_B_term = Br_reset_delta_p_mat(row, 0) * dp_x +
                                       Br_reset_delta_p_mat(row, 1) * dp_y +
                                       Br_reset_delta_p_mat(row, 2) * dp_z;
                model.addConstr(lhs == rhs_A_term + rhs_B_term, "reset_n" + std::to_string(n) + "_r" + std::to_string(row));
            }
        }

        // Constraint 9d (Foothold region selection - Big M)
        double M_big = 100.0;
        for (int n = 0; n < N_horizon; ++n) {
            for (int i_region = 0; i_region < N_REGIONS; ++i_region) {
                const auto& F_mat = regions_F_list[i_region];
                const auto& c_vec = regions_c_list[i_region];
                for (int r_idx = 0; r_idx < F_mat.rows(); ++r_idx) {
                    GRBLinExpr lhs = 0.0;
                    for (int c_idx = 0; c_idx < 3; ++c_idx) {
                        lhs.addTerm(F_mat(r_idx, c_idx), p_foot_vars[n][c_idx]);
                    }
                    model.addConstr(lhs <= c_vec(r_idx) + M_big * (1.0 - mu_vars[n][i_region]),
                                     "foot_reg_n" + std::to_string(n) + "_reg" + std::to_string(i_region) + "_r" + std::to_string(r_idx));
                }
            }
            GRBLinExpr sum_mu = 0.0;
            for (int i_reg = 0; i_reg < N_REGIONS; ++i_reg) {
                sum_mu.addTerm(1.0, mu_vars[n][i_reg]);
            }
            model.addConstr(sum_mu == 1.0, "sum_mu_n" + std::to_string(n));
        }

        // CoM y-limit
        double y_com_max = 0.15;
        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots; ++k) {
                model.addConstr(x_alip_vars[n][k][1] <= y_com_max, "ycom_max_n" + std::to_string(n) + "_k" + std::to_string(k));
                model.addConstr(x_alip_vars[n][k][1] >= -y_com_max, "ycom_min_n" + std::to_string(n) + "_k" + std::to_string(k));
            }
        }
        
        // Kinematic limits on footstep delta
        double max_dx = 0.4, max_dy = 0.3;
        GRBVar prev_p_kin_x, prev_p_kin_y; // GRBVar or double
        for (int n = 0; n < N_horizon; ++n) {
            GRBLinExpr dx, dy;
            if (n == 0) {
                dx = p_foot_vars[n][0] - p_current_foot_val(0);
                dy = p_foot_vars[n][1] - p_current_foot_val(1);
            } else {
                dx = p_foot_vars[n][0] - p_foot_vars[n-1][0];
                dy = p_foot_vars[n][1] - p_foot_vars[n-1][1];
            }
            model.addConstr(dx <= max_dx, "max_dx_n" + std::to_string(n));
            model.addConstr(dx >= -max_dx, "min_dx_n" + std::to_string(n));
            model.addConstr(dy <= max_dy, "max_dy_n" + std::to_string(n));
            model.addConstr(dy >= -max_dy, "min_dy_n" + std::to_string(n));
        }


        // --- Objective Function ---
        GRBQuadExpr objective = 0.0;
        Matrix4d Q_state_cost = Matrix4d::Identity() * 1.0;
        double R_input_cost_val = 0.1;

        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots; ++k) {
                for (int r_idx = 0; r_idx < 4; ++r_idx) {
                    GRBLinExpr err_r = x_alip_vars[n][k][r_idx] - x_d_horizon[n][k](r_idx);
                    objective.addTerm(Q_state_cost(r_idx, r_idx), err_r, err_r);
                }
                if (k < K_knots - 1) {
                    objective.addTerm(R_input_cost_val, u_ankle_vars[n][k], u_ankle_vars[n][k]);
                }
            }
        }
        Matrix4d Q_f_state_cost = Q_state_cost * 0.1;
        for (int r_idx = 0; r_idx < 4; ++r_idx) {
            GRBLinExpr err_f_r = x_alip_vars[N_horizon-1][K_knots-1][r_idx] - x_d_horizon[N_horizon-1][K_knots-1](r_idx);
            objective.addTerm(Q_f_state_cost(r_idx, r_idx), err_f_r, err_f_r);
        }
        model.setObjective(objective, GRB_MINIMIZE);

        // --- Solve ---
        model.set(GRB_DoubleParam_MIPGap, 0.05);
        model.set(GRB_DoubleParam_TimeLimit, 2.0); // Increased for C++ overhead/first run
        model.optimize();

        // --- Process and Print Results ---
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) {
            std::cout << "MPC solution found." << std::endl;
            std::cout << "Gurobi optimization time: " << std::fixed << std::setprecision(4) << model.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
            std::cout << "Objective value: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

            std::vector<Vector3d> p_planned_vals_optimized(N_horizon);
            std::vector<std::vector<double>> mu_planned_vals_optimized(N_horizon, std::vector<double>(N_REGIONS));

            std::cout << "\n--- Planned Footstep Sequence (p1, p2, ...) ---" << std::endl;
            for (int n_opt = 0; n_opt < N_horizon; ++n_opt) {
                for (int j = 0; j < 3; ++j) {
                    p_planned_vals_optimized[n_opt](j) = p_foot_vars[n_opt][j].get(GRB_DoubleAttr_X);
                }
                 std::cout << "p" << n_opt + 1 << ": x=" << std::fixed << std::setprecision(3) << p_planned_vals_optimized[n_opt](0)
                          << ", y=" << p_planned_vals_optimized[n_opt](1)
                          << ", z=" << p_planned_vals_optimized[n_opt](2) << " | mu: [";
                for(int i_reg=0; i_reg < N_REGIONS; ++i_reg) {
                    mu_planned_vals_optimized[n_opt][i_reg] = mu_vars[n_opt][i_reg].get(GRB_DoubleAttr_X);
                    std::cout << (mu_planned_vals_optimized[n_opt][i_reg] > 0.5 ? 1:0) << (i_reg < N_REGIONS-1 ? ",":"");
                }
                std::cout << "]" << std::endl;
            }
            
            // --- Data Export for Plotting ---
            std::ofstream results_file("mpc_results.csv");
            if (results_file.is_open()) {
                results_file << "type,x,y,z,region_idx_if_planned_foot\n"; // Header
                results_file << "current_stance," << p_current_foot_val(0) << "," << p_current_foot_val(1) << "," << p_current_foot_val(2) << ",-1\n";
                for (int n = 0; n < N_horizon; ++n) {
                    int chosen_region = -1;
                    for(int i_reg=0; i_reg < N_REGIONS; ++i_reg) {
                        if (mu_planned_vals_optimized[n][i_reg] > 0.5) {
                            chosen_region = i_reg;
                            break;
                        }
                    }
                    results_file << "planned_foot_p" << (n+1) << ","
                                 << p_planned_vals_optimized[n](0) << ","
                                 << p_planned_vals_optimized[n](1) << ","
                                 << p_planned_vals_optimized[n](2) << ","
                                 << chosen_region << "\n";
                }
                results_file.close();
                std::cout << "Results exported to mpc_results.csv" << std::endl;
            } else {
                std::cerr << "Unable to open mpc_results.csv for writing." << std::endl;
            }

            std::ofstream region_file("region_data.txt");
             if (region_file.is_open()) {
                for(size_t i=0; i < regions_F_list.size(); ++i) {
                    region_file << "Region " << i << std::endl;
                    region_file << "F_matrix:\n" << regions_F_list[i] << std::endl;
                    region_file << "c_vector:\n" << regions_c_list[i] << std::endl << std::endl;
                }
                region_file.close();
                std::cout << "Region data exported to region_data.txt" << std::endl;
            } else {
                std::cerr << "Unable to open region_data.txt for writing." << std::endl;
            }


        } else if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            std::cout << "MPC problem is infeasible." << std::endl;
            model.computeIIS();
            model.write("mpfc_infeasible.ilp");
            std::cout << "IIS written to mpfc_infeasible.ilp" << std::endl;
        } else {
            std::cout << "Optimization ended with status: " << model.get(GRB_IntAttr_Status) << std::endl;
        }

    } catch (GRBException& e) {
        std::cerr << "Gurobi Error code " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}