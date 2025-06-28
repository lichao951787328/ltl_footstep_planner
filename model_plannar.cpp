#define _USE_MATH_DEFINES // For M_PI
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <array>
#include <numeric> // For std::iota, std::accumulate
#include <map>
#include <iomanip> // For std::fixed, std::setprecision
#include <Eigen/LU> // For .lu() decomposition
#include <Eigen/QR> // For .colPivHouseholderQr() if needed for robustness

// ... (other includes and functions) ...
#include "gurobi_c++.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions> // For matrixExponential

#include "matplotlibcpp.h" // For plotting
#include <glog/logging.h>
using namespace std;
namespace plt = matplotlibcpp;

// --- Helper Gurobi quicksum ---
GRBLinExpr gurobi_quicksum(const std::vector<GRBVar>& vars, const Eigen::VectorXd& coeffs) {
    GRBLinExpr expr = 0;
    if (vars.size() != coeffs.size()) {
        throw std::runtime_error("gurobi_quicksum: vars and coeffs size mismatch");
    }
    for (size_t i = 0; i < vars.size(); ++i) {
        expr += vars[i] * coeffs(i);
    }
    return expr;
}

GRBLinExpr gurobi_quicksum(const Eigen::MatrixXd& M, int row, const std::vector<GRBVar>& vars) {
    GRBLinExpr expr = 0;
    if (M.cols() != vars.size()) {
        throw std::runtime_error("gurobi_quicksum matrix: M.cols and vars.size mismatch");
    }
    for (int col = 0; col < M.cols(); ++col) {
        expr += M(row, col) * vars[col];
    }
    return expr;
}

GRBLinExpr gurobi_quicksum_vars(const std::vector<GRBVar>& vars) {
    GRBLinExpr expr = 0;
    for (const auto& var : vars) {
        expr += var;
    }
    return expr;
}


// --- ALIP Dynamics Functions (using Eigen) ---
Eigen::Matrix4d get_autonomous_alip_matrix_A(double H_com, double mass, double g) 
{
    Eigen::Matrix4d A_c_autonomous;
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    return A_c_autonomous;
}

std::pair<Eigen::Matrix4d, Eigen::Vector4d> get_alip_matrices_with_input(double H_com, double mass, double g, double T_ss_dt) 
{
    Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Vector4d B_c_input_effect;
    B_c_input_effect << 0, 0, 0, 1;

    Eigen::Matrix4d A_d_for_mpc = (A_c_autonomous * T_ss_dt).exp();
    Eigen::Vector4d B_d_for_mpc;
    // This is valid if A_c is invertible
    if (std::abs(A_c_autonomous.determinant()) < 1e-9) 
    {
         std::cerr << "Warning: A_c_autonomous is singular or near-singular in get_alip_matrices_with_input. Using numerical integration for B_d." << std::endl;
        // Fallback or more robust method for singular A_c
        // For simple Euler integration of B: B_d approx A_c_autonomous.inverse() * (A_d_for_mpc - I) * B_c is not robust
        // A simple approximation: B_d = B_c_input_effect * T_ss_dt (first order hold for B_c)
        // Or integrate exp(A*s)*B from 0 to T_ss_dt.
        // The Python code's inv approach assumes invertibility. Let's try it but warn.
        // A more robust way to compute int_0^T exp(As) ds B can be done using the matrix exponential of an augmented matrix.
        // M = [[A, B], [0, 0]] -> exp(M*T) = [[Ad, Bd_int], [0, I]] where Bd_int = int_0^T exp(As)B ds
        // However, the python code uses A_c_inv * (A_d - I) * B_c
        B_d_for_mpc = A_c_autonomous.colPivHouseholderQr().solve((A_d_for_mpc - Eigen::Matrix4d::Identity()) * B_c_input_effect);

    } 
    else 
    {
      B_d_for_mpc = A_c_autonomous.inverse() * (A_d_for_mpc - Eigen::Matrix4d::Identity()) * B_c_input_effect;
    }
    return {A_d_for_mpc, B_d_for_mpc};
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 3>> get_alip_reset_map_matrices_detailed(double T_ds, double H_com, double mass, double g) 
{
    Eigen::Matrix4d A_c = get_autonomous_alip_matrix_A(H_com, mass, g);
    Eigen::Matrix4d Ar_ds = (A_c * T_ds).exp();

    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0,
                     0, 0,
                     0, mass * g,
                     -mass * g, 0;

    Eigen::Matrix<double, 4, 2> B_ds = Eigen::Matrix<double, 4, 2>::Zero();
    if (std::abs(A_c.determinant()) < 1e-9 || std::abs(Ar_ds.determinant()) < 1e-9) 
    {
        std::cerr << "Warning: A_c or Ar_ds is singular in get_alip_reset_map_matrices_detailed. B_ds set to zero." << std::endl;
    } 
    else
    {
        Eigen::Matrix4d A_c_inv = A_c.inverse();
        Eigen::Matrix4d Ar_ds_inv = Ar_ds.inverse();
        B_ds = Ar_ds * A_c_inv * ((1.0/T_ds) * A_c_inv * (Eigen::Matrix4d::Identity() - Ar_ds_inv) - Ar_ds_inv) * B_CoP_for_Bds;
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



std::pair<Eigen::Vector4d, Eigen::Vector4d> calculate_periodic_alip_reference_states(
    double vx_d, double vy_d, double stance_width_l,
    double T_s2s, const Eigen::Matrix4d& A_s2s_autonomous_cycle,
    const Eigen::Matrix<double, 4, 2>& Br_map_for_cycle_2d, // Expects 4x2
    bool initial_stance_is_left)
{
    Eigen::Vector2d v_d_vec(vx_d, vy_d);
    double sigma_1 = initial_stance_is_left ? -1.0 : 1.0;
    double sigma_2 = initial_stance_is_left ? 1.0 : -1.0;

    Eigen::Vector2d delta_p_A_2d;
    delta_p_A_2d(0) = v_d_vec(0) * T_s2s;
    delta_p_A_2d(1) = v_d_vec(1) * T_s2s + sigma_1 * stance_width_l;

    Eigen::Vector2d delta_p_B_2d;
    delta_p_B_2d(0) = v_d_vec(0) * T_s2s;
    delta_p_B_2d(1) = v_d_vec(1) * T_s2s + sigma_2 * stance_width_l;

    Eigen::Matrix4d I_mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d M_cycle = A_s2s_autonomous_cycle;
    Eigen::Matrix4d M_sq_cycle = M_cycle * M_cycle;

    Eigen::Matrix4d lhs_matrix = I_mat - M_sq_cycle;
    Eigen::Vector4d rhs_vector = M_cycle * (Br_map_for_cycle_2d * delta_p_A_2d) + (Br_map_for_cycle_2d * delta_p_B_2d);

    Eigen::Vector4d x_start_of_2_step_cycle_ref;

    // Check determinant before attempting LU decomposition or use a more robust solver
    if (std::abs(lhs_matrix.determinant()) < 1e-9) { // Check for singularity
        std::cerr << "Warning: lhs_matrix is singular or near-singular in calculating periodic reference." << std::endl;
        // Option 1: Use a solver robust to rank deficiency (e.g., QR decomposition)
        // x_start_of_2_step_cycle_ref = lhs_matrix.colPivHouseholderQr().solve(rhs_vector);
        // Check if a solution was found (QR can find least-squares solutions)
        // if (!(lhs_matrix * x_start_of_2_step_cycle_ref).isApprox(rhs_vector, 1e-5)) {
        //    std::cerr << "  QR solver did not find an exact solution. Using zeros." << std::endl;
        //    x_start_of_2_step_cycle_ref.setZero();
        // } else {
        //    std::cerr << "  Using QR decomposition based solution." << std::endl;
        // }

        // Option 2: Fallback to zeros (as in your Python version's except block)
        std::cerr << "  Using zeros for periodic reference states due to singular matrix." << std::endl;
        x_start_of_2_step_cycle_ref.setZero();

    } else {
        // For well-conditioned, invertible square matrices, LU decomposition is common.
        x_start_of_2_step_cycle_ref = lhs_matrix.lu().solve(rhs_vector);
        // You can also check if the solution is valid:
        // if(!(lhs_matrix*x_start_of_2_step_cycle_ref).isApprox(rhs_vector, 1e-5)) {
        //    std::cerr << "Warning: Solution from LU decomposition might be inaccurate." << std::endl;
        // }
    }

    Eigen::Vector4d ref_state_phase0 = x_start_of_2_step_cycle_ref;
    Eigen::Vector4d ref_state_phase1 = M_cycle * ref_state_phase0 + Br_map_for_cycle_2d * delta_p_A_2d;

    return {ref_state_phase0, ref_state_phase1};
}


// --- Plotting Function (Simplified for Bounding Boxes) ---
void plot_results_cpp(
    const Eigen::Vector3d& p_current_val_eigen,
    const std::vector<Eigen::Vector3d>& p_planned_vals_eigen,
    const std::vector<std::vector<double>>& mu_planned_vals, // mu[n][i_reg]
    const std::vector<Eigen::MatrixXd>& regions_F_list_eigen, // F should be 6x3
    const std::vector<Eigen::VectorXd>& regions_c_list_eigen, // c should be 6x1
    int N_horizon_plot, int N_REGIONS_plot) 
{
    plt::figure_size(1000, 800);
    // For 3D plot, pass {{"projection", "3d"}} to subplot
    // plt::subplot(1, 1, 1, {{"projection", "3d"}}); // This might not work directly with older matplotlib-cpp
    // Instead, use plt::plot3, plt::scatter3, etc.

    // 1. Plot initial footstep
    std::vector<double> p_curr_x = {p_current_val_eigen(0)};
    std::vector<double> p_curr_y = {p_current_val_eigen(1)};
    std::vector<double> p_curr_z = {p_current_val_eigen(2)};
    std::map<std::string, std::string> current_stance_kwargs;
    current_stance_kwargs["c"] = "black";
    current_stance_kwargs["marker"] = "x";
    // current_stance_kwargs["s"] = "100"; // Size might need to be double
    current_stance_kwargs["label"] = "Current Stance Foot (p0)";
    plt::scatter(p_curr_x, p_curr_y, 100.0, current_stance_kwargs); // Using 2D scatter, adapt if true 3D needed


    // 2. Plot planned footsteps and connecting lines
    std::vector<Eigen::Vector3d> all_ps_eigen = {p_current_val_eigen};
    all_ps_eigen.insert(all_ps_eigen.end(), p_planned_vals_eigen.begin(), p_planned_vals_eigen.end());

    for (int n = 0; n < N_horizon_plot; ++n) {
        const Eigen::Vector3d& p_n = all_ps_eigen[n];
        const Eigen::Vector3d& p_np1 = all_ps_eigen[n+1];
        
        int chosen_region_idx = -1;
        for (int i_reg = 0; i_reg < N_REGIONS_plot; ++i_reg) {
            if (mu_planned_vals[n][i_reg] > 0.5) {
                chosen_region_idx = i_reg;
                break;
            }
        }
        
        std::string color_str = "gray";
        if (chosen_region_idx != -1) {
            // Simple color cycling for regions
            std::vector<std::string> region_plot_colors = {"blue", "green", "red", "purple", "orange"};
            color_str = region_plot_colors[chosen_region_idx % region_plot_colors.size()];
        }

        std::map<std::string, std::string> planned_kwargs;
        planned_kwargs["c"] = color_str;
        planned_kwargs["marker"] = "o";
        // planned_kwargs["s"] = "80";
        if (n == 0) planned_kwargs["label"] = "Planned p" + std::to_string(n+1) + " (Region " + std::to_string(chosen_region_idx+1) + ")";
        
        // plt::scatter({p_np1(0)}, {p_np1(1)}, 80.0, planned_kwargs); // 2D scatter
        plt::plot({p_n(0), p_np1(0)}, {p_n(1), p_np1(1)}, {{"color", "gray"}, {"linestyle", "--"}}); // 2D plot
    }

    // 3. Plot foothold regions (as bounding boxes)
    std::vector<std::string> region_face_colors = {"skyblue", "lightgreen", "lightcoral", "plum", "navajowhite"};

    for (int i_reg = 0; i_reg < N_REGIONS_plot; ++i_reg) {
        // Assuming F_region_common and c_vec defining axis-aligned box
        // c_vec: [x_max, -x_min, y_max, -y_min, z_max, -z_min]
        const Eigen::VectorXd& c_vec = regions_c_list_eigen[i_reg];
        double x_max_val = c_vec(0);
        double x_min_val = -c_vec(1);
        double y_max_val = c_vec(2);
        double y_min_val = -c_vec(3);
        // double z_max_val = c_vec(4); // For 3D
        // double z_min_val = -c_vec(5); // For 3D

        // For 2D plot:
        std::vector<double> rect_x = {x_min_val, x_max_val, x_max_val, x_min_val, x_min_val};
        std::vector<double> rect_y = {y_min_val, y_min_val, y_max_val, y_max_val, y_min_val};
        
        std::map<std::string, std::string> region_fill_kwargs;
        region_fill_kwargs["facecolor"] = region_face_colors[i_reg % region_face_colors.size()];
        region_fill_kwargs["alpha"] = "0.3";
        region_fill_kwargs["edgecolor"] = "k";
        region_fill_kwargs["label"] = "Region " + std::to_string(i_reg+1);
        plt::fill(rect_x, rect_y, region_fill_kwargs);
    }

    plt::xlabel("X world (m)");
    plt::ylabel("Y world (m)");
    // plt::zlabel("Z world (m)"); // If 3D
    plt::title("MPC Planned Footholds and Regions (2D Projection)");
    plt::legend();
    plt::grid(true);
    // plt::axis("equal"); // For 2D
    
    // Attempt to set reasonable limits (can be tricky without true 3D projection)
    double min_x=p_current_val_eigen(0), max_x=p_current_val_eigen(0);
    double min_y=p_current_val_eigen(1), max_y=p_current_val_eigen(1);

    for(const auto& p : all_ps_eigen) {
        min_x = std::min(min_x, p(0)); max_x = std::max(max_x, p(0));
        min_y = std::min(min_y, p(1)); max_y = std::max(max_y, p(1));
    }
     for (int i_reg = 0; i_reg < N_REGIONS_plot; ++i_reg) {
        const Eigen::VectorXd& c_vec = regions_c_list_eigen[i_reg];
        min_x = std::min(min_x, -c_vec(1)); max_x = std::max(max_x, c_vec(0));
        min_y = std::min(min_y, -c_vec(3)); max_y = std::max(max_y, c_vec(2));
    }
    double xrange = max_x - min_x; double yrange = max_y - min_y;
    double padding = std::max(xrange, yrange) * 0.1 + 0.5; // Add padding
    plt::xlim(min_x - padding, max_x + padding);
    plt::ylim(min_y - padding, max_y + padding);

    plt::save("mpfc_cpp_footsteps.png");
    // plt::show(); // Can be blocking
    std::cout << "Plot saved to mpfc_cpp_footsteps.png" << std::endl;
}


// --- Main Function ---
int main(int argc, char *argv[]) 
{
    try {
        // --- Model Parameters ---
        const int N_horizon = 9;
        const int K_knots = 10;
        const double T_ss = 0.4;
        const double T_ds = 0.1;
        const double T_ss_dt = T_ss / (K_knots - 1);
        const double T_s2s_cycle = T_ss + T_ds;

        const double mass = 30.0;
        const double g = 9.81;
        const double H_com_nominal = 0.8;

        // --- Get Dynamics Matrices ---
        auto [A_d_mpc, B_d_mpc_vec] = get_alip_matrices_with_input(H_com_nominal, mass, g, T_ss_dt);
        Eigen::MatrixXd B_d_mpc = B_d_mpc_vec; // Convert 4x1 vector to 4x1 matrix for consistency
        
        auto [Ar_reset, Br_reset_delta_p] = get_alip_reset_map_matrices_detailed(T_ds, H_com_nominal, mass, g);
        
        Eigen::Matrix4d A_c_autonomous = get_autonomous_alip_matrix_A(H_com_nominal, mass, g);
        Eigen::Matrix4d A_d_autonomous_knot = (A_c_autonomous * T_ss_dt).exp();
        Eigen::Matrix4d A_s2s_autonomous_cycle = (A_c_autonomous * T_s2s_cycle).exp();


        // 如果这个速度是一个变化的，该怎么办？，就是每步的速度不一样

        // const double vx_desired = 0.15;
        // const double vy_desired = 0.0;

        std::vector<std::pair<double, double>> desired_velocities_horizon(N_horizon);
        double vx_start = 0.1;
        double vx_end = 0.3;
        for (int n = 0; n < N_horizon; ++n) 
        {
            double current_vx = vx_start + (vx_end - vx_start) * static_cast<double>(n) / std::max(1, N_horizon - 1);
            // desired_velocities_horizon[n] = {current_vx, 0.0}; // vy_desired is 0.0
            if (n %2 == 0)
            {
                desired_velocities_horizon[n] = {current_vx, -0.02}; 
            }
            else
            {
                desired_velocities_horizon[n] = {current_vx, 0.02}; 
            }
            
        }
        
        for (auto & v : desired_velocities_horizon)
        {
            cout<<"v: "<<v.first<<" "<<v.second<<endl;
            // v.first = 0.1;
            // v.second = 0.0; // Set vy_desired to 0.0
        }
        

        const double nominal_stance_width = 0.2;

        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.set(GRB_StringAttr_ModelName, "acosta_mpc_revised_cpp");

        // --- Create Gurobi Variables ---
        // x_alip_vars[n][k][state_idx]
        std::vector<std::vector<std::vector<GRBVar>>> x_alip_vars(
            N_horizon, std::vector<std::vector<GRBVar>>(
                           K_knots, std::vector<GRBVar>(4)));
        // u_ankle_vars[n][k]
        std::vector<std::vector<GRBVar>> u_ankle_vars(
            N_horizon, std::vector<GRBVar>(K_knots - 1));
        
        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots; ++k) {
                for (int i = 0; i < 4; ++i) {
                    x_alip_vars[n][k][i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 
                                                        "x_n" + std::to_string(n) + "_k" + std::to_string(k) + "_" + std::to_string(i));
                }
                if (k < K_knots - 1) {
                    u_ankle_vars[n][k] = model.addVar(-5.0, 5.0, 0.0, GRB_CONTINUOUS,
                                                      "u_n" + std::to_string(n) + "_k" + std::to_string(k));
                }
            }
        }

        // p_foot_vars[n][coord_idx] -> p_{n+1}
        std::vector<std::vector<GRBVar>> p_foot_vars(N_horizon, std::vector<GRBVar>(3));
        for (int n = 0; n < N_horizon; ++n) {
            for (int j = 0; j < 3; ++j) {
                p_foot_vars[n][j] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                                 "p_n" + std::to_string(n+1) + "_" + std::to_string(j));
            }
        }

        const int N_REGIONS = 2;
        // mu_vars[n][region_idx] -> for p_{n+1}
        std::vector<std::vector<GRBVar>> mu_vars(N_horizon, std::vector<GRBVar>(N_REGIONS));
        for (int n = 0; n < N_horizon; ++n) {
            for (int i_reg = 0; i_reg < N_REGIONS; ++i_reg) {
                mu_vars[n][i_reg] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                                 "mu_n" + std::to_string(n+1) + "_reg" + std::to_string(i_reg));
            }
        }
        model.update(); // Important after adding vars

        // --- Define Perception Data (Regions) ---
        Eigen::Matrix<double, 6, 3> F_region_common;
        F_region_common << 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1;
        
        Eigen::Matrix<double, 6, 1> c_region1_vec, c_region2_vec;
        c_region1_vec << 1, 0.5, 1, 1, 0.01, -0.01;
        c_region2_vec << 5, -1.1, 1, 1, 0.1, -0.1;
        
        std::vector<Eigen::MatrixXd> regions_F_list = {F_region_common, F_region_common};
        std::vector<Eigen::VectorXd> regions_c_list = {c_region1_vec, c_region2_vec};

        // --- Initial Conditions & Reference Trajectory ---
        Eigen::Vector3d p_current_foot_val_eigen(0.0, 0.1, 0.01);
        bool current_stance_is_left = true;

        Eigen::Matrix<double, 4, 2> Br_map_for_cycle_2d = Br_reset_delta_p.block<4,2>(0,0);
        
        auto [ref_state_cycle_phase0, ref_state_cycle_phase1] = calculate_periodic_alip_reference_states(
            desired_velocities_horizon.front().first, desired_velocities_horizon.front().second, nominal_stance_width,
            T_s2s_cycle, A_s2s_autonomous_cycle, Br_map_for_cycle_2d,
            current_stance_is_left
        );
        // 为了保证后续的落脚点的连续稳定性，初始com使用理想的参考com
        Eigen::Vector4d x_current_alip_val_eigen = current_stance_is_left ? ref_state_cycle_phase0 : ref_state_cycle_phase1;
        
        for (int i = 0; i < 4; ++i) {
            model.addConstr(x_alip_vars[0][0][i] == x_current_alip_val_eigen(i), "init_alip_state_" + std::to_string(i));
        }
        

        // std::vector<std::vector<Eigen::Vector4d>> x_d_horizon(N_horizon, std::vector<Eigen::Vector4d>(K_knots));
        // for (int n_mpc = 0; n_mpc < N_horizon; ++n_mpc) {
        //     Eigen::Vector4d x_d_nk_start_of_stage = (n_mpc % 2 == (current_stance_is_left ? 0:1) ) ? ref_state_cycle_phase0 : ref_state_cycle_phase1;
        //     cout<<x_d_nk_start_of_stage.transpose()<<endl;
        //     Eigen::Vector4d x_d_nk = x_d_nk_start_of_stage;
        //     x_d_horizon[n_mpc][0] = x_d_nk;
        //     for (int k_knot = 1; k_knot < K_knots; ++k_knot) {
        //         x_d_nk = A_d_autonomous_knot * x_d_nk;
        //         x_d_horizon[n_mpc][k_knot] = x_d_nk;
        //     }
        // }

        // Precompute x_d_horizon
        std::vector<std::vector<Eigen::Vector4d>> x_d_horizon(N_horizon, std::vector<Eigen::Vector4d>(K_knots));
        for (int n = 0; n < N_horizon; ++n) 
        {
            // For each stage n, calculate its specific periodic reference states
            // The stance leg for calculating this stage's reference depends on the *previous* stage's swing
            // If current_stance_is_left is true for the *overall MPC problem's start*:
            // Stage 0 reference uses current_stance_is_left
            // Stage 1 reference uses !current_stance_is_left
            // Stage k reference uses current_stance_is_left if k is even, !current_stance_is_left if k is odd.
            // 1. 确定为计算阶段 n 的参考周期，应该假设哪只脚是名义上的起始支撑脚
            bool is_left_stance_for_stage_n_dynamics = (n % 2 == 0) ? current_stance_is_left : !current_stance_is_left;
            cout<<"n: "<<n<<" is_left_stance_for_stage_n_dynamics: "<<is_left_stance_for_stage_n_dynamics<<endl;
            // 2. 根据阶段 n 的期望速度 和 上面确定的名义支撑脚，计算该阶段的参考周期状态
            auto [cycle_p0_for_stage_n, cycle_p1_for_stage_n] = calculate_periodic_alip_reference_states(
                desired_velocities_horizon[n].first,  // vx_d for stage n
                desired_velocities_horizon[n].second, // vy_d for stage n
                nominal_stance_width,
                T_s2s_cycle, A_s2s_autonomous_cycle, Br_map_for_cycle_2d,
                is_left_stance_for_stage_n_dynamics // 这个参数是关键
            );
            // cout<<"cycle_p0_for_stage_n: " << cycle_p0_for_stage_n.transpose() << endl;
            // cout<<"cycle_p1_for_stage_n: " << cycle_p1_for_stage_n.transpose() << endl;
            // 3. 阶段 n 的单支撑初始参考状态 x_d_horizon[n][0] 总是 cycle_p0_for_stage_n
            //    因为 cycle_p0_for_stage_n 已经是根据 is_left_stance_for_stage_n_dynamics 调整过的
            //    “正确”的起始相位了。
            Eigen::Vector4d x_d_nk_start_of_stage = cycle_p0_for_stage_n;
            // if (is_left_stance_for_stage_n_dynamics)
            // {
            //     x_d_nk_start_of_stage = cycle_p0_for_stage_n;
            // }
            // else
            // {
            //     x_d_nk_start_of_stage = cycle_p1_for_stage_n;
            // }
            cout<<"x_d_nk_start_of_stage: " << x_d_nk_start_of_stage.transpose()<<endl;
            // 4. 在该阶段内自主传播
            Eigen::Vector4d x_d_nk = x_d_nk_start_of_stage;
            x_d_horizon[n][0] = x_d_nk;
            for (int k = 1; k < K_knots; ++k) {
                x_d_nk = A_d_autonomous_knot * x_d_nk;
                x_d_horizon[n][k] = x_d_nk;
            }
        }
        

        // --- Constraints ---
        // 9b: ALIP dynamics within a single support phase
        for (int n = 0; n < N_horizon; ++n) 
        {
            for (int k = 0; k < K_knots - 1; ++k) 
            {
                for (int row = 0; row < 4; ++row) 
                {
                    model.addConstr(
                        x_alip_vars[n][k+1][row] ==
                        gurobi_quicksum(A_d_mpc, row, x_alip_vars[n][k]) +
                        B_d_mpc(row,0) * u_ankle_vars[n][k],
                        "alip_dyn_n" + std::to_string(n) + "_k" + std::to_string(k) + "_r" + std::to_string(row)
                    );
                }
            }
        }
        
        // 9c: ALIP reset map (double support phase dynamics)
        for (int n = 0; n < N_horizon - 1; ++n) {
            GRBLinExpr dp_x, dp_y, dp_z;
            if (n == 0) 
            {
                dp_x = p_foot_vars[n][0] - p_current_foot_val_eigen(0);
                dp_y = p_foot_vars[n][1] - p_current_foot_val_eigen(1);
                dp_z = p_foot_vars[n][2] - p_current_foot_val_eigen(2);
            }
            else 
            {
                dp_x = p_foot_vars[n][0] - p_foot_vars[n-1][0];
                dp_y = p_foot_vars[n][1] - p_foot_vars[n-1][1];
                dp_z = p_foot_vars[n][2] - p_foot_vars[n-1][2];
            }

            for (int row = 0; row < 4; ++row) 
            {
                GRBLinExpr br_term = Br_reset_delta_p(row,0) * dp_x +
                                     Br_reset_delta_p(row,1) * dp_y +
                                     Br_reset_delta_p(row,2) * dp_z;
                model.addConstr(
                    x_alip_vars[n+1][0][row] ==
                    gurobi_quicksum(Ar_reset, row, x_alip_vars[n][K_knots-1]) +
                    br_term,
                    "alip_reset_n" + std::to_string(n) + "_r" + std::to_string(row)
                );
            }
        }

        // 9d: Foothold region constraints
        const double M_big = 100.0;
        for (int n = 0; n < N_horizon; ++n) 
        { // p_foot_vars[n] is p_{n+1}
            for (int i_reg = 0; i_reg < N_REGIONS; ++i_reg) 
            {
                const Eigen::MatrixXd& F_mat = regions_F_list[i_reg]; // Should be 6x3
                const Eigen::VectorXd& c_vec = regions_c_list[i_reg]; // Should be 6x1
                for (int row_idx = 0; row_idx < F_mat.rows(); ++row_idx) 
                {
                    GRBLinExpr Fp_expr = 0;
                    for (int col_idx = 0; col_idx < 3; ++col_idx) 
                    {
                        Fp_expr += F_mat(row_idx, col_idx) * p_foot_vars[n][col_idx];
                    }
                    model.addConstr( Fp_expr <= c_vec(row_idx) + M_big * (1.0 - mu_vars[n][i_reg]),
                        "foothold_n" + std::to_string(n+1) + "_reg" + std::to_string(i_reg) + "_row" + std::to_string(row_idx)
                    );
                }
            }
            GRBLinExpr sum_mu = 0;
            for(int i_reg=0; i_reg < N_REGIONS; ++i_reg) sum_mu += mu_vars[n][i_reg];
            model.addConstr(sum_mu == 1.0, "sum_mu_n" + std::to_string(n+1));
        }

        // CoM y-excursion limits
        const double y_com_max = 0.15;
        for (int n = 0; n < N_horizon; ++n) 
        {
            for (int k = 0; k < K_knots; ++k) 
            {
                model.addConstr(x_alip_vars[n][k][1] <= y_com_max, "ycom_max_n" + std::to_string(n) + "_k" + std::to_string(k));
                model.addConstr(x_alip_vars[n][k][1] >= -y_com_max, "ycom_min_n" + std::to_string(n) + "_k" + std::to_string(k));
            }
        }

        // Kinematic limits (p_{n+1} - p_n)
        const double max_dx = 0.4, max_dy = 0.3;
        // std::vector<GRBVar> prev_p_for_kin_limit_vars(3); // Not needed if p_current is const
        Eigen::Vector3d prev_p_val_for_kin = p_current_foot_val_eigen;

        for (int n = 0; n < N_horizon; ++n) 
        { // p_foot_vars[n] is p_{n+1}
            GRBLinExpr dx, dy;
            if (n == 0) 
            {
                 dx = p_foot_vars[n][0] - prev_p_val_for_kin(0);
                 dy = p_foot_vars[n][1] - prev_p_val_for_kin(1);
            }
            else 
            {
                 dx = p_foot_vars[n][0] - p_foot_vars[n-1][0];
                 dy = p_foot_vars[n][1] - p_foot_vars[n-1][1];
            }
            model.addConstr(dx <= max_dx, "max_dx_n" + std::to_string(n+1));
            model.addConstr(dx >= -max_dx, "min_dx_n" + std::to_string(n+1));
            model.addConstr(dy <= max_dy, "max_dy_n" + std::to_string(n+1));
            model.addConstr(dy >= -max_dy, "min_dy_n" + std::to_string(n+1));
            // prev_p_for_kin_limit_vars becomes p_foot_vars[n] for next iter, handled by n-1 indexing
        }

        // --- Objective Function ---
        GRBQuadExpr objective = 0;
        Eigen::Matrix4d Q_state = Eigen::Matrix4d::Identity(); // [1,1,1,1] diag
        double R_input_val = 0.1;

        for (int n = 0; n < N_horizon; ++n) {
            for (int k = 0; k < K_knots; ++k) {
                for (int r_idx = 0; r_idx < 4; ++r_idx) {
                    GRBLinExpr err_r = x_alip_vars[n][k][r_idx] - x_d_horizon[n][k](r_idx);
                    objective += err_r * Q_state(r_idx, r_idx) * err_r;
                }
                if (k < K_knots - 1) {
                    objective += u_ankle_vars[n][k] * R_input_val * u_ankle_vars[n][k];
                }
            }
        }
        
        Eigen::Matrix4d Q_f_state = Q_state * 0.1;
        for (int r_idx = 0; r_idx < 4; ++r_idx) {
            GRBLinExpr err_f_r = x_alip_vars[N_horizon-1][K_knots-1][r_idx] - x_d_horizon[N_horizon-1][K_knots-1](r_idx);
            objective += err_f_r * Q_f_state(r_idx, r_idx) * err_f_r;
        }
        model.setObjective(objective, GRB_MINIMIZE);

        // --- Solve ---
        model.set(GRB_DoubleParam_MIPGap, 0.05);
        model.set(GRB_DoubleParam_TimeLimit, 2.5); // Slightly more time for C++ overhead potentially
        // model.set(GRB_IntParam_NonConvex, 2); // Only if quadratic constraints
        
        model.optimize();

        // --- Process and Plot Results ---
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) {
            std::cout << "MPC solution found." << std::endl;
            std::cout << "Gurobi optimization time: " << std::fixed << std::setprecision(4) << model.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
            
            std::vector<Eigen::Vector3d> p_planned_vals_optimized_eigen(N_horizon);
            std::cout << "\n--- Planned Footstep Sequence (p1, p2, ...) ---" << std::endl;
            for (int n_opt = 0; n_opt < N_horizon; ++n_opt) {
                for (int j = 0; j < 3; ++j) {
                    p_planned_vals_optimized_eigen[n_opt](j) = p_foot_vars[n_opt][j].get(GRB_DoubleAttr_X);
                }
                std::cout << "p" << n_opt + 1 << ": x=" << std::fixed << std::setprecision(3) << p_planned_vals_optimized_eigen[n_opt](0)
                          << ", y=" << p_planned_vals_optimized_eigen[n_opt](1)
                          << ", z=" << p_planned_vals_optimized_eigen[n_opt](2) << std::endl;
            }
            
            std::vector<std::vector<double>> mu_planned_vals_optimized(N_horizon, std::vector<double>(N_REGIONS));
            for (int n_opt = 0; n_opt < N_horizon; ++n_opt) {
                for (int i_reg = 0; i_reg < N_REGIONS; ++i_reg) {
                    mu_planned_vals_optimized[n_opt][i_reg] = mu_vars[n_opt][i_reg].get(GRB_DoubleAttr_X);
                }
            }

            std::cout << "Optimal first planned footstep (p1): (" 
                      << std::fixed << std::setprecision(3) << p_planned_vals_optimized_eigen[0](0) << ", "
                      << p_planned_vals_optimized_eigen[0](1) << ", "
                      << p_planned_vals_optimized_eigen[0](2) << ")" << std::endl;

            for (int i_reg_check = 0; i_reg_check < N_REGIONS; ++i_reg_check) {
                if (mu_planned_vals_optimized[0][i_reg_check] > 0.5) {
                    std::cout << "Footstep p1 planned for region " << i_reg_check + 1 << std::endl;
                    break;
                }
            }
            
            plot_results_cpp(p_current_foot_val_eigen,
                             p_planned_vals_optimized_eigen,
                             mu_planned_vals_optimized,
                             regions_F_list, // Already Eigen matrices
                             regions_c_list, // Already Eigen vectors
                             N_horizon, N_REGIONS);

        } else if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
            std::cerr << "MPC problem is infeasible." << std::endl;
            model.computeIIS();
            model.write("mpfc_infeasible_cpp.ilp");
            std::cerr << "IIS written to mpfc_infeasible_cpp.ilp" << std::endl;
        } else {
            std::cerr << "Optimization ended with status: " << model.get(GRB_IntAttr_Status) << std::endl;
        }

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