#pragma once    
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


class doublelayer_planning
{
private:
    // 机器人参数 alip模型参数
    double H_com = 0.8; // Height of the center of mass
    double mass = 30; // Mass of the robot
    double g = 9.81; // Gravitational acceleration

    // A
    Eigen::Matrix4d A_c_autonomous;
    // A dynamic single swing
    Eigen::Matrix4d A_d_autonomous_knot;
    Eigen::Matrix<double, 4, 2> B_d_for_mpc;

    // reset
    Eigen::Matrix4d Ar_ds;
    Eigen::Matrix<double, 4, 3> B_r;

    // reference
    Eigen::Matrix4d A_s2s_autonomous_cycle;

    const double T_ss = 0.9;
    const double T_ds = 0.1;
    const int K_knots = 10;
    double T_ss_dt;
    double nominal_stance_width = 0.2;

    // Parameters
    const int N = 18; // Number of footsteps
    const int N_REG = 5; // Number of safe regions
    const double M_BIG = 20.0; // Big M for indicator constraints

    // 初始步态
    Eigen::Matrix<double, 5, 2> initial_stance_footsteps; // Initial stance footstep positions
    Eigen::Vector4d initial_alip_x;
    // 环境多边形约束
    std::vector<std::array<std::array<double, 3>, 6>> A_regions;
    std::vector<std::array<double, 6>> b_regions;
    std::vector<Eigen::MatrixXd> regions_F_list;
    std::vector<Eigen::VectorXd> regions_c_list;

    // Footsteps: x, y, theta, sin(theta), cos(theta)
    std::vector<std::vector<GRBVar>> footsteps;
    std::vector<std::vector<GRBVar>> S_vars; // For sin
    std::vector<std::vector<GRBVar>> C_vars; // For cos
    std::vector<std::vector<GRBVar>> H_vars; //safe regions 

    // Reachability parameters
    double deta1 = 0.05, deta2 = 0.55, dis_th = 0.43;

    // 上层输出的是速度
    std::vector<Eigen::Vector2d> centroids_velocity;

    // lower layer
    bool current_stance_is_left = false; // Initial stance
    
    std::vector<std::vector<std::vector<GRBVar>>> x_alip_vars;
    std::vector<std::vector<GRBVar>> v_ankle_vars;//roll 
    std::vector<std::vector<GRBVar>> u_ankle_vars;//pitch
    std::vector<std::vector<GRBVar>> p_foot_vars; // footstep for the lower layer
    std::vector<std::vector<GRBVar>> mu_vars; // region 
    // 参考轨迹
    std::vector<std::vector<Eigen::Vector4d>> x_d_horizon; // Reference trajectory for the lower layer

    // 双层整数规划
    GRBEnv env_upper = GRBEnv();
    GRBModel model_upper = GRBModel(env_upper);


    GRBEnv env_lower = GRBEnv();
    GRBModel model_lower = GRBModel(env_lower);

    inline GRBLinExpr quicksum(const std::vector<GRBVar>& vars, const std::vector<double>& coeffs) 
    {
        GRBLinExpr expr = 0;
        if (vars.size() != coeffs.size()) {
            throw std::runtime_error("quicksum: vars and coeffs size mismatch");
        }
        for (size_t i = 0; i < vars.size(); ++i) {
            expr += vars[i] * coeffs[i];
        }
        return expr;
    }

    inline GRBLinExpr quicksum(const std::vector<GRBVar>& vars) 
    {
        GRBLinExpr expr = 0;
        for (const auto& var : vars) {
            expr += var;
        }
        return expr;
    }
public:
    doublelayer_planning(/* args */);

    void get_autonomous_alip_matrix_A();
    void get_alip_matrices_with_input();
    void get_alip_matrices();
    void get_alip_reset_map_matrices_detailed();
    std::pair<Eigen::Vector4d, Eigen::Vector4d> calculate_periodic_alip_reference_states(
        double vx_d, double vy_d, double stance_width_l,
        double T_s2s, const Eigen::Matrix4d& A_s2s_autonomous_cycle,
        const Eigen::Matrix<double, 4, 2>& Br_map_for_cycle_2d, // Expects 4x2
        bool initial_stance_is_left);

    void add_upper_layer_variables();
    void add_lower_layer_variables();
    void add_upper_layer_constraints();
    void add_lower_layer_constraints();
    void solve_upper_layer();
    void solve_lower_layer();

    void run();
    void initial_env();
    ~doublelayer_planning();
};


