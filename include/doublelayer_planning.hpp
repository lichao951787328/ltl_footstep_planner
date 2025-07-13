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

struct polygon3D
{
    Eigen::Vector3d normal; // Normal vector of the polygon
    std::vector<Eigen::Vector3d> vertices; // Vertices of the polygon
    Eigen::MatrixX3d F; 
    Eigen::VectorXd C; // Coefficients for the linear inequality constraints
    std::pair<Eigen::MatrixX3d, Eigen::VectorXd> calculateCoefficients()
    {
        if (vertices.size() < 3) {
            throw std::runtime_error("A polygon must have at least 3 vertices.");
        }
        double d = -normal.dot(vertices.at(0)); // Distance from the origin to the plane

        F.resize(vertices.size() + 2, 3);
        C.resize(vertices.size() + 2);

        F.row(0) = normal.transpose();
        C(0) = -d;
        F.row(1) = - normal.transpose();
        C(1) = d;

        for (int i = 0; i < vertices.size(); ++i) 
        {
            Eigen::Vector3d v1 = vertices.at(i);
            Eigen::Vector3d v2 = vertices.at((i + 1) % vertices.size()); // Wrap around to the first vertex
            if ((v2-v1).norm() < 1e-3) continue; // 忽略重复顶点
            Eigen::Vector2d edge = (v2 - v1).head(2).normalized(); // Take only the first two dimensions for 2D polygon
            // Rotate edge counterclockwise by 90 degrees
            Eigen::Vector2d rotated_edge(edge.y(), -edge.x());
            double c = rotated_edge.x() * v1.x() + rotated_edge.y() * v1.y(); // Coefficient for the linear inequality constraint
            // F.row(i + 2) = rotated_edge.transpose();
            F.row(i + 2) <<rotated_edge.x(), rotated_edge.y(), 0; // Set z-component to 0 for 2D polygon
            C(i + 2) = c;
        }
        return {F, C};
    }
};



class doublelayer_planning
{
private:
    // 机器人参数 alip模型参数
    double H_com = 0.65; // BHR8P
    // double H_com = 0.8; // Height of the center of mass
    double mass = 64; // BHR8P
    // double mass = 30; // Mass of the robot
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

    const double T_ss = 0.96;// 1.2 * 0.8 BHR8P
    const double T_ds = 0.24;// 1.2 * 0.2 BHR8P
    // const double T_ss = 0.9;
    // const double T_ds = 0.1;
    const int K_knots = 10;
    double T_ss_dt;
    double nominal_stance_width = 0.2;

    // Parameters
    int N = 14; // Number of footsteps
    int N_REG; // Number of safe regions
    const double M_BIG = 50.0; // Big M for indicator constraints

    // 初始步态
    Eigen::Matrix<double, 6, 2> initial_stance_footsteps; // Initial stance footstep positions
    Eigen::Vector4d initial_alip_x;
    // 环境多边形约束
    std::vector<std::array<std::array<double, 3>, 6>> A_regions;
    std::vector<std::array<double, 6>> b_regions;
    std::vector<Eigen::MatrixXd> regions_F_list;
    std::vector<Eigen::VectorXd> regions_c_list;

    // std::vector<polygon3D> polygons; // For 3D polygons

    // Footsteps: x, y, z, theta, sin(theta), cos(theta)
    // 实际上的变量关联顺序是从theta和S_vars来确定sin(theta)
    // theta 和 C_vars来确定cos(theta)
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

std::vector<polygon3D> polygons; // For 3D polygons
// std::vector<std::vector<GRBVar>> footsteps;
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
    // void plot_results();
    ~doublelayer_planning();
};


