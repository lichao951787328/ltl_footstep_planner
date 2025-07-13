#pragma once
#include <vector>
#include <string>
#include "gurobi_c++.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// 简单的多边形结构体，和之前一样
struct polygon3D
{
    std::vector<Eigen::Vector3d> vertices;
    std::pair<Eigen::MatrixX3d, Eigen::VectorXd> calculateCoefficients();
};

class MonolithicPlanner
{
private:
    // --- Parameters ---
    double H_com = 0.65; // BHR8P
    // double H_com = 0.8; // Height of the center of mass
    double mass = 64; // BHR8P
    double g = 9.81;
    const double T_ss = 0.96;
    const double T_ds = 0.24;
    const int K_knots = 10;
    double T_ss_dt;

    // IMPORTANT: Reduced horizon for tractability
    const int N = 14; 
    int N_REG = 5;
    const double M_BIG = 50.0;

    double deta1 = 0.05, deta2 = 0.55, dis_th = 0.43;

    // --- ALIP Matrices ---
    Eigen::Matrix4d A_c_autonomous;
    Eigen::Matrix4d A_d_autonomous_knot;
    Eigen::Matrix<double, 4, 2> B_d_for_mpc;
    Eigen::Matrix4d Ar_ds;
    Eigen::Matrix<double, 4, 3> B_r;

    // --- Initial Conditions ---
    Eigen::Matrix<double, 4, 2> initial_stance_footsteps;
    Eigen::Vector4d initial_alip_x;
    std::vector<polygon3D> polygons;

    // --- Gurobi Model ---
    GRBEnv env;
    GRBModel model;

    // --- Decision Variables --- 所有footstep的变量
    std::vector<std::vector<GRBVar>> p_foot_vars; // x, y, z
    std::vector<GRBVar> theta_vars; // theta
    std::vector<GRBVar> sin_vars, cos_vars; // sin cos 

    // 区域选择变量，角度近似变量
    std::vector<std::vector<GRBVar>> H_vars, S_vars, C_vars;
    // alip状态变量
    std::vector<std::vector<std::vector<GRBVar>>> x_alip_vars;
    std::vector<std::vector<GRBVar>> u_ankle_vars, v_ankle_vars;

    // --- Helper Methods ---
    void initialize_parameters();
    void create_gurobi_variables();
    void add_constraints();
    void set_objective();
    GRBLinExpr gurobi_quicksum(const Eigen::MatrixXd& M, int row, const std::vector<GRBVar>& vars);


public:
    MonolithicPlanner();
    void run();
};