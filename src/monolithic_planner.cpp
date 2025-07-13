#include "monolithic_planner.hpp"
#include <iostream>
#include <iomanip>
#include <fstream> 
// #include <fstream>
// --- polygon3D Implementation ---
std::pair<Eigen::MatrixX3d, Eigen::VectorXd> polygon3D::calculateCoefficients() {
    Eigen::MatrixX3d F;
    Eigen::VectorXd C;
    Eigen::Vector3d normal(0, 0, 1);
    double d = -normal.dot(vertices.at(0));
    F.resize(vertices.size() + 2, 3);
    C.resize(vertices.size() + 2);
    F.row(0) = normal.transpose(); C(0) = -d;
    F.row(1) = -normal.transpose(); C(1) = d;

    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector3d v1 = vertices.at(i);
        Eigen::Vector3d v2 = vertices.at((i + 1) % vertices.size());
        Eigen::Vector2d edge = (v2 - v1).head<2>();
        Eigen::Vector2d inward_normal(edge.y(), -edge.x());
        double c_val = inward_normal.dot(v1.head<2>());
        F.row(i + 2) << inward_normal.x(), inward_normal.y(), 0;
        C(i + 2) = c_val;
    }
    return {F, C};
}

// --- MonolithicPlanner Implementation ---
MonolithicPlanner::MonolithicPlanner() : env(), model(env) {
    model.set(GRB_StringAttr_ModelName, "MonolithicFootstepPlanner");
    initialize_parameters();
}

void MonolithicPlanner::initialize_parameters() {
    // --- Initialize ALIP matrices ---
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;

    T_ss_dt = T_ss / (K_knots - 1);
    A_d_autonomous_knot = (A_c_autonomous * T_ss_dt).exp();

    Eigen::Matrix<double, 4, 2> B_c_input_effect;
    B_c_input_effect << 0, 0, 0, 0, 1, 0, 0, 1;
    B_d_for_mpc = A_c_autonomous.inverse() * (A_d_autonomous_knot - Eigen::Matrix4d::Identity()) * B_c_input_effect;

    Ar_ds = (A_c_autonomous * T_ds).exp();
    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0, 0, 0, 0, mass*g, -mass*g, 0;
    Eigen::Matrix4d A_c_inv = A_c_autonomous.inverse();
    Eigen::Matrix<double, 4, 2> B_ds = Ar_ds * A_c_inv * ((1.0/T_ds) * A_c_inv * (Eigen::Matrix4d::Identity() - Ar_ds.inverse()) - Ar_ds.inverse()) * B_CoP_for_Bds;
    Eigen::Matrix<double, 4, 3> B_fp;
    B_fp << 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
    Eigen::Matrix<double, 4, 3> B_ds_padded = Eigen::Matrix<double, 4, 3>::Zero();
    B_ds_padded.block<4,2>(0,0) = B_ds;
    B_r = B_ds_padded + B_fp;

    // --- Initialize Polygons ---
    // double R1_xmax = 1, R1_xmin = 0, R1_ymax = 1, R1_ymin = 0;
    // double R2_xmax = 1.6, R2_xmin = 1.1, R2_ymax = 2, R2_ymin = 0;
    // double R3_xmax = 2, R3_xmin = 1.1, R3_ymax = 2.5, R3_ymin = 2.1;

    double R1_xmax = 0.54, R1_xmin = -0.1, R1_ymax = 0.8, R1_ymin = - 0.8, z1 = 0.0;
    double R2_xmax = 1.54, R2_xmin = 0.86, R2_ymax = 0.8, R2_ymin = - 0.8, z2 = 0.15;
    double R3_xmax = 2.24, R3_xmin = 1.86, R3_ymax = 0.8, R3_ymin = - 0.8, z3 = 0.25;
    double R4_xmax = 3.24, R4_xmin = 2.56, R4_ymax = 0.8, R4_ymin = - 0.8, z4 = 0.37;
    std::vector<std::vector<Eigen::Vector3d>> region_vertices = {
        {Eigen::Vector3d(R1_xmin, R1_ymin, z1), Eigen::Vector3d(R1_xmax, R1_ymin, z1),  Eigen::Vector3d(R1_xmax, R1_ymax,  z1), Eigen::Vector3d(R1_xmin, R1_ymax, z1)},
        {Eigen::Vector3d(R2_xmin, R2_ymin, z2), Eigen::Vector3d(R2_xmax, R2_ymin, z2),  Eigen::Vector3d(R2_xmax, R2_ymax, z2), Eigen::Vector3d(R2_xmin, R2_ymax, z2)},
        {Eigen::Vector3d(R3_xmin, R3_ymin, z3), Eigen::Vector3d(R3_xmax, R3_ymin, z3),  Eigen::Vector3d(R3_xmax, R3_ymax, z3), Eigen::Vector3d(R3_xmin, R3_ymax, z3)},
        {Eigen::Vector3d(R4_xmin, R4_ymin, z4), Eigen::Vector3d(R4_xmax, R4_ymin, z4),  Eigen::Vector3d(R4_xmax, R4_ymax, z4), Eigen::Vector3d(R4_xmin, R4_ymax, z4)},
        // Add more regions if N is larger
    };
    N_REG = region_vertices.size();
    for (const auto& vertices : region_vertices) {
        polygons.push_back({vertices});
    }

    // --- Initialize Stances and ALIP state ---
    initial_stance_footsteps.resize(4, 2);
    initial_stance_footsteps << 0.0, 0.0,
                                0.1, -0.1,
                                0.0, 0.0,
                                0.0, 0.0;
    // initial_alip_x << 0.0, 0.1, 0.0, 0.0;
    // initial_alip_x << 0, 0, 0.566648, 1.26405;
    initial_alip_x << 0, 0, 0, 0;
}

void MonolithicPlanner::create_gurobi_variables() 
{
    // x，y，z
    p_foot_vars.resize(N, std::vector<GRBVar>(3));
    // theta sin cos 近似值
    theta_vars.resize(N);
    sin_vars.resize(N); 
    cos_vars.resize(N);
    // 区域选择变量
    H_vars.resize(N, std::vector<GRBVar>(N_REG));
    // sin cos 近似分段线选择
    S_vars.resize(N, std::vector<GRBVar>(5)); 
    C_vars.resize(N, std::vector<GRBVar>(5));

    x_alip_vars.resize(N - 2, std::vector<std::vector<GRBVar>>(K_knots, std::vector<GRBVar>(4)));
    u_ankle_vars.resize(N - 2, std::vector<GRBVar>(K_knots - 1));
    v_ankle_vars.resize(N - 2, std::vector<GRBVar>(K_knots - 1));

    for (int c = 0; c < N; ++c) {
        p_foot_vars[c][0] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "px_"+std::to_string(c));
        p_foot_vars[c][1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "py_"+std::to_string(c));
        p_foot_vars[c][2] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "pz_"+std::to_string(c));
        theta_vars[c] = model.addVar(-M_PI, M_PI, 0, GRB_CONTINUOUS, "theta_"+std::to_string(c));
        sin_vars[c] = model.addVar(-1.0, 1.0, 0, GRB_CONTINUOUS, "sin_"+std::to_string(c));
        cos_vars[c] = model.addVar(-1.0, 1.0, 0, GRB_CONTINUOUS, "cos_"+std::to_string(c));
        for (int r = 0; r < N_REG; ++r) H_vars[c][r] = model.addVar(0, 1, 0, GRB_BINARY, "H_"+std::to_string(c)+"_"+std::to_string(r));
        for (int j = 0; j < 5; ++j) 
        {
            S_vars[c][j] = model.addVar(0, 1, 0, GRB_BINARY, "S_"+std::to_string(c)+"_"+std::to_string(j));
            C_vars[c][j] = model.addVar(0, 1, 0, GRB_BINARY, "C_"+std::to_string(c)+"_"+std::to_string(j));
        }
    }

    for (int n = 0; n < N - 2; ++n) {
        for (int k = 0; k < K_knots; ++k) {
            for (int d = 0; d < 4; ++d) x_alip_vars[n][k][d] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x_"+std::to_string(n)+"_"+std::to_string(k)+"_"+std::to_string(d));
            if (k < K_knots - 1) {
                u_ankle_vars[n][k] = model.addVar(-10, 10, 0, GRB_CONTINUOUS, "u_"+std::to_string(n)+"_"+std::to_string(k));
                v_ankle_vars[n][k] = model.addVar(-10, 10, 0, GRB_CONTINUOUS, "v_"+std::to_string(n)+"_"+std::to_string(k));
            }
        }
    }
    model.update();
}

void MonolithicPlanner::add_constraints() 
{
    // --- Initial Conditions ---
    // for(int j=0; j<3; ++j) 
    // {
    //     model.addConstr(p_foot_vars[0][j] == initial_stance_footsteps(j,0));
    // }
    // model.addConstr(theta_vars[0] == initial_stance_footsteps(3,0));
    // for(int j=0; j<3; ++j) 
    // {
    //     model.addConstr(p_foot_vars[1][j] == initial_stance_footsteps(j,1));
    // }

    // model.addConstr(theta_vars[1] == initial_stance_footsteps(3,1));
    // for(int i=0; i<4; ++i) 
    // {
    //     model.addConstr(x_alip_vars[0][0][i] == initial_alip_x(i));
    // }
    // 0        0 0.566648  1.26405
    // add constraint to initial alip x by 3rd step
    // Eigen::Vector2d p_current = initial_stance_footsteps.col(1).head<2>();
    // double next_foot_x = p_foot_vars[3][0];
    // double next_foot_y = p_foot_vars[3][1];
    // Eigen::Vector2d p_next(next_foot_x, next_foot_y);
    // Eigen::Vector2d r0 = initial_alip_x.head<2>();
    // 2. 计算 ω
    // double omega = std::sqrt(g / H_com);

    // // 3. 反解初始DCM (ξ₀)
    // Eigen::Vector2d xi_0 = (p_next - p_current) * std::exp(-omega * T_ss) + p_current;

    // // 4. 反解初始角动量 (L₀)
    // Eigen::Vector2d L0_xy = mass * omega * (xi_0 - r0);
    // initial_alip_x(2) = L0_xy(0); // 设置初始 Lx
    // initial_alip_x(3) = L0_xy(1); // 设置初始 Ly


// 这是根据初始状态和落脚点来反解初始角动量的约束，是一个理想的值。后续如果在后续中要遇到实时规划，直接将其替换成控制端给定的状态
    
#ifdef USE_MIQP
    // 1. 固定真正的初始条件：质心位置 (ALIP state 的 x 和 y)
    model.addConstr(x_alip_vars[0][0][0] == initial_alip_x(0), "init_x_pos");
    model.addConstr(x_alip_vars[0][0][1] == initial_alip_x(1), "init_y_pos");

    // 用于构建约束的常量
    double omega = std::sqrt(g / H_com);
    double exp_factor = std::exp(-omega * T_ss);
    double p_stance_x = initial_stance_footsteps(0, 1); // 作为参考点的初始支撑脚 x
    double p_stance_y = initial_stance_footsteps(1, 1); // 作为参考点的初始支撑脚 y
    double r0_x = initial_alip_x(0); // 初始 CoM x
    double r0_y = initial_alip_x(1); // 初始 CoM y

    // 获取相关的 Gurobi 决策变量
    GRBVar p_next_x = p_foot_vars[2][0];
    GRBVar p_next_y = p_foot_vars[2][1];
    GRBVar Lx_0 = x_alip_vars[0][0][2];
    GRBVar Ly_0 = x_alip_vars[0][0][3];

    // 构建初始DCM (xi_0) 的线性表达式
    // xi_0_x = (p_next_x - p_stance_x) * exp_factor + p_stance_x
    GRBLinExpr xi_0_x_expr = p_next_x * exp_factor - p_stance_x * exp_factor + p_stance_x;
    // xi_0_y = (p_next_y - p_stance_y) * exp_factor + p_stance_y
    GRBLinExpr xi_0_y_expr = p_next_y * exp_factor - p_stance_y * exp_factor + p_stance_y;

    // 添加关于初始角动量的约束
    // Lx_0 = mass * omega * (r0_y - xi_0_y)
    model.addConstr(Lx_0 == mass * omega * (r0_y - xi_0_y_expr), "init_Lx_dynamic_constr");

    // Ly_0 = -mass * omega * (r0_x - xi_0_x)
    model.addConstr(Ly_0 == -mass * omega * (r0_x - xi_0_x_expr), "init_Ly_dynamic_constr");
#endif

    // 关于初始落脚点 p_foot_vars[0] 和 p_foot_vars[1] 的约束保持不变
    for(int j=0; j<3; ++j) 
    {
        model.addConstr(p_foot_vars[0][j] == initial_stance_footsteps(j,0));
    }
    model.addConstr(theta_vars[0] == initial_stance_footsteps(3,0));
    for(int j=0; j<3; ++j) 
    {
        model.addConstr(p_foot_vars[1][j] == initial_stance_footsteps(j,1));
    }
    model.addConstr(theta_vars[1] == initial_stance_footsteps(3,1));

    // --- Region Constraints ---
    for (int c = 0; c < N; ++c) 
    {
        for (int r = 0; r < N_REG; ++r) 
        {
            auto [F, C_vec] = polygons.at(r).calculateCoefficients();
            for (int i = 0; i < F.rows(); i++) 
            {
                GRBLinExpr expr = F(i, 0)*p_foot_vars[c][0] + F(i, 1)*p_foot_vars[c][1] + F(i, 2)*p_foot_vars[c][2];
                model.addConstr(expr <= C_vec(i) + M_BIG * (1.0 - H_vars[c][r]));
            }
        }
        GRBLinExpr sum_H;
        for (int r = 0; r < N_REG; ++r) sum_H += H_vars[c][r];
        model.addConstr(sum_H == 1.0);
    }
    
    // --- Kinematic and Reachability Constraints (simplified) ---
    for (int c = 2; c < N; ++c) 
    {
        GRBQuadExpr reach_sq = (p_foot_vars[c][0] - p_foot_vars[c-1][0]) * (p_foot_vars[c][0] - p_foot_vars[c-1][0]) +
                               (p_foot_vars[c][1] - p_foot_vars[c-1][1]) * (p_foot_vars[c][1] - p_foot_vars[c-1][1]);
        model.addQConstr(reach_sq <= dis_th * dis_th);
    }

    // details
    // Reachability constraint
    for (int c = 2; c < N; ++c) 
    {
        GRBVar xn = p_foot_vars[c][0];
        GRBVar yn = p_foot_vars[c][1];
        GRBVar zn = p_foot_vars[c][2];
        GRBVar xc = p_foot_vars[c-1][0];
        GRBVar yc = p_foot_vars[c-1][1];
        GRBVar zc = p_foot_vars[c-1][2];
        // GRBVar thetac = footsteps[c-1][2]; // Not used directly, but sin/cos are
        GRBVar sinc_prev = sin_vars[c-1];
        GRBVar cosc_prev = cos_vars[c-1];
        // 实际上是和左右脚有关系的,to do list, add flag to vars
        if (c % 2 != 0) 
        { // Odd step (e.g., f3, f5, ...), right leg
            std::array<double, 2> p1 = {0, deta1};
            std::array<double, 2> p2 = {0, -deta2};
            double d1 = dis_th, d2 = dis_th;

            GRBLinExpr term1_a_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
            GRBLinExpr term2_a_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
            model.addQConstr(term1_a_expr * term1_a_expr + term2_a_expr * term2_a_expr <= d1*d1, "reach_a_c" + std::to_string(c));
            
            GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev);
            GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev);
            model.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));

        } 
        else 
        { // Even step, left leg
            std::array<double, 2> p1 = {0, -deta1};
            std::array<double, 2> p2 = {0, deta2};
            double d1 = dis_th, d2 = dis_th;

            GRBLinExpr term1_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
            GRBLinExpr term2_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
            model.addQConstr(term1_expr * term1_expr + term2_expr * term2_expr <= d1*d1, "reach_a_c" + std::to_string(c));

            GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev); // Python had this as term1 again
            GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev); // Python had this as term2 again
            model.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));
        }

        GRBLinExpr z_diff = zn - zc;
        model.addConstr(z_diff <= 0.18, "z_diff_upper_c" + std::to_string(c));
        model.addConstr(z_diff >= -0.18, "z_diff_lower_c" + std::to_string(c));
    }
    
    // Sin approximation constraints
    std::vector<std::tuple<double, double, double, double>> sin_params = {
        {-M_PI, 1 - M_PI, -1, -M_PI},
        {1 - M_PI, -1, 0, -1},
        {-1, 1, 1, 0},
        {1, M_PI - 1, 0, 1},
        {M_PI - 1, M_PI, -1, M_PI}
    };
    for (int c = 0; c < N; ++c) 
    {
        GRBVar theta_c = theta_vars[c];
        GRBVar sin_theta_c = sin_vars[c];
        for (int i = 0; i < 5; ++i) 
        {
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
    for (int c = 0; c < N; ++c) 
    {
        // 3 is the index for theta, 4 for sin(theta), 5 for cos(theta)
        GRBVar theta_c = theta_vars[c];
        GRBVar cos_theta_c = cos_vars[c];
        for (int i = 0; i < 5; ++i) 
        {
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
    
    // double del_theta_max = M_PI / 8.0;
    // for (int c = 1; c < N; ++c) 
    // {
    //     GRBLinExpr delta_theta = theta_vars[c] - theta_vars[c-1];
    //     model.addConstr(delta_theta <= del_theta_max, "max_rot_pos_c" + std::to_string(c));
    //     model.addConstr(delta_theta >= -del_theta_max, "max_rot_neg_c" + std::to_string(c));
    // }

    // 两脚的角度约束
    for (int c = 2; c < N; ++c) 
    {
        GRBLinExpr delta_theta = theta_vars[c] - theta_vars[c-1];
        if (c % 2 == 0) // 左脚
        {
            model.addConstr(delta_theta <= M_PI / 15.0, "max_rot_pos_left_c" + std::to_string(c));
            model.addConstr(delta_theta >= -M_PI / 10.0, "max_rot_neg_left_c" + std::to_string(c));
        } 
        else // 右脚
        {
            model.addConstr(delta_theta <= M_PI / 10.0, "max_rot_pos_right_c" + std::to_string(c));
            model.addConstr(delta_theta >= -M_PI / 15.0, "max_rot_neg_right_c" + std::to_string(c));
        }
    }
#ifdef USE_MIQP
    // --- ALIP Dynamics ---
    for (int n = 0; n < N - 2; ++n) 
    {
        for (int k = 0; k < K_knots - 1; ++k) 
        {
            for (int row = 0; row < 4; ++row) 
            {
                model.addConstr(x_alip_vars[n][k+1][row] == gurobi_quicksum(A_d_autonomous_knot, row, x_alip_vars[n][k]) + B_d_for_mpc(row,0) * v_ankle_vars[n][k] + B_d_for_mpc(row, 1) * u_ankle_vars[n][k]);
            }
        }
    }

    // --- ALIP Reset Map ---
    for (int n = 0; n < N - 3; ++n) 
    {
        GRBLinExpr dp_x = p_foot_vars[n+2][0] - p_foot_vars[n+1][0];
        GRBLinExpr dp_y = p_foot_vars[n+2][1] - p_foot_vars[n+1][1];
        GRBLinExpr dp_z = p_foot_vars[n+2][2] - p_foot_vars[n+1][2];
        for (int row = 0; row < 4; ++row) 
        {
            GRBLinExpr br_term = B_r(row,0)*dp_x + B_r(row,1)*dp_y + B_r(row,2)*dp_z;
            model.addConstr(x_alip_vars[n+1][0][row] == gurobi_quicksum(Ar_ds, row, x_alip_vars[n][K_knots-1]) + br_term);
        }
    }

#endif
}

void MonolithicPlanner::set_objective() 
{
    GRBQuadExpr objective = 0;
    
    // 1. Terminal cost
    std::array<double, 3> g_target = {3.0, 0.6, 0.05};
    // GRBLinExpr e0 = p_foot_vars[N-1][0] - g_target[0];
    // GRBLinExpr e1 = p_foot_vars[N-1][1] - g_target[1] ;
    // objective += 200 * (e0*e0 + e1*e1);
    
    double target_theta = M_PI / 4.0;
    GRBLinExpr e3 = theta_vars[N - 1] - target_theta;
    GRBLinExpr e4 = theta_vars[N - 2] - target_theta;

    



    // 根据角度和原点值来计算终点的左右脚位置
    // Calculate left and right foot positions based on target_theta and g_target
    double foot_offset = 0.1; // Half of the distance between left and right foot (0.2 / 2)
    GRBLinExpr left_foot_x = g_target[0] - foot_offset * std::sin(target_theta);
    GRBLinExpr left_foot_y = g_target[1] + foot_offset * std::cos(target_theta);
    GRBLinExpr right_foot_x = g_target[0] + foot_offset * std::sin(target_theta);
    GRBLinExpr right_foot_y = g_target[1] - foot_offset * std::cos(target_theta);

    // Add terminal cost for left and right foot positions
    GRBLinExpr e_left_x = p_foot_vars[N-1][0] - left_foot_x;
    GRBLinExpr e_left_y = p_foot_vars[N-1][1] - left_foot_y;
    GRBLinExpr e_right_x = p_foot_vars[N-2][0] - right_foot_x;
    GRBLinExpr e_right_y = p_foot_vars[N-2][1] - right_foot_y;

    objective += 100 * (e_left_x * e_left_x + e_left_y * e_left_y);
    objective += 100 * (e_right_x * e_right_x + e_right_y * e_right_y);
    


    objective += 100 * (e3*e3);
    objective += 100 * (e4*e4);

    // 2. Step displacement cost
    for (int c = 1; c < N; ++c) {
        GRBLinExpr dx = p_foot_vars[c][0] - p_foot_vars[c-1][0];
        GRBLinExpr dy = p_foot_vars[c][1] - p_foot_vars[c-1][1];
        objective += 0.5 * (dx*dx + dy*dy);
    }
#ifdef USE_MIQP
    // 3. Ankle torque cost
    for (int n = 0; n < N - 2; ++n) {
        for (int k = 0; k < K_knots-1; ++k) {
            objective += 0.1 * (u_ankle_vars[n][k]*u_ankle_vars[n][k] + v_ankle_vars[n][k]*v_ankle_vars[n][k]);
        }
    }
#endif
    model.setObjective(objective, GRB_MINIMIZE);
}

void MonolithicPlanner::run() {
    std::cout << "Building monolithic model..." << std::endl;
    create_gurobi_variables();
    add_constraints();
    set_objective();

    std::cout << "Solving..." << std::endl;
    model.set(GRB_IntParam_NonConvex, 2);
    model.set(GRB_DoubleParam_TimeLimit, 60.0); // 1 minute time limit
    model.set(GRB_DoubleParam_MIPGap, 0.1); // 10% gap is acceptable

    model.optimize();
    std::ofstream outFile("footsteps.data");
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }
    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL || status == GRB_TIME_LIMIT) {
        std::cout << "--- Solution Found ---" << std::endl;
        std::cout << "Objective value: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
        for (int c = 0; c < N; ++c) {

            std::ofstream envFile("env.data");
            if (!envFile.is_open()) {
                std::cerr << "Error opening environment file." << std::endl;
                return;
            }
            for (size_t r = 0; r < polygons.size(); ++r) {
                // envFile << "Region " << r << " vertices:" << std::endl;
                for (const auto& vertex : polygons[r].vertices) {
                    envFile << std::fixed << std::setprecision(3) << vertex.x() << " " << vertex.y() << " " << vertex.z() << std::endl;
                }
            }
            envFile.close();

            outFile <<std::fixed << std::setprecision(3) << p_foot_vars[c][0].get(GRB_DoubleAttr_X) << " " << p_foot_vars[c][1].get(GRB_DoubleAttr_X) << " " << p_foot_vars[c][2].get(GRB_DoubleAttr_X) << " "<< theta_vars[c].get(GRB_DoubleAttr_X) << std::endl;
            
            std::cout << "Footstep " << c << ": "
                      << "x=" << std::fixed << std::setprecision(3) << p_foot_vars[c][0].get(GRB_DoubleAttr_X)
                      << ", y=" << p_foot_vars[c][1].get(GRB_DoubleAttr_X)
                      << ", z=" << p_foot_vars[c][2].get(GRB_DoubleAttr_X)
                      << ", theta = "<<theta_vars[c].get(GRB_DoubleAttr_X)
                      << std::endl;
            for(int r=0; r < N_REG; ++r) {
                if(H_vars[c][r].get(GRB_DoubleAttr_X) > 0.5) {
                    std::cout << "  -> in Region " << r << std::endl;
                }
            }
        }
    } else if (status == GRB_INFEASIBLE) {
        std::cerr << "Model is INFEASIBLE. Computing IIS..." << std::endl;
        model.computeIIS();
        model.write("monolithic_infeasible.ilp");
    } else {
        std::cerr << "Optimization was stopped with status " << status << std::endl;
    }
}

GRBLinExpr MonolithicPlanner::gurobi_quicksum(const Eigen::MatrixXd& M, int row, const std::vector<GRBVar>& vars) {
    GRBLinExpr expr = 0;
    for (int col = 0; col < M.cols(); ++col) {
        expr += M(row, col) * vars[col];
    }
    return expr;
}