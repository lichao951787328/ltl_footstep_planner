#include <doublelayer_planning.hpp>
#include <iomanip>
#include <random>
// #include <matplotlibcpp.h>
using namespace std;
// namespace plt = matplotlibcpp;
doublelayer_planning::doublelayer_planning()
{
    model_upper.set(GRB_StringAttr_ModelName, "double_layer_upper");
    model_lower.set(GRB_StringAttr_ModelName, "double_layer_lower");

}

GRBLinExpr gurobi_quicksum(const std::vector<GRBVar>& vars, const Eigen::VectorXd& coeffs) 
{
    GRBLinExpr expr = 0;
    if (vars.size() != coeffs.size()) {
        throw std::runtime_error("gurobi_quicksum: vars and coeffs size mismatch");
    }
    for (size_t i = 0; i < vars.size(); ++i) {
        expr += vars[i] * coeffs(i);
    }
    return expr;
}

GRBLinExpr gurobi_quicksum(const Eigen::MatrixXd& M, int row, const std::vector<GRBVar>& vars) 
{
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

void doublelayer_planning::get_autonomous_alip_matrix_A()
{
    A_c_autonomous << 0, 0, 0, 1 / (mass * H_com),
                      0, 0, -1 / (mass * H_com), 0,
                      0, -mass * g, 0, 0,
                      mass * g, 0, 0, 0;
    cout<< "A_c_autonomous: " << endl << A_c_autonomous << endl;
}

void doublelayer_planning::get_alip_matrices_with_input()
{
    Eigen::Matrix<double, 4, 2> B_c_input_effect;
    B_c_input_effect.leftCols(1) = Eigen::Vector4d(0, 0, 1, 0); // roll 
    B_c_input_effect.rightCols(1) = Eigen::Vector4d(0, 0, 0, 1); // pitch

    A_d_autonomous_knot = (A_c_autonomous * T_ss_dt).exp();
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
        B_d_for_mpc = A_c_autonomous.colPivHouseholderQr().solve((A_d_autonomous_knot - Eigen::Matrix4d::Identity()) * B_c_input_effect);

    } 
    else 
    {
      B_d_for_mpc = A_c_autonomous.inverse() * (A_d_autonomous_knot - Eigen::Matrix4d::Identity()) * B_c_input_effect;
    }
    cout << "A_d_autonomous_knot: " << endl << A_d_autonomous_knot << endl;
    cout << "B_d_for_mpc: " << endl << B_d_for_mpc << endl;
}

void doublelayer_planning::get_alip_matrices()
{
    A_s2s_autonomous_cycle = (A_c_autonomous * (T_ss + T_ds)).exp();
    cout<< "A_s2s_autonomous_cycle: " << endl << A_s2s_autonomous_cycle << endl;
}

void doublelayer_planning::get_alip_reset_map_matrices_detailed()
{
    Ar_ds = (A_c_autonomous * T_ds).exp();

    Eigen::Matrix<double, 4, 2> B_CoP_for_Bds;
    B_CoP_for_Bds << 0, 0,
                     0, 0,
                     0, mass * g,
                     -mass * g, 0;

    Eigen::Matrix<double, 4, 2> B_ds = Eigen::Matrix<double, 4, 2>::Zero();
    if (std::abs(A_c_autonomous.determinant()) < 1e-9 || std::abs(Ar_ds.determinant()) < 1e-9) 
    {
        std::cerr << "Warning: A_c or Ar_ds is singular in get_alip_reset_map_matrices_detailed. B_ds set to zero." << std::endl;
    } 
    else
    {
        Eigen::Matrix4d A_c_inv = A_c_autonomous.inverse();
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

    B_r = B_ds_padded + B_fp;
    cout<< "Ar_ds: " << endl << Ar_ds << endl;
    cout<< "B_r: " << endl << B_r << endl;
}

std::pair<Eigen::Vector4d, Eigen::Vector4d> doublelayer_planning::calculate_periodic_alip_reference_states(double vx_d, double vy_d, double stance_width_l, double T_s2s, const Eigen::Matrix4d& A_s2s_autonomous_cycle, const Eigen::Matrix<double, 4, 2>& Br_map_for_cycle_2d, bool initial_stance_is_left)
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
    // cout<<"Br_map_for_cycle_2d: "<<Br_map_for_cycle_2d<<endl;
    // cout<<"rhs_vector: "<<rhs_vector.transpose()<<endl;
    Eigen::Vector4d x_start_of_2_step_cycle_ref;

    // Check determinant before attempting LU decomposition or use a more robust solver
    if (std::abs(lhs_matrix.determinant()) < 1e-9) { // Check for singularity
        std::cerr << "FATAL WARNING: lhs_matrix is singular! Returning a zero reference state. This will likely lead to a zero objective function in the lower layer." << std::endl; 
        // std::cout<<"-----------------------------------------"<<endl;
        x_start_of_2_step_cycle_ref.setZero();

    } else {
        // For well-conditioned, invertible square matrices, LU decomposition is common.
        x_start_of_2_step_cycle_ref = lhs_matrix.lu().solve(rhs_vector);
    }

    Eigen::Vector4d ref_state_phase0 = x_start_of_2_step_cycle_ref;
    Eigen::Vector4d ref_state_phase1 = M_cycle * ref_state_phase0 + Br_map_for_cycle_2d * delta_p_A_2d;

    return {ref_state_phase0, ref_state_phase1};
}


void doublelayer_planning::initial_env()
{
    // 尝试把这个环境内的多边形表示成论文中的形式
    // 多边形表达成法向量和顶点的形式，且顶点默认为逆时针
    // vector<polygon3D> polygons;
    double R1_xmax = 1, R1_xmin = 0, R1_ymax = 1, R1_ymin = 0;
    double R2_xmax = 1.6, R2_xmin = 1.1, R2_ymax = 2, R2_ymin = 0;
    double R3_xmax = 2, R3_xmin = 1.1, R3_ymax = 2.5, R3_ymin = 2.1;
    double R4_xmax = 1, R4_xmin = -0.5, R4_ymax = 2.7, R4_ymin = 2.1;
    double R5_xmax = 2, R5_xmin = 1.5, R5_ymax = 3, R5_ymin = 2.55;
    // Define vertices for each region in counterclockwise order
    std::vector<std::vector<Eigen::Vector3d>> region_vertices = {
        {Eigen::Vector3d(R1_xmin, R1_ymin, 0), Eigen::Vector3d(R1_xmax, R1_ymin, 0),  Eigen::Vector3d(R1_xmax, R1_ymax,  0), Eigen::Vector3d(R1_xmin, R1_ymax, 0)},
        {Eigen::Vector3d(R2_xmin, R2_ymin, 0.05), Eigen::Vector3d(R2_xmax, R2_ymin, 0.05),  Eigen::Vector3d(R2_xmax, R2_ymax, 0.05), Eigen::Vector3d(R2_xmin, R2_ymax, 0.05)},
        {Eigen::Vector3d(R3_xmin, R3_ymin, 0.1), Eigen::Vector3d(R3_xmax, R3_ymin, 0.1),  Eigen::Vector3d(R3_xmax, R3_ymax, 0.1), Eigen::Vector3d(R3_xmin, R3_ymax, 0.1)},
        {Eigen::Vector3d(R4_xmin, R4_ymin, 0.1), Eigen::Vector3d(R4_xmax, R4_ymin, 0.1),  Eigen::Vector3d(R4_xmax, R4_ymax, 0.1), Eigen::Vector3d(R4_xmin, R4_ymax, 0.1)},
        {Eigen::Vector3d(R5_xmin, R5_ymin, 0.1), Eigen::Vector3d(R5_xmax, R5_ymin, 0.1),  Eigen::Vector3d(R5_xmax, R5_ymax, 0.1), Eigen::Vector3d(R5_xmin, R5_ymax, 0.1)}
    };

    // Convert vertices into polygon3D objects
    for (const auto& vertices : region_vertices) {
        polygon3D poly;
        for (const auto& vertex : vertices) {
            poly.vertices.push_back(vertex);
        }
        poly.normal = Eigen::Vector3d(0, 0, 1); // Assuming all polygons are in the XY plane
        polygons.push_back(poly);
    }
    N_REG = polygons.size(); // Update the number of regions based on the polygons defined



    // A_regions.resize(N_REG);
    // b_regions.resize(N_REG);

    // std::array<std::array<double, 3>, 6> A_template = {{
    //     {{1, 0, 0}}, {{-1, 0, 0}}, {{0, 1, 0}}, {{0, -1, 0}}, {{0, 0, 1}}, {{0, 0, -1}}
    // }};

    // double R1_xmax = 1, R1_xmin = 0, R1_ymax = 1, R1_ymin = 0;
    // double R2_xmax = 1.6, R2_xmin = 1.1, R2_ymax = 2, R2_ymin = 0;
    // double R3_xmax = 2, R3_xmin = 1.1, R3_ymax = 2.5, R3_ymin = 2.1;
    // double R4_xmax = 1, R4_xmin = -0.5, R4_ymax = 2.7, R4_ymin = 2.1;
    // double R5_xmax = 2, R5_xmin = 1.5, R5_ymax = 3, R5_ymin = 2.55;

    // std::array<double, 2> R1_midpt = {(R1_xmax + R1_xmin) / 2, (R1_ymax + R1_ymin) / 2};
    // std::array<double, 2> R2_midpt = {(R2_xmax + R2_xmin) / 2, (R2_ymax + R2_ymin) / 2};
    // std::array<double, 2> R3_midpt = {(R3_xmax + R3_xmin) / 2, (R3_ymax + R3_ymin) / 2};
    // std::array<double, 2> R4_midpt = {(R4_xmax + R4_xmin) / 2, (R4_ymax + R4_ymin) / 2};
    // std::array<double, 2> R5_midpt = {(R5_xmax + R5_xmin) / 2, (R5_ymax + R5_ymin) / 2};

    // A_regions[0] = A_template; b_regions[0] = {R1_xmax, -R1_xmin, R1_ymax, -R1_ymin, M_PI, M_PI/2.0}; // Note: Python was math.pi, math.pi/2 for b[4],b[5]
    // A_regions[1] = A_template; b_regions[1] = {R2_xmax, -R2_xmin, R2_ymax, -R2_ymin, M_PI, M_PI/2.0}; // Assuming last two b are upper bounds
    // A_regions[2] = A_template; b_regions[2] = {R3_xmax, -R3_xmin, R3_ymax, -R3_ymin, M_PI, M_PI/2.0}; // If they were ranges, one would be -theta_lower
    // A_regions[3] = A_template; b_regions[3] = {R4_xmax, -R4_xmin, R4_ymax, -R4_ymin, M_PI, M_PI/2.0};
    // A_regions[4] = A_template; b_regions[4] = {R5_xmax, -R5_xmin, R5_ymax, -R5_ymin, M_PI, M_PI/2.0};

    // Eigen::Matrix<double, 6, 3> F_region_common;
    // F_region_common << 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1;
    // Eigen::Matrix<double, 6, 1> c_region0_vec, c_region1_vec, c_region2_vec, c_region3_vec, c_region4_vec;
    // c_region0_vec<<1, 0, 1, 0, 0, 0;
    // c_region1_vec<<1.6, -1.1, 2, 0, 0, 0;
    // c_region2_vec<<2, -1.1, 2.5, -2.1, 0, 0;
    // c_region3_vec<<1, 0.5, 2.7, -2.1, 0, 0;
    // c_region4_vec<<2, -1.5, 3, -2.55, 0, 0;
    // regions_F_list = {
    //     F_region_common, F_region_common, F_region_common, F_region_common, F_region_common
    // };
    // regions_c_list = {
    //     c_region0_vec, c_region1_vec, c_region2_vec, c_region3_vec, c_region4_vec
    // };

    // 在实际应用中，需要根据x，y来解算z
    T_ss_dt = T_ss / (K_knots - 1);

    double init_theta = 0.0;
    double f1_s = std::sin(init_theta);
    double f1_c = std::cos(init_theta);
    // 第一个步态是左脚，第二个步态是右脚
    initial_stance_footsteps(0, 0) = 0.0; // x
    initial_stance_footsteps(1, 0) = nominal_stance_width; // y
    initial_stance_footsteps(2, 0) = 0; // 
    initial_stance_footsteps(3, 0) = init_theta;
    initial_stance_footsteps(4, 0) = f1_s; // sin(theta)
    initial_stance_footsteps(5, 0) = f1_c; // cos(theta)
    initial_stance_footsteps(0, 1) = 0.0; // x
    initial_stance_footsteps(1, 1) = 0.0; // y
    initial_stance_footsteps(2, 1) = 0.0; // z
    initial_stance_footsteps(3, 1) = init_theta;
    initial_stance_footsteps(4, 1) = f1_s; // sin(theta)
    initial_stance_footsteps(5, 1) = f1_c; // cos(theta)

    // initial_alip_x.x() = (initial_stance_footsteps(0, 0) + initial_stance_footsteps(0, 1)) / 2.0; // x
    // initial_alip_x.y() = (initial_stance_footsteps(1, 0) + initial_stance_footsteps(1, 1)) / 2.0; // y
    
    // initial_alip_x.x() = 0.05; // x
    // initial_alip_x.y() = 0.16; // y
    // // 初始时刻没有力矩
    // initial_alip_x.z() = 11.5;
    // initial_alip_x.w() = -3.5; 

    // x_current_alip_val_eigen: 0.0367794  0.138334   11.3165  -3.22315

    get_autonomous_alip_matrix_A();
    get_alip_matrices_with_input();
    get_alip_matrices();
    get_alip_reset_map_matrices_detailed();


}

void doublelayer_planning::add_upper_layer_variables()
{
    cout<<"Add upper layer variables here"<<endl;
    // Add upper layer variables here
    // Example: footsteps = std::vector<std::vector<GRBVar>>(N, std::vector<GRBVar>(5));
    footsteps.resize(N, std::vector<GRBVar>(6)); // x y z theta sin cos
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < 6; ++j) 
        {
            footsteps[i][j] = model_upper.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "F" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // Trig approx binary variables
    S_vars.resize(N, std::vector<GRBVar>(5)); // For sin
    C_vars.resize(N, std::vector<GRBVar>(5)); // For cos    
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < 5; ++j) 
        {
            S_vars[i][j] = model_upper.addVar(0.0, 1.0, 0.0, GRB_BINARY, "S" + std::to_string(i) + "_" + std::to_string(j));
            C_vars[i][j] = model_upper.addVar(0.0, 1.0, 0.0, GRB_BINARY, "C" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    // Safe regions binary variables
    H_vars.resize(N, std::vector<GRBVar>(N_REG));
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < N_REG; ++j) 
        {
            H_vars[i][j] = model_upper.addVar(0.0, 1.0, 0.0, GRB_BINARY, "H" + std::to_string(i) + "_" + std::to_string(j));
        }
    }
    model_upper.update(); // Important after adding vars
}

void doublelayer_planning::add_upper_layer_constraints()
{
    cout<<"Add upper layer constraints here"<<endl;
    // double theta_upper_bound = M_PI;
    // double theta_lower_bound = -M_PI/2.0; 
    // for(int r=0; r<N_REG; ++r) 
    // {
    //     b_regions[r][4] = theta_upper_bound;
    //     b_regions[r][5] = -theta_lower_bound; // for -theta <= -theta_min
    // }
    // auto {F, C} = polygons.calculateCoefficients();
    // All footsteps must be in one of the regions
    for (int c = 0; c < N; ++c) 
    {
        for (int r = 0; r < N_REG; ++r) 
        {
            std::pair<Eigen::MatrixX3d, Eigen::VectorXd> FC = polygons.at(r).calculateCoefficients();
            Eigen::MatrixX3d F = FC.first; // Coefficients for inequalities
            Eigen::VectorXd c_vec = FC.second; // Constants for inequalities
            for (int i = 0; i < F.rows(); i++)
            {
                GRBLinExpr expr = 0;
                for (int j = 0; j < 3; ++j) 
                { // x, y, theta
                    expr += F(i, j) * footsteps[c][j];
                }
                model_upper.addConstr(expr <= c_vec(i) + M_BIG * (1.0 - H_vars[c][r]), "region_constr_c" + std::to_string(c) + "_r" + std::to_string(r) + "_i" + std::to_string(i));
            }
            // model_upper.addConstr(footsteps[c][2] <= theta_upper_bound + M_BIG * (1.0 - H_vars[c][r]), "theta_upper_c" + std::to_string(c) + "_r" + std::to_string(r));
            // for (int i = 0; i < 6; ++i) 
            // { // 6 inequalities per region
            //     GRBLinExpr expr = 0;
            //     for (int j = 0; j < 3; ++j) 
            //     { // x, y, theta
            //         expr += A_regions[r][i][j] * footsteps[c][j];
            //     }
            //     model_upper.addConstr(expr - b_regions[r][i] <= M_BIG * (1.0 - H_vars[c][r]), "region_constr_c" + std::to_string(c) + "_r" + std::to_string(r) + "_i" + std::to_string(i));
            // }

        }
        // Sum of H_vars[c][j] for j must be 1
        GRBLinExpr sum_H = 0;
        for (int j = 0; j < N_REG; ++j) 
        {
            sum_H += H_vars[c][j];
        }
        model_upper.addConstr(sum_H == 1.0, "sum_H_c" + std::to_string(c));
    }

    cout<<"Upper layer constraints added."<<endl;

    // Reachability constraint
    for (int c = 2; c < N; ++c) 
    {
        GRBVar xn = footsteps[c][0];
        GRBVar yn = footsteps[c][1];
        GRBVar zn = footsteps[c][2];
        GRBVar xc = footsteps[c-1][0];
        GRBVar yc = footsteps[c-1][1];
        GRBVar zc = footsteps[c-1][2];
        // GRBVar thetac = footsteps[c-1][2]; // Not used directly, but sin/cos are
        GRBVar sinc_prev = footsteps[c-1][4];
        GRBVar cosc_prev = footsteps[c-1][5];
        // 实际上是和左右脚有关系的
        if (c % 2 != 0) 
        { // Odd step (e.g., f3, f5, ...), right leg
            std::array<double, 2> p1 = {0, deta1};
            std::array<double, 2> p2 = {0, -deta2};
            double d1 = dis_th, d2 = dis_th;

            GRBLinExpr term1_a_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
            GRBLinExpr term2_a_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
            model_upper.addQConstr(term1_a_expr * term1_a_expr + term2_a_expr * term2_a_expr <= d1*d1, "reach_a_c" + std::to_string(c));
            
            GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev);
            GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev);
            model_upper.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));

        } 
        else 
        { // Even step, left leg
            std::array<double, 2> p1 = {0, -deta1};
            std::array<double, 2> p2 = {0, deta2};
            double d1 = dis_th, d2 = dis_th;

            GRBLinExpr term1_expr = xn - (xc + p1[0]*cosc_prev - p1[1]*sinc_prev);
            GRBLinExpr term2_expr = yn - (yc + p1[0]*sinc_prev + p1[1]*cosc_prev);
            model_upper.addQConstr(term1_expr * term1_expr + term2_expr * term2_expr <= d1*d1, "reach_a_c" + std::to_string(c));

            GRBLinExpr term1_b_expr = xn - (xc + p2[0]*cosc_prev - p2[1]*sinc_prev); // Python had this as term1 again
            GRBLinExpr term2_b_expr = yn - (yc + p2[0]*sinc_prev + p2[1]*cosc_prev); // Python had this as term2 again
            model_upper.addQConstr(term1_b_expr * term1_b_expr + term2_b_expr * term2_b_expr <= d2*d2, "reach_b_c" + std::to_string(c));
        }

        GRBLinExpr z_diff = zn - zc;
        model_upper.addConstr(z_diff <= 0.15, "z_diff_upper_c" + std::to_string(c));
        model_upper.addConstr(z_diff >= -0.15, "z_diff_lower_c" + std::to_string(c));
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
        // 3 is the index for theta, 4 for sin(theta), 5 for cos(theta)
        GRBVar theta_c = footsteps[c][3];
        GRBVar sin_theta_c = footsteps[c][4];
        for (int i = 0; i < 5; ++i) 
        {
            double phi_l = std::get<0>(sin_params[i]);
            double phi_lp1 = std::get<1>(sin_params[i]);
            double g_l = std::get<2>(sin_params[i]);
            double h_l = std::get<3>(sin_params[i]);
            GRBVar S_ci = S_vars[c][i];

            // theta_c in [phi_l, phi_lp1] if S_ci = 1
            model_upper.addConstr(theta_c <= phi_lp1 + M_BIG * (1.0 - S_ci)); // Original: -M*(1-S) - phi_lp1 + theta <= 0 => theta - phi_lp1 <= M(1-S)
            model_upper.addConstr(theta_c >= phi_l - M_BIG * (1.0 - S_ci));   // Original: -M*(1-S) + phi_l - theta <= 0 => phi_l - theta <= M(1-S) => theta - phi_l >= -M(1-S)

            // sin_theta_c = g_l * theta_c + h_l if S_ci = 1
            model_upper.addConstr(sin_theta_c <= g_l * theta_c + h_l + M_BIG * (1.0 - S_ci));
            model_upper.addConstr(sin_theta_c >= g_l * theta_c + h_l - M_BIG * (1.0 - S_ci));
        }
        GRBLinExpr sum_S = 0;
        for (int j = 0; j < 5; ++j) sum_S += S_vars[c][j];
        model_upper.addConstr(sum_S == 1.0, "sum_S_c" + std::to_string(c));
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
        GRBVar theta_c = footsteps[c][3];
        GRBVar cos_theta_c = footsteps[c][5];
        for (int i = 0; i < 5; ++i) 
        {
            double phi_l = std::get<0>(cos_params[i]);
            double phi_lp1 = std::get<1>(cos_params[i]);
            double g_l = std::get<2>(cos_params[i]);
            double h_l = std::get<3>(cos_params[i]);
            GRBVar C_ci = C_vars[c][i];

            model_upper.addConstr(theta_c <= phi_lp1 + M_BIG * (1.0 - C_ci));
            model_upper.addConstr(theta_c >= phi_l - M_BIG * (1.0 - C_ci));
            model_upper.addConstr(cos_theta_c <= g_l * theta_c + h_l + M_BIG * (1.0 - C_ci));
            model_upper.addConstr(cos_theta_c >= g_l * theta_c + h_l - M_BIG * (1.0 - C_ci));
        }
        GRBLinExpr sum_C = 0;
        for (int j = 0; j < 5; ++j) sum_C += C_vars[c][j];
        model_upper.addConstr(sum_C == 1.0, "sum_C_c" + std::to_string(c));
    }
    
    // Initial footstep constraints
    double init_theta = 0.0;
    double f1_s = std::sin(init_theta);
    double f1_c = std::cos(init_theta);

    model_upper.addConstr(footsteps[0][0] == initial_stance_footsteps(0, 0), "f0_x");
    model_upper.addConstr(footsteps[0][1] == initial_stance_footsteps(1, 0), "f0_y");
    model_upper.addConstr(footsteps[0][2] == initial_stance_footsteps(2, 0), "f0_z");
    
    // footsteps[0][2] (theta) is not explicitly set here, but its sin/cos are.
    // This implies theta is such that sin(theta) = f1_s and cos(theta) = f1_c
    // We need to pick the S and C vars that correspond to theta=0
    // For sin: theta=0 is in segment 2 (-1 to 1), where S[c][2] = 1. g_l=1, h_l=0. sin(0) = 1*0+0=0
    // For cos: theta=0 is in segment 2 (1-pi/2 to pi/2-1), where C[c][2] = 1. g_l=0, h_l=1. cos(0) = 0*0+1=1

    model_upper.addConstr(footsteps[0][4] == initial_stance_footsteps(4, 0), "f0_s");
    model_upper.addConstr(footsteps[0][5] == initial_stance_footsteps(5, 0), "f0_c");
    model_upper.addConstr(S_vars[0][2] == 1.0, "S0_2"); // sin(0) is in segment where g=1,h=0 (-1 to 1)
    model_upper.addConstr(C_vars[0][2] == 1.0, "C0_2"); // cos(0) is in segment where g=0,h=1 (1-pi/2 to pi/2-1, approx -0.57 to 0.57)

    model_upper.addConstr(footsteps[1][0] == initial_stance_footsteps(0, 1), "f1_x");
    model_upper.addConstr(footsteps[1][1] == initial_stance_footsteps(1, 1), "f1_y");
    model_upper.addConstr(footsteps[1][2] == initial_stance_footsteps(2, 1), "f1_z");
    model_upper.addConstr(footsteps[1][3] == initial_stance_footsteps(3, 1), "f1_theta"); // Explicitly set theta here
    model_upper.addConstr(footsteps[1][4] == initial_stance_footsteps(4, 1), "f1_s");
    model_upper.addConstr(footsteps[1][5] == initial_stance_footsteps(5, 1), "f1_c");
    model_upper.addConstr(S_vars[1][2] == 1.0, "S1_2");
    model_upper.addConstr(C_vars[1][2] == 1.0, "C1_2");

    // Small displacement thresholds
    double small_displacement_threshold = 0.15;
    // double T_step = 1.0/; // Time step duration (not T_cycle for velocity smoothness)

    GRBLinExpr dx_init = footsteps[2][0] - footsteps[0][0];
    GRBLinExpr dy_init = footsteps[2][1] - footsteps[0][1];
    model_upper.addQConstr(dx_init*dx_init + dy_init*dy_init <= std::pow(small_displacement_threshold * (T_ss + T_ds), 2), "init_step_displacement_sq");

    GRBLinExpr dx_final = footsteps[N-1][0] - footsteps[N-3][0];
    GRBLinExpr dy_final = footsteps[N-1][1] - footsteps[N-3][1];
    model_upper.addQConstr(dx_final*dx_final + dy_final*dy_final <= std::pow(small_displacement_threshold * (T_ss + T_ds), 2), "final_step_displacement_sq");

    // Max rotation per step
    double del_theta_max = M_PI / 8.0;
    for (int c = 1; c < N; ++c) 
    {
        GRBLinExpr delta_theta = footsteps[c][3] - footsteps[c-1][3];
        model_upper.addConstr(delta_theta <= del_theta_max, "max_rot_pos_c" + std::to_string(c));
        model_upper.addConstr(delta_theta >= -del_theta_max, "max_rot_neg_c" + std::to_string(c));
    }
    
}

void doublelayer_planning::solve_upper_layer()
{
    cout<<"Solve upper layer here"<<endl;
    GRBQuadExpr total_objective = 0;
    // Terminal cost (Scenario 3 goal)
    std::array<double, 4> g_target = {1.5, 2.2, 0, 3.0 * M_PI / 4.0};
    GRBLinExpr e0 = footsteps[N-1][0] - g_target[0]; // x
    GRBLinExpr e1 = footsteps[N-1][1] - g_target[1]; // y
    GRBLinExpr e2 = footsteps[N-1][2] - g_target[2]; // z
    GRBLinExpr e3 = footsteps[N-1][3] - g_target[3]; // theta
    
    // std::array<std::array<double,3>,3> Q_mat = {{ {300,0,0}, {0,300,0}, {0,0,300} }};
    Eigen::Matrix4d Q_mat = Eigen::Matrix4d::Identity() * 300.0; // Diagonal matrix for terminal cost
    GRBQuadExpr term_cost = 0;
    // term_cost += e0*e0*Q_mat[0][0] + e0*e1*Q_mat[0][1] + e0*e2*Q_mat[0][2];
    // term_cost += e1*e0*Q_mat[1][0] + e1*e1*Q_mat[1][1] + e1*e2*Q_mat[1][2];
    // term_cost += e2*e0*Q_mat[2][0] + e2*e1*Q_mat[2][1] + e2*e2*Q_mat[2][2];
    term_cost += e0*e0*Q_mat(0, 0);
    term_cost += e1*e1*Q_mat(1, 1);
    term_cost += e2*e2*Q_mat(2, 2);
    term_cost += e3*e3*Q_mat(3, 3);
    total_objective += term_cost;

    // Incremental cost
    // std::array<std::array<double,3>,3> R_mat = {{ {0.5,0,0}, {0,0.5,0}, {0,0,0.5} }};
    Eigen::Matrix3d R_mat = Eigen::Matrix3d::Identity() * 0.5;
    GRBQuadExpr inc_cost = 0;
    // Python: for j in range(0,N). If j=0, footsteps[j-1] is invalid.
    // Assuming it's range(1,N) for differences, or special handling for j=0.
    // The Python code `footsteps[j][0]-footsteps[j-1][0]` will fail for j=0 if footsteps[-1] is not defined.
    // Let's assume it was meant for j from 1 to N-1 (N-1 differences).
    // Or, if it's N terms, then footsteps[-1] is perhaps footsteps[N-1] (cyclic) or 0.
    // Given the context of path smoothness, range(1,N) seems more likely.
    for (int j = 1; j < N; ++j) 
    {
        GRBLinExpr dx = footsteps[j][0] - footsteps[j-1][0];
        GRBLinExpr dy = footsteps[j][1] - footsteps[j-1][1];
        GRBLinExpr dtheta = footsteps[j][3] - footsteps[j-1][3];
        inc_cost += dx*dx*R_mat(0, 0) + dy*dy*R_mat(1, 1) + dtheta*dtheta*R_mat(2, 2);
        // inc_cost += dx*dx*R_mat[0][0] + dy*dy*R_mat[1][1] + dtheta*dtheta*R_mat[2][2];
        // // Add off-diagonal terms if R_mat is not diagonal
        // inc_cost += dx*dy*R_mat[0][1] + dx*dtheta*R_mat[0][2];
        // inc_cost += dy*dx*R_mat[1][0] + dy*dtheta*R_mat[1][2];
        // inc_cost += dtheta*dx*R_mat[2][0] + dtheta*dy*R_mat[2][1];
    }
    total_objective += inc_cost;

    double T_cycle = T_ss + T_ds; // Cycle time for velocity smoothness
    double G_vel_smoothness = 5.0;
    GRBQuadExpr velocity_smoothness_cost_expr = 0;

    if (N > 2) 
    { // Need at least 3 footsteps for 2 midpoints -> 1 velocity
        std::vector<std::pair<GRBLinExpr, GRBLinExpr>> midpoints_xy_exprs;
        for (int j = 1; j < N; ++j) 
        { // N-1 midpoints
            GRBLinExpr mid_x = (footsteps[j][0] + footsteps[j-1][0]) / 2.0;
            GRBLinExpr mid_y = (footsteps[j][1] + footsteps[j-1][1]) / 2.0;
            midpoints_xy_exprs.push_back({mid_x, mid_y});
        }

        if (midpoints_xy_exprs.size() > 1) 
        { // Need at least 2 midpoints for 1 velocity
            std::vector<std::pair<GRBLinExpr, GRBLinExpr>> velocities_xy_exprs;
            for (size_t k = 1; k < midpoints_xy_exprs.size(); ++k) 
            { // N-2 velocities
                GRBLinExpr vel_x = (midpoints_xy_exprs[k].first - midpoints_xy_exprs[k-1].first) / T_cycle;
                GRBLinExpr vel_y = (midpoints_xy_exprs[k].second - midpoints_xy_exprs[k-1].second) / T_cycle;
                velocities_xy_exprs.push_back({vel_x, vel_y});
            }
            
            if (velocities_xy_exprs.size() > 1) 
            { // Need at least 2 velocities for 1 acceleration
                for (size_t l = 1; l < velocities_xy_exprs.size(); ++l) 
                { // N-3 accelerations
                    GRBLinExpr accel_x = velocities_xy_exprs[l].first - velocities_xy_exprs[l-1].first;
                    GRBLinExpr accel_y = velocities_xy_exprs[l].second - velocities_xy_exprs[l-1].second;
                    velocity_smoothness_cost_expr += accel_x*accel_x + accel_y*accel_y;
                }
            }
        }
    }
    total_objective += G_vel_smoothness * velocity_smoothness_cost_expr;

    model_upper.setObjective(total_objective, GRB_MINIMIZE);
    model_upper.optimize();

    cout<<"Upper layer optimization complete."<<endl;
    // --- NEW DEBUGGING BLOCK ---
    int status = model_upper.get(GRB_IntAttr_Status);

    // <<< FIX: 更健壮的状态检查逻辑
    if (status == GRB_INF_OR_UNBD) {
        std::cerr << "Warning: Model is infeasible or unbounded. Turning off presolve and re-optimizing to distinguish." << std::endl;
        // 关闭预处理器
        model_upper.set(GRB_IntParam_Presolve, 0);
        // 重新求解
        model_upper.optimize();
        // 再次获取状态
        status = model_upper.get(GRB_IntAttr_Status);
    }


    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
        std::cout << "Upper layer optimization successful." << std::endl;
        std::cout << "Gurobi optimization time: " << model_upper.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
        std::cout << "Objective value: " << model_upper.get(GRB_DoubleAttr_ObjVal) << std::endl;
    } 
    else if (status == GRB_INFEASIBLE) { // 现在可以正确进入这个分支了
        std::cerr << "ERROR: Upper layer model is INFEASIBLE." << std::endl;
        std::cerr << "Computing IIS (Irreducible Inconsistent Subsystem) to find the conflict..." << std::endl;
        model_upper.computeIIS();
        model_upper.write("upper_layer_infeasible.ilp");
        std::cerr << "IIS written to 'upper_layer_infeasible.ilp'. Inspect this file." << std::endl;
        throw std::runtime_error("Upper layer failed to solve due to infeasibility.");
    }
    else if (status == GRB_UNBOUNDED) {
        std::cerr << "ERROR: Upper layer model is UNBOUNDED." << std::endl;
        std::cerr << "This usually means the objective can be improved infinitely." << std::endl;
        std::cerr << "Check if all variables have proper bounds or if the objective function is correct." << std::endl;
        throw std::runtime_error("Upper layer failed to solve due to being unbounded.");
    }
    else {
        std::cerr << "ERROR: Upper layer optimization failed with unhandled status code " << status << std::endl;
        throw std::runtime_error("Upper layer failed to solve.");
    }
    std::cout << "Gurobi optimization time: " << model_upper.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
    std::cout << "Objective value: " << model_upper.get(GRB_DoubleAttr_ObjVal) << std::endl;


    // 根据上层解算出来的步态，更新下层的初始x
    Eigen::Vector2d p_current = initial_stance_footsteps.col(1).head<2>(); // 假设p_current是 (0,0)
    double next_foot_x = footsteps[2][0].get(GRB_DoubleAttr_X);
    double next_foot_y = footsteps[2][1].get(GRB_DoubleAttr_X);
    Eigen::Vector2d p_next(next_foot_x, next_foot_y);

    Eigen::Vector2d r0 = initial_alip_x.head<2>(); // 初始CoM位置

    // 2. 计算 ω
    double omega = std::sqrt(g / H_com);

    // 3. 反解初始DCM (ξ₀)
    Eigen::Vector2d xi_0 = (p_next - p_current) * std::exp(-omega * T_ss) + p_current;

    // 4. 反解初始角动量 (L₀)
    Eigen::Vector2d L0_xy = mass * omega * (xi_0 - r0);

    // 5. 更新 initial_alip_x
    // 注意ALIP状态和物理角动量的关系: L_alip = [Lx, Ly], L_physics = [Lx, Ly, Lz]
    // CoM在(x,y)平面, 角动量 L = r x (m*v)
    // Lx = y*m*vz - z*m*vy (通常忽略 z) ≈ -H*m*vy
    // Ly = z*m*vx - x*m*vz (通常忽略 z) ≈ H*m*vx
    // 我们计算出的 L0_xy 是 (Lx_d, Ly_d), 但需要注意符号和坐标系
    // 假设 L_alip = [Lx, Ly]
    // 我们的模型中，L = [Lx, Ly]^T，其中 Lx 产生 y 方向运动，Ly 产生 x 方向运动
    // 所以，我们需要将计算出的 L0_xy 映射到 ALIP 状态的 z 和 w 分量

    // 重新审视 L0_xy 与 (vx, vy) 的关系
    // L0_xy = m * omega * (xi_0 - r0) = L_physics / H
    // 所以，我们计算出的 L0_xy(0) 对应物理的 Lx/H, L0_xy(1) 对应物理的 Ly/H.
    // 而在你的模型中，ALIP状态的第3个分量是Lx，第4个是Ly。
    // 所以 L0_xy(0) -> Lx, L0_xy(1) -> Ly
    initial_alip_x(2) = L0_xy(0); // 设置初始 Lx
    initial_alip_x(3) = L0_xy(1); // 设置初始 Ly

    std::cout << "Computed initial ALIP state based on first step:" << std::endl;
    std::cout << initial_alip_x.transpose() << std::endl;
}


void doublelayer_planning::add_lower_layer_variables()
{    
    // N 为包含初始步的所有步态 
    x_alip_vars = std::vector<std::vector<std::vector<GRBVar>>>(N-2, std::vector<std::vector<GRBVar>>(K_knots, std::vector<GRBVar>(4)));
    v_ankle_vars = std::vector<std::vector<GRBVar>>(N-2, std::vector<GRBVar>(K_knots - 1));
    u_ankle_vars = std::vector<std::vector<GRBVar>>(N-2, std::vector<GRBVar>(K_knots - 1));

    for (int n = 0; n < N - 2; ++n) 
    {
        for (int k = 0; k < K_knots; ++k) 
        {
            for (int i = 0; i < 4; ++i) 
            {
                x_alip_vars[n][k][i] = model_lower.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x_n" + std::to_string(n) + "_k" + std::to_string(k) + "_" + std::to_string(i));
            }
            if (k < K_knots - 1) 
            {
                u_ankle_vars[n][k] = model_lower.addVar(-10.0, 10.0, 0.0, GRB_CONTINUOUS, "u_n" + std::to_string(n) + "_k" + std::to_string(k));
                v_ankle_vars[n][k] = model_lower.addVar(-10.0, 10.0, 0.0, GRB_CONTINUOUS, "v_n" + std::to_string(n) + "_k" + std::to_string(k));
            }
        }
    }
        
    p_foot_vars.resize(N - 2, std::vector<GRBVar>(3)); // 3 coordinates: x, y, 0
    for (int n = 0; n < N - 2; ++n) 
    {
        for (int j = 0; j < 3; ++j) 
        {
            p_foot_vars[n][j] = model_lower.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "p_n" + std::to_string(n+1) + "_" + std::to_string(j));
        }
    }

    mu_vars.resize(N - 2, std::vector<GRBVar>(N_REG)); // N_horizon x N_REGIONS binary variables
    for (int n = 0; n < N - 2; ++n) 
    {
        for (int i_reg = 0; i_reg < N_REG; ++i_reg) 
        {
            mu_vars[n][i_reg] = model_lower.addVar(0.0, 1.0, 0.0, GRB_BINARY, "mu_n" + std::to_string(n+1) + "_reg" + std::to_string(i_reg));
        }
    }
    model_lower.update(); // Important after adding vars
}

void doublelayer_planning::add_lower_layer_constraints()
{
    // 不再以速度驱动
    // Eigen::Matrix<double, 4, 2> Br_map_for_cycle_2d = B_r.block<4,2>(0,0);
    // // // // 设置理想的周期参考状态
    // auto [ref_state_cycle_phase0, ref_state_cycle_phase1] = calculate_periodic_alip_reference_states(
    //     centroids_velocity.front().x(), centroids_velocity.front().y(), nominal_stance_width,
    //     (T_ss + T_ds), A_s2s_autonomous_cycle, Br_map_for_cycle_2d,
    //     current_stance_is_left
    // );
    // Eigen::Vector4d x_current_alip_val_eigen = current_stance_is_left ? ref_state_cycle_phase0 : ref_state_cycle_phase1;
    // cout<<"x_current_alip_val_eigen: "<<x_current_alip_val_eigen.transpose()<<endl;
    cout<<"initial_alip_x: "<<initial_alip_x.transpose()<<endl;
    // Add white noise to initial_alip_x
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist1(0.0, 0.01); // Mean 0, standard deviation 0.01
    std::normal_distribution<double> noise_dist2(0.0, 0.01); // Mean 0, standard deviation 0.01
    std::normal_distribution<double> noise_dist3(0.0, 0.04); // Mean 0, standard deviation 0.01
    std::normal_distribution<double> noise_dist4(0.0, 0.04); // Mean 0, standard deviation 0.01

    initial_alip_x(0) += noise_dist1(gen); // Add noise to x
    initial_alip_x(1) += noise_dist2(gen); // Add noise to y
    initial_alip_x(2) += noise_dist3(gen); // Add noise to Lx
    initial_alip_x(3) += noise_dist4(gen); // Add noise to Ly
    cout<<"initial_alip_x: "<<initial_alip_x.transpose()<<endl;

    for (int i = 0; i < 4; ++i) 
    {
        model_lower.addConstr(x_alip_vars[0][0][i] == initial_alip_x(i), "init_alip_state_" + std::to_string(i));
    }

    // // 计算参考轨迹
    // x_d_horizon.resize(N-2, std::vector<Eigen::Vector4d>(K_knots));
    // for (int n = 0; n < N-2; ++n) 
    // {
    //     // 1. 确定为计算阶段 n 的参考周期，应该假设哪只脚是名义上的起始支撑脚
    //     bool is_left_stance_for_stage_n_dynamics = (n % 2 == 0) ? current_stance_is_left : !current_stance_is_left;
    //     // cout<<"n: "<<n<<" is_left_stance_for_stage_n_dynamics: "<<is_left_stance_for_stage_n_dynamics<<endl;
    //     // 2. 根据阶段 n 的期望速度 和 上面确定的名义支撑脚，计算该阶段的参考周期状态
    //     // cout<<"n: "<<n<<" centroids_velocity.at(n).x(): "<<centroids_velocity.at(n).x()<<" centroids_velocity.at(n).y(): "<<centroids_velocity.at(n).y()<<endl;
    //     auto [cycle_p0_for_stage_n, cycle_p1_for_stage_n] = calculate_periodic_alip_reference_states(
    //         centroids_velocity.at(n).x(),  // vx_d for stage n
    //         centroids_velocity.at(n).y(), // vy_d for stage n
    //         nominal_stance_width,
    //         T_ss + T_ds, A_s2s_autonomous_cycle, Br_map_for_cycle_2d,
    //         is_left_stance_for_stage_n_dynamics // 这个参数是关键
    //     );
    //     Eigen::Vector4d x_d_nk_start_of_stage = cycle_p0_for_stage_n;
    //     // 4. 在该阶段内自主传播
    //     Eigen::Vector4d x_d_nk = x_d_nk_start_of_stage;
    //     x_d_horizon[n][0] = x_d_nk;
    //     // cout<<"n: "<<n<<" x_d_horizon[0]: "<<x_d_horizon[n][0].transpose()<<endl;
    //     for (int k = 1; k < K_knots; ++k) {
    //         x_d_nk = A_d_autonomous_knot * x_d_nk;
    //         x_d_horizon[n][k] = x_d_nk;
    //         // cout<<"n: "<<n<<" k: "<<k<<" x_d_horizon[n][k]: "<<x_d_horizon[n][k].transpose()<<endl;
    //     }
    // }

    // --- Constraints ---
    // 9b: ALIP dynamics within a single support phase
    for (int n = 0; n < N-2; ++n) 
    {
        for (int k = 0; k < K_knots - 1; ++k) 
        {
            for (int row = 0; row < 4; ++row) 
            {
                model_lower.addConstr(x_alip_vars[n][k+1][row] == gurobi_quicksum(A_d_autonomous_knot, row, x_alip_vars[n][k]) + B_d_for_mpc(row,0) * v_ankle_vars[n][k] + B_d_for_mpc(row, 1) * u_ankle_vars[n][k], "alip_dyn_n" + std::to_string(n) + "_k" + std::to_string(k) + "_r" + std::to_string(row));
            }
        }
    }
    cout<<"Added ALIP dynamics constraints for single support phase."<<endl;
    // 注意这里缺少一个环节，因为上层规划的落脚点没有z坐标，下层规划的ALIP需要z坐标
    // 这里的z需要根据初始时刻的双脚与平面的约束来决定

    // 9c: ALIP reset map (double support phase dynamics)
    for (int n = 0; n < N-3; ++n) {
        GRBLinExpr dp_x, dp_y, dp_z;
        if (n == 0) 
        {  
            // 第二列为初始支撑脚的落脚点
            dp_x = p_foot_vars[n][0] - initial_stance_footsteps(0, 1);
            dp_y = p_foot_vars[n][1] - initial_stance_footsteps(1, 1);
            dp_z = p_foot_vars[n][2] - initial_stance_footsteps(2, 1); // Assuming z=0 for initial stance foot
        }
        else 
        {
            dp_x = p_foot_vars[n][0] - p_foot_vars[n-1][0];
            dp_y = p_foot_vars[n][1] - p_foot_vars[n-1][1];
            dp_z = p_foot_vars[n][2] - p_foot_vars[n-1][2];
        }

        for (int row = 0; row < 4; ++row) 
        {
            GRBLinExpr br_term = B_r(row,0) * dp_x + B_r(row,1) * dp_y + B_r(row,2) * dp_z;
            model_lower.addConstr(x_alip_vars[n+1][0][row] == gurobi_quicksum(Ar_ds, row, x_alip_vars[n][K_knots-1]) + br_term, "alip_reset_n" + std::to_string(n) + "_r" + std::to_string(row));
        }
    }
    cout<<"Added ALIP reset map constraints for double support phase."<<endl;
    // 9d: Foothold region constraints
    for (int c = 0; c < N - 2; ++c) 
    {
        for (int r = 0; r < N_REG; ++r) 
        {
            std::pair<Eigen::MatrixX3d, Eigen::VectorXd> FC = polygons.at(r).calculateCoefficients();
            Eigen::MatrixX3d F = FC.first; // Coefficients for inequalities
            Eigen::VectorXd c_vec = FC.second; // Constants for inequalities
            for (int i = 0; i < F.rows(); i++)
            {
                GRBLinExpr expr = 0;
                for (int j = 0; j < 3; ++j) 
                { // x, y, theta
                    expr += F(i, j) * p_foot_vars[c][j];
                }
                model_lower.addConstr(expr <= c_vec(i) + M_BIG * (1.0 - mu_vars[c][r]), "region_constr_c" + std::to_string(c) + "_r" + std::to_string(r) + "_i" + std::to_string(i));
            }
            // model_upper.addConstr(footsteps[c][2] <= theta_upper_bound + M_BIG * (1.0 - H_vars[c][r]), "theta_upper_c" + std::to_string(c) + "_r" + std::to_string(r));
            // for (int i = 0; i < 6; ++i) 
            // { // 6 inequalities per region
            //     GRBLinExpr expr = 0;
            //     for (int j = 0; j < 3; ++j) 
            //     { // x, y, theta
            //         expr += A_regions[r][i][j] * footsteps[c][j];
            //     }
            //     model_upper.addConstr(expr - b_regions[r][i] <= M_BIG * (1.0 - H_vars[c][r]), "region_constr_c" + std::to_string(c) + "_r" + std::to_string(r) + "_i" + std::to_string(i));
            // }

        }
        // Sum of H_vars[c][j] for j must be 1
        GRBLinExpr sum_H = 0;
        for (int j = 0; j < N_REG; ++j) 
        {
            sum_H += mu_vars[c][j];
        }
        model_lower.addConstr(sum_H == 1.0, "sum_H_c" + std::to_string(c));
    }
    cout<<"Added foothold region constraints."<<endl;
    // 9d: Foothold region constraints
    // const double M_big = 100.0;
    // for (int n = 0; n < N - 2; ++n) 
    // { // p_foot_vars[n] is p_{n+1}
    //     for (int i_reg = 0; i_reg < N_REG; ++i_reg) 
    //     {
    //         const Eigen::MatrixXd& F_mat = regions_F_list[i_reg]; // Should be 6x3
    //         const Eigen::VectorXd& c_vec = regions_c_list[i_reg]; // Should be 6x1
    //         for (int row_idx = 0; row_idx < F_mat.rows(); ++row_idx) 
    //         {
    //             GRBLinExpr Fp_expr = 0;
    //             for (int col_idx = 0; col_idx < 3; ++col_idx) 
    //             {
    //                 Fp_expr += F_mat(row_idx, col_idx) * p_foot_vars[n][col_idx];
    //             }
    //             model_lower.addConstr( Fp_expr <= c_vec(row_idx) + M_big * (1.0 - mu_vars[n][i_reg]), "foothold_n" + std::to_string(n+1) + "_reg" + std::to_string(i_reg) + "_row" + std::to_string(row_idx)
    //             );
    //         }
    //     }
    //     GRBLinExpr sum_mu = 0;
    //     for(int i_reg=0; i_reg < N_REG; ++i_reg) sum_mu += mu_vars[n][i_reg];
    //     model_lower.addConstr(sum_mu == 1.0, "sum_mu_n" + std::to_string(n+1));
    // }
    // 


    // // CoM y-excursion limits
    // const double y_com_max = 0.15;
    // for (int n = 0; n < N - 2; ++n) 
    // {
    //     for (int k = 0; k < K_knots; ++k) 
    //     {
    //         model_lower.addConstr(x_alip_vars[n][k][1] <= y_com_max, "ycom_max_n" + std::to_string(n) + "_k" + std::to_string(k));
    //         model_lower.addConstr(x_alip_vars[n][k][1] >= -y_com_max, "ycom_min_n" + std::to_string(n) + "_k" + std::to_string(k));
    //     }
    // }

    // Kinematic limits (p_{n+1} - p_n)
    const double max_dx = 0.4, max_dy = 0.3, max_dz = 0.2; // Adjust these limits as needed
    // std::vector<GRBVar> prev_p_for_kin_limit_vars(3); // Not needed if p_current is const
    // Eigen::Vector3d prev_p_val_for_kin = initial_stance_footsteps.head(3);
    Eigen::Vector3d prev_p_val_for_kin = initial_stance_footsteps.col(1).head(3); // Initial stance foot position
    
    // Eigen::Vector3d(initial_stance_footsteps(0, 1), initial_stance_footsteps(1, 1), 0.0); // Assuming z=0 for initial stance foot

    for (int n = 0; n < N - 2; ++n) 
    { // p_foot_vars[n] is p_{n+1}
        GRBLinExpr dx, dy, dz;
        if (n == 0) 
        {
                dx = p_foot_vars[n][0] - prev_p_val_for_kin(0);
                dy = p_foot_vars[n][1] - prev_p_val_for_kin(1);
                dz = p_foot_vars[n][2] - prev_p_val_for_kin(2);
        }
        else 
        {
                dx = p_foot_vars[n][0] - p_foot_vars[n-1][0];
                dy = p_foot_vars[n][1] - p_foot_vars[n-1][1];
                dz = p_foot_vars[n][2] - p_foot_vars[n-1][2];
        }
        model_lower.addConstr(dx <= max_dx, "max_dx_n" + std::to_string(n+1));
        model_lower.addConstr(dx >= -max_dx, "min_dx_n" + std::to_string(n+1));
        model_lower.addConstr(dy <= max_dy, "max_dy_n" + std::to_string(n+1));
        model_lower.addConstr(dy >= -max_dy, "min_dy_n" + std::to_string(n+1));
        model_lower.addConstr(dz <= max_dz, "max_dz_n" + std::to_string(n+1));
        model_lower.addConstr(dz >= -max_dz, "min_dz_n" + std::to_string(n+1));
        // prev_p_for_kin_limit_vars becomes p_foot_vars[n] for next iter, handled by n-1 indexing
    }

}


void doublelayer_planning::solve_lower_layer()
{
    cout<<"Solving lower layer optimization problem..."<<endl;
    // --- Objective Function ---
    GRBQuadExpr objective = 0;

    // --- 1. 终端成本 (Terminal Cost on Final State) ---
    // 让最终的ALIP状态尽可能稳定（例如，角动量为零）
    Eigen::Matrix4d Q_terminal = Eigen::Matrix4d::Zero();
    Q_terminal(2,2) = 8.0; // 惩罚 Lx
    Q_terminal(3,3) = 8.0; // 惩罚 Ly
    // 你也可以惩罚最终的 CoM 位置，让它停在最后一个落脚点中间
    // ...
    GRBLinExpr err_Lx_final = x_alip_vars[N-3][K_knots-1][2] - 0.0;
    GRBLinExpr err_Ly_final = x_alip_vars[N-3][K_knots-1][3] - 0.0;
    objective += err_Lx_final * Q_terminal(2,2) * err_Lx_final;
    objective += err_Ly_final * Q_terminal(3,3) * err_Ly_final;

    // --- 2. 落脚点追踪成本 (Footstep Tracking Cost) ---
    // 这是核心！强力追踪上层的footsteps
    double W_footstep = 2; // 非常大的权重
    for (int n = 0; n < N - 2; ++n) 
    {
        // 目标落脚点 p_d 来自上层规划
        double target_px = footsteps[n + 2][0].get(GRB_DoubleAttr_X);
        double target_py = footsteps[n + 2][1].get(GRB_DoubleAttr_X);
        double target_pz = footsteps[n + 2][2].get(GRB_DoubleAttr_X); // 假设平地

        GRBLinExpr err_px = p_foot_vars[n][0] - target_px;
        GRBLinExpr err_py = p_foot_vars[n][1] - target_py;
        GRBLinExpr err_pz = p_foot_vars[n][2] - target_pz;

        objective += W_footstep * (err_px * err_px + err_py * err_py + err_pz * err_pz);
    }

    // --- 3. 阶段成本 (Stage Cost for smoothness) ---
    // 惩罚控制输入，让动作平滑
    double W_input = 4;
    for (int n = 0; n < N - 2; ++n) {
        for (int k = 0; k < K_knots - 1; ++k) {
            objective += W_input * (u_ankle_vars[n][k] * u_ankle_vars[n][k] + v_ankle_vars[n][k] * v_ankle_vars[n][k]);
        }
    }
    
    // model_lower.setObjective(objective, GRB_MINIMIZE);
    // model_lower.optimize();

    
    // Eigen::Matrix4d Q_state = Eigen::Matrix4d::Identity(); // [1,1,1,1] diag
    // double R_input_val = 0.1;
    // double R_input_val_v = 0.2; // For v_ankle_vars

    // for (int n = 0; n < N- 2; ++n) 
    // {
    //     for (int k = 0; k < K_knots; ++k) 
    //     {
    //         for (int r_idx = 0; r_idx < 4; ++r_idx) 
    //         {
    //             GRBLinExpr err_r = x_alip_vars[n][k][r_idx] - x_d_horizon[n][k](r_idx);
    //             objective += err_r * Q_state(r_idx, r_idx) * err_r;
    //         }
    //         if (k < K_knots - 1) 
    //         {
    //             objective += u_ankle_vars[n][k] * R_input_val * u_ankle_vars[n][k];
    //         }
    //         if (k < K_knots - 1) 
    //         {
    //             objective += v_ankle_vars[n][k] * R_input_val_v * v_ankle_vars[n][k];
    //         }
    //     }
    // }
    
    // Eigen::Matrix4d Q_f_state = Q_state * 0.1;
    // for (int r_idx = 0; r_idx < 4; ++r_idx) 
    // {
    //     GRBLinExpr err_f_r = x_alip_vars[N -3][K_knots-1][r_idx] - x_d_horizon[N - 3][K_knots-1](r_idx);
    //     objective += err_f_r * Q_f_state(r_idx, r_idx) * err_f_r;
    // }
    model_lower.setObjective(objective, GRB_MINIMIZE);

    // --- Solve ---
    model_lower.set(GRB_DoubleParam_MIPGap, 0.05);
    model_lower.set(GRB_DoubleParam_TimeLimit, 2.5); // Slightly more time for C++ overhead potentially
    // model.set(GRB_IntParam_NonConvex, 2); // Only if quadratic constraints
    
    model_lower.optimize();

    // --- Process and Plot Results ---
    if (model_lower.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model_lower.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) 
    {
        std::cout << "MPC solution found." << std::endl;
        std::cout << "Gurobi optimization time: " << std::fixed << std::setprecision(4) << model_lower.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
        
        std::vector<Eigen::Vector3d> p_planned_vals_optimized_eigen(N - 2);
        std::cout << "\n--- Planned Footstep Sequence (p1, p2, ...) ---" << std::endl;
        for (int n_opt = 0; n_opt < N - 2; ++n_opt) 
        {
            for (int j = 0; j < 3; ++j) {
                p_planned_vals_optimized_eigen[n_opt](j) = p_foot_vars[n_opt][j].get(GRB_DoubleAttr_X);
            }
            std::cout << "p" << n_opt + 1 << ": x=" << std::fixed << std::setprecision(3) << p_planned_vals_optimized_eigen[n_opt](0)
                      << ", y=" << p_planned_vals_optimized_eigen[n_opt](1)
                      << ", z=" << p_planned_vals_optimized_eigen[n_opt](2) << std::endl;
        }
        std::cout << "\n--- Optimized Ankle Inputs (u_ankle_vars and v_ankle_vars) ---" << std::endl;
        // for (int n_opt = 0; n_opt < N - 2; ++n_opt) {
        //     for (int k_opt = 0; k_opt < K_knots - 1; ++k_opt) {
        //         double u_val = u_ankle_vars[n_opt][k_opt].get(GRB_DoubleAttr_X);
        //         double v_val = v_ankle_vars[n_opt][k_opt].get(GRB_DoubleAttr_X);
        //         std::cout << "Stage " << n_opt << ", Knot " << k_opt << ": u_ankle = " << std::fixed << std::setprecision(3) << u_val
        //                   << ", v_ankle = " << v_val << std::endl;
        //     }
        // }
        
        std::vector<std::vector<double>> mu_planned_vals_optimized(N - 2, std::vector<double>(N_REG));
        for (int n_opt = 0; n_opt < N - 2; ++n_opt) 
        {
            for (int i_reg = 0; i_reg < N_REG; ++i_reg) 
            {
                mu_planned_vals_optimized[n_opt][i_reg] = mu_vars[n_opt][i_reg].get(GRB_DoubleAttr_X);
            }
        }

        std::cout << "Optimal first planned footstep (p1): (" 
                  << std::fixed << std::setprecision(3) << p_planned_vals_optimized_eigen[0](0) << ", "
                  << p_planned_vals_optimized_eigen[0](1) << ", "
                  << p_planned_vals_optimized_eigen[0](2) << ")" << std::endl;

        for (int i_reg_check = 0; i_reg_check < N_REG; ++i_reg_check) {
            if (mu_planned_vals_optimized[0][i_reg_check] > 0.5) {
                std::cout << "Footstep p1 planned for region " << i_reg_check + 1 << std::endl;
                break;
            }
        }
        
        // plot_results_cpp(p_current_foot_val_eigen,
        //                  p_planned_vals_optimized_eigen,
        //                  mu_planned_vals_optimized,
        //                  regions_F_list, // Already Eigen matrices
        //                  regions_c_list, // Already Eigen vectors
        //                  N_horizon, N_REGIONS);
        std::cout << "Plot saved to mpfc_cpp_footsteps.png" << std::endl;
    } 
    else if (model_lower.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) 
    {
        std::cerr << "MPC problem is infeasible." << std::endl;
        model_lower.computeIIS();
        model_lower.write("mpfc_infeasible_cpp.ilp");
        std::cerr << "IIS written to mpfc_infeasible_cpp.ilp" << std::endl;
    } 
    else 
    {
        model_lower.computeIIS();
        model_lower.write("mpfc_infeasible_cpp.ilp");
        std::cerr << "Optimization ended with status: " << model_lower.get(GRB_IntAttr_Status) << std::endl;
    }
}
//     catch (GRBException& e) 
//     {
//         std::cerr << "Gurobi Error code = " << e.getErrorCode() << std::endl;
//         std::cerr << e.getMessage() << std::endl;
//         return 1;
//     } catch (const std::exception& e) {
//         std::cerr << "Standard Exception: " << e.what() << std::endl;
//         return 1;
//     } catch (...) {
//         std::cerr << "Unknown exception" << std::endl;
//         return 1;
//     }
// }

// 新增的绘图函数实现
// void doublelayer_planning::plot_results()
// {
//     plt::figure(); // 创建一个新的图形窗口
//     plt::title("Footstep Planning Results");
    
//     // 1. 绘制安全区域 (从 initial_env 移过来的代码)
//     for (auto & ploy : polygons)
//     {
//         std::vector<double> xs, ys;
//         for (const auto& vertex : ploy.vertices) 
//         {
//             xs.push_back(vertex.x());
//             ys.push_back(vertex.y());
//         }
//         // 闭合多边形以绘制边框
//         if (!xs.empty()) {
//             xs.push_back(xs.front());
//             ys.push_back(ys.front());
//         }

//         std::map<std::string, std::string> keywords;
//         keywords["color"] = "gray";
//         keywords["alpha"] = "0.3";
//         plt::fill(xs, ys, keywords);
//     }
    
//     // 2. 绘制上层规划的脚步
//     // std::vector<double> upper_x, upper_y;
//     // for (int i = 0; i < N; ++i) {
//     //     upper_x.push_back(footsteps[i][0].get(GRB_DoubleAttr_X));
//     //     upper_y.push_back(footsteps[i][1].get(GRB_DoubleAttr_X));
//     // }
//     // plt::plot(upper_x, upper_y, "ro-", {{"label", "Upper Layer Path"}}); // 红色圆点线

//     // // 3. 绘制下层规划的脚步
//     // std::vector<double> lower_x, lower_y;
//     // // 添加初始支撑脚
//     // lower_x.push_back(initial_stance_footsteps(0, 1));
//     // lower_y.push_back(initial_stance_footsteps(1, 1));
//     // for (int n = 0; n < N - 2; ++n) {
//     //     lower_x.push_back(p_foot_vars[n][0].get(GRB_DoubleAttr_X));
//     //     lower_y.push_back(p_foot_vars[n][1].get(GRB_DoubleAttr_X));
//     // }
//     // plt::plot(lower_x, lower_y, "bs--", {{"label", "Lower Layer Path"}}); // 蓝色方块虚线

//     // 设置坐标轴、图例等
//     // plt::xlabel("X (m)");
//     // plt::ylabel("Y (m)");
//     // plt::legend();
//     // plt::grid(true);
//     // plt::axis("equal");

//     // 最后调用 show
//     plt::show();
// }

void doublelayer_planning::run()
{
    initial_env();
    add_upper_layer_variables();
    add_upper_layer_constraints();
    solve_upper_layer();

   
    
    // Print results
    for (int i = 0; i < N; ++i) 
    {
        std::cout << "Footstep " << i << ": ";
        for (int j = 0; j < 5; ++j) 
        {
            std::cout << footsteps[i][j].get(GRB_DoubleAttr_X) << " ";
        }
        std::cout << std::endl;
    }

    // 以两点之间的中点作为质心
    // for (int i = 0; i < N; ++i) 
    // {
    //     for (int r = 0; r < N_REG; ++r) 
    //     {
    //         if (H_vars[i][r].get(GRB_DoubleAttr_X) > 0.5) 
    //         { // Assuming binary variables are either 0 or 1
    //             std::cout << "Footstep " << i << " is in region " << r << std::endl;
    //             break; // Each footstep can only belong to one region
    //         }            
    //     }
    // }

    // 求出质心的位置
    std::vector<Eigen::Vector2d> centroids;
    for (int i = 0; i < footsteps.size() - 1; i++)
    {
        double x_mid = (footsteps[i][0].get(GRB_DoubleAttr_X) + footsteps[i+1][0].get(GRB_DoubleAttr_X)) / 2.0;
        double y_mid = (footsteps[i][1].get(GRB_DoubleAttr_X) + footsteps[i+1][1].get(GRB_DoubleAttr_X)) / 2.0;
        centroids.emplace_back(Eigen::Vector2d(x_mid, y_mid));
    }
    
    for (int i = 0; i < centroids.size() - 1; i++)
    {
        centroids_velocity.emplace_back((centroids[i+1] - centroids[i]) / (T_ss + T_ds));
        // std::cout<<"velocity "<<i<<": "<<centroids_velocity[i].x()<<", "<<centroids_velocity[i].y()<<std::endl;
    }
    // Add lower layer variables and constraints here
    add_lower_layer_variables();
    add_lower_layer_constraints();
    solve_lower_layer();
    
    // 画出多边形
    // for (auto & ploy : polygons)
    // {
    //     vector<double> xs, ys;
    //     for (const auto& vertex : ploy.vertices) 
    //     {
    //         xs.push_back(vertex.x());
    //         ys.push_back(vertex.y());
    //     }
    //     std::map<std::string, std::string> keywords;
    //     keywords["color"] = "blue";
    //     keywords["alpha"] = "0.5"; // 50% 的透明度
    //     plt::fill(xs, ys, keywords);
    // }
    // plt::show();
    // plot_results();
}
doublelayer_planning::~doublelayer_planning()
{
}