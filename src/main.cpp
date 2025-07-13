#include <doublelayer_planning.hpp>
// #include <matplotlibcpp.h>
using namespace std;
// namespace plt = matplotlibcpp;
// void plot_results(std::vector<polygon3D> polygons)
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

int main(int argc, char *argv[]) 
{
    try {
        doublelayer_planning planner;
        planner.run();
        // plot_results(planner.polygons);
    } catch (GRBException &e) {
        std::cerr << "Gurobi error code = " << e.getErrorCode() << std::endl;
        std::cerr << "Gurobi error message: " << e.getMessage() << std::endl;
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}