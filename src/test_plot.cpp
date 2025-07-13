// main_test_plot.cpp
#include "matplotlibcpp.h"
#include <vector>
#include <iostream>

namespace plt = matplotlibcpp;

int main() {
    std::cout << "Testing matplotlib-cpp plotting..." << std::endl;

    // 创建一个简单的矩形
    std::vector<double> xs = {1, 2, 2, 1, 1};
    std::vector<double> ys = {1, 1, 2, 2, 1};

    std::map<std::string, std::string> keywords;
    keywords["color"] = "blue";
    keywords["alpha"] = "0.5";

    try {
        plt::title("Minimal Plot Test");
        plt::fill(xs, ys, keywords);
        plt::grid(true);
        
        std::cout << "Calling plt::show()..." << std::endl;
        plt::show(); // 这里是关键
        std::cout << "plt::show() returned." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Program finished successfully." << std::endl;
    return 0;
}