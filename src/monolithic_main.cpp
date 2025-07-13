
#include "monolithic_planner.hpp"
#include <iostream>

int main() {
    try {
        MonolithicPlanner planner;
        planner.run();
    } catch (const GRBException& e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << " (code " << e.getErrorCode() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}