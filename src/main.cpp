#include <doublelayer_planning.hpp>

int main(int argc, char *argv[]) 
{
    try {
        doublelayer_planning planner;
        planner.run();
    } catch (GRBException &e) {
        std::cerr << "Gurobi error code = " << e.getErrorCode() << std::endl;
        std::cerr << "Gurobi error message: " << e.getMessage() << std::endl;
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}