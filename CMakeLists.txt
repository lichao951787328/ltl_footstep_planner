cmake_minimum_required(VERSION 3.10)
project(FootstepPlannerCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Gurobi ---
# Option 1: Set GUROBI_HOME environment variable
# Option 2: Provide GUROBI_HOME as a CMake variable: cmake -DGUROBI_HOME=/path/to/gurobi ...
# Option 3: If Gurobi's FindGUROBI.cmake is in CMAKE_MODULE_PATH
set(GUROBI_HOME_ENV $ENV{GUROBI_HOME})
if(NOT GUROBI_HOME AND GUROBI_HOME_ENV)
    set(GUROBI_HOME ${GUROBI_HOME_ENV})
    message(STATUS "Using GUROBI_HOME from environment: ${GUROBI_HOME}")
elseif(NOT GUROBI_HOME)
    message(FATAL_ERROR "GUROBI_HOME is not set. Please set it as an environment variable or CMake variable.")
endif()

set(GUROBI_INCLUDE_DIR "${GUROBI_HOME}/include")
if(WIN32)
    set(GUROBI_LIBRARY "${GUROBI_HOME}/lib/gurobi_c++md2017.lib") # Adjust VS version if needed
    set(GUROBI_CXX_LIBRARY "${GUROBI_HOME}/lib/gurobi110.lib") # Adjust Gurobi version
else()
    set(GUROBI_LIBRARY "${GUROBI_HOME}/src/build/libgurobi_c++.a")
    set(GUROBI_CXX_LIBRARY "${GUROBI_HOME}/lib/libgurobi110.so") # Adjust Gurobi version
endif()

include_directories(${GUROBI_INCLUDE_DIR})

# --- Eigen3 ---
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# find_package(glog REQUIRED)

# --- Matplotlib-cpp (requires Python) ---
find_package(Python3 3.8 EXACT REQUIRED COMPONENTS Interpreter Development NumPy)
# matplotlib-cpp.h should be in your include path or project directory
# For simplicity, assuming it's in the same directory as CMakeLists.txt or a known include path
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # If matplotlibcpp.h is here
include_directories(${PYTHON_INCLUDE_DIRS})
if(UNIX AND NOT APPLE)
  find_package(Threads REQUIRED)
endif()
# --- Executable ---
add_executable(footstep_planner2 footstep_planar2.cpp)
add_executable(model_plannar model_plannar.cpp)
add_executable(model_plannar_ankle model_plannar_ankle.cpp)
add_executable(doublelayer_planning src/doublelayer_planning.cpp src/main.cpp)

# add_executable(model_palnar_v1 model_palnar_v1.cpp)

target_link_libraries(model_plannar_ankle
    PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
    Eigen3::Eigen # Modern Eigen usage
    Python3::Python # For matplotlib-cpp
    # golg::glog # Google Logging
    # Add pthread on Linux if matplotlib-cpp requires it (often does)
    # $<$<PLATFORM_ID:Linux>:Threads::Threads> # Or just -pthread
)


target_link_libraries(model_plannar
    PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
    Eigen3::Eigen # Modern Eigen usage
    Python3::Python # For matplotlib-cpp
    # golg::glog # Google Logging
    # Add pthread on Linux if matplotlib-cpp requires it (often does)
    # $<$<PLATFORM_ID:Linux>:Threads::Threads> # Or just -pthread
)

target_link_libraries(doublelayer_planning
    PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
    Eigen3::Eigen # Modern Eigen usage
    Python3::Python # For matplotlib-cpp
    # golg::glog # Google Logging
    # Add pthread on Linux if matplotlib-cpp requires it (often does)
    # $<$<PLATFORM_ID:Linux>:Threads::Threads> # Or just -pthread
)

# target_link_libraries(model_palnar_v1
#     PRIVATE
#     ${GUROBI_LIBRARY}
#     ${GUROBI_CXX_LIBRARY}
#     Eigen3::Eigen # Modern Eigen usage
#     Python3::Python # For matplotlib-cpp
#     # Add pthread on Linux if matplotlib-cpp requires it (often does)
#     # $<$<PLATFORM_ID:Linux>:Threads::Threads> # Or just -pthread
# )

target_link_libraries(footstep_planner2
    PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
    Eigen3::Eigen # Modern Eigen usage
    Python3::Python # For matplotlib-cpp
    # Add pthread on Linux if matplotlib-cpp requires it (often does)
    # $<$<PLATFORM_ID:Linux>:Threads::Threads> # Or just -pthread
)


if(UNIX AND NOT APPLE)
    target_link_libraries(footstep_planner2 PRIVATE Threads::Threads) # For pthread
endif()

# Ensure Gurobi DLL/SO can be found at runtime if not in system path
if(WIN32)
    # For Windows, you might need to copy gurobiXX.dll to the executable directory
    # or add Gurobi's bin directory to PATH.
    # This CMake snippet can help for build-time copying:
    # add_custom_command(TARGET footstep_planner2 POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #     "${GUROBI_HOME}/bin/gurobi110.dll" # Adjust Gurobi version
    #     $<TARGET_FILE_DIR:footstep_planner2>)
elseif(APPLE)
    # For macOS, set rpath
    set_target_properties(footstep_planner2 PROPERTIES
        INSTALL_RPATH "${GUROBI_HOME}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
else() # Linux
    set_target_properties(footstep_planner2 PROPERTIES
        INSTALL_RPATH "${GUROBI_HOME}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()