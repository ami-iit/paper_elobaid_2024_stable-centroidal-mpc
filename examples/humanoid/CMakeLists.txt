cmake_minimum_required(VERSION 3.10)

# Project name
project(CentroidalMPCExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "default install path" FORCE)

# Find packages
find_package(BipedalLocomotionFramework REQUIRED)

# Add executable
add_executable(humanoid-test main.cpp)

# Link libraries
target_link_libraries(humanoid-test
    PRIVATE
    BipedalLocomotion::ParametersHandler
    BipedalLocomotion::ReducedModelControllers
    BipedalLocomotion::ContinuousDynamicalSystem
    BipedalLocomotion::Contacts
    BipedalLocomotion::Math
)

# Include directories
target_include_directories(humanoid-test PRIVATE ${BipedalLocomotionFramework_INCLUDE_DIRS})

# specify CMAKE_INSTALL_PREFIX as the current source/build directory


# Install executable
install(TARGETS humanoid-test
    RUNTIME DESTINATION bin
)

# Install additional files
install(FILES plot_data.m Step.m
    DESTINATION bin
)
