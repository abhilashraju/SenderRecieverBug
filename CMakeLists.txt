cmake_minimum_required(VERSION 3.1)

project (sender_reciever)

# set the C++ standard to C++ 17
set(CMAKE_CXX_STANDARD 17)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

add_compile_options(-O1 -Wall -Wextra -pedantic -fconcepts -lpthread) 

SET (PROJECT_ROOT "${PROJECT_SOURCE_DIR}")
SET (PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
SET (PROJECT_EXAMPLES_DIR "${PROJECT_ROOT}/examples")
SET (PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}/bin")
MESSAGE ("Source path: ${PROJECT_SOURCE_DIR}")
MESSAGE ("Binary path: ${PROJECT_BINARY_DIR}")

include_directories("
    ${PROJECT_SOURCE_DIR}
    ")

# Add an executable
add_executable(sender_reciever  ${PROJECT_SOURCE_DIR}/sender_reciever.cpp)

# link against the boost libraries
target_link_libraries( sender_reciever
    PRIVATE
        pthread
        
)