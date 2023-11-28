cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(superglue)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(superglue superglue.cpp)
target_link_libraries(superglue ${TORCH_LIBRARIES} ${OpenCV_LIBS} -lstdc++fs)
include_directories(${OpenCV_INCLUDE_DIRS} include)

set_property(TARGET superglue PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/SuperPoint.zip
  ${CMAKE_CURRENT_SOURCE_DIR}/SuperGlue.zip
  DESTINATION ${CMAKE_BINARY_DIR}
)
