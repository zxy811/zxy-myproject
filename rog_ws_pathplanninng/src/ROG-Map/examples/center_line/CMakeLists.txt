cmake_minimum_required(VERSION 3.0.2)
project(center_line)

# 使用C++14标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找catkin宏和库
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  rog_map
)

# 查找Eigen库
find_package(Eigen3 REQUIRED)

# 声明catkin包
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES center_line
  CATKIN_DEPENDS roscpp std_msgs nav_msgs sensor_msgs visualization_msgs rog_map
  DEPENDS EIGEN3
)

# 包含目录 - 关键修改
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 创建库文件
add_library(${PROJECT_NAME}
  src/center_line.cpp
)

# 添加依赖
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 链接库
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# 添加可执行文件
add_executable(center_line_node src/center_line_node.cpp)

# 添加依赖
add_dependencies(center_line_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 链接库
target_link_libraries(center_line_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# 安装目标
install(TARGETS ${PROJECT_NAME} center_line_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
