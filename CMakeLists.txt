cmake_minimum_required(VERSION 3.9)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -lpthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


project(main)
add_executable(main "")
target_link_libraries(main ${OpenCV_LIBS})
target_sources(main PRIVATE
    # test.cpp
    ./include/image.cpp
    ./include/image.h
    ./include/line_segmentation.cpp
    ./include/line_segmentation.h
    main.cpp
)
