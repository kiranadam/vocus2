cmake_minimum_required(VERSION 2.8) 
PROJECT(vocus2)
find_package(OpenCV REQUIRED)
set(NAME_SRC
    	src/vocus2.cpp
	src/vocus_test.cpp	
)

set(NAME_HEADERS       
     include/vocus2.hpp
)

# add a compiler flag for the C++2011 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -fopenmp")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(vocus2 ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries(vocus2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
