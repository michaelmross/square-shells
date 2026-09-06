cmake_minimum_required(VERSION 3.20)
project(exact_k5_stream LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch a current, pinned primesieve release if it is not already installed.
find_package(primesieve QUIET)

if(NOT primesieve_FOUND)
    message(STATUS "Installed primesieve not found; fetching primesieve v12.15")
    FetchContent_Declare(
        primesieve
        GIT_REPOSITORY https://github.com/kimwalisch/primesieve.git
        GIT_TAG v12.15
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(primesieve)
endif()

add_executable(exact_k5_stream exact_k5_stream.cpp)
target_link_libraries(exact_k5_stream PRIVATE primesieve::primesieve)

if(MSVC)
    target_compile_options(exact_k5_stream PRIVATE /O2 /EHsc)
else()
    target_compile_options(exact_k5_stream PRIVATE -O3 -march=native)
endif()
