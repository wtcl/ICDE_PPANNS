cmake_minimum_required(VERSION 3.27)
project(PPANNS)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/zhangyd/soft/intel/oneapiall/2024.1/include/" "/home/zhangyd/soft/intel/oneapi/mkl/2024.1/include/" "/home/zhangyd/soft/openblas/include/" hnswlib/hnswlib/)
link_directories("/home/zhangyd/soft/intel/oneapiall/2024.1/lib/" "/home/zhangyd/soft/intel/oneapi/mkl/2024.1/lib/" "/home/zhangyd/soft/openblas/lib/")


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
    if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
    else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
            message("set -mcpu=apple-m1 flag")
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -msse2 -DNDEBUG -Ofast -ffast-math -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -pthread -fno-rtti -flto -DMKL-ILP64 -lmkl_sequential -lmkl_intel_lp64 -lmkl_core -g0 -lopenblas -lgfortran -DCMAKE_BUILD_TYPE=Release -DEIGEN_FAST_MATH=1")
#
add_executable(PPANNS main.cpp
        hnswlib/hnswlib/bruteforce.h
        hnswlib/hnswlib/hnswalg.h
        hnswlib/hnswlib/hnswlib.h
        hnswlib/hnswlib/space_ip.h
        hnswlib/hnswlib/space_l2.h
        hnswlib/hnswlib/stop_condition.h
        hnswlib/hnswlib/visited_list_pool.h)
target_link_libraries(PPANNS mkl_intel_thread iomp5 mkl_core mkl_intel_lp64 pthread)




