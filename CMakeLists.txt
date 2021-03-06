cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda-8.0/lib64)

project(marian CXX)
find_package(CUDA "8.0" REQUIRED)
if(CUDA_FOUND)
    set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
endif(CUDA_FOUND)

option(COMPILE_EXAMPLES "Compile examples" OFF)
option(COMPILE_TESTS "Compile tests" OFF)

SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O3 -Wno-unused-result -Wno-deprecated -fPIC -Wno-deprecated-gpu-targets")
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -g; -O3; --use_fast_math; -Xcompiler '-fPIC'; -arch=sm_35;)
LIST(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(ZLIB)
if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
else(ZLIB_FOUND)
    message(SEND_ERROR "Cannot find zlib" )
endif(ZLIB_FOUND)

find_package(Boost COMPONENTS system timer iostreams filesystem chrono program_options thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

include_directories(${marian_SOURCE_DIR}/src)
add_subdirectory(src)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
