#if(MSVC)
#    FILE(GLOB LIBRARY_FILES_C "*.cpp" "*.rc" "*.hpp")
#else()
#    FILE(GLOB LIBRARY_FILES_C "*.cpp" "*.hpp")
#endif()
#FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl" "*.hpp")
#
#cxx_executable_with_flags_no_pch(dense_mapping "Apps" "${cxx_default}" "MVS" ${LIBRARY_FILES_C} ${LIBRARY_FILES_H})
#
#if(OpenMVS_USE_BREAKPAD AND BREAKPAD_FOUND)
#    target_link_libraries(dense_mapping ${BREAKPAD_LIBS})
#endif()
#
## Install
#INSTALL(TARGETS dense_mapping
#        EXPORT OpenMVSTargets
#        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
cmake_minimum_required( VERSION 2.8 )
project( dense_monocular )

set(CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp" )

############### dependencies ######################
# Eigen
include_directories( "/usr/local/include/eigen3" )
# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus
set(Sophus_DIR /home/jxq/Sophus/Sophus/build)
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        )

add_executable( dense_mapping dense_mapping.cpp )
target_link_libraries( dense_mapping ${THIRD_PARTY_LIBS} )