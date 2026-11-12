cmake_minimum_required(VERSION 3.18)
project(rlprim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The C++ library is header-only. Link `rlprim::rlprim` or just copy include/.
add_library(rlprim INTERFACE)
add_library(rlprim::rlprim ALIAS rlprim)
target_include_directories(rlprim INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

option(RLPRIM_BUILD_TESTS "Build the C++ smoke test" OFF)
if(RLPRIM_BUILD_TESTS)
  enable_testing()
  add_executable(rlprim_smoke tests/cpp/smoke.cpp)
  target_link_libraries(rlprim_smoke PRIVATE rlprim::rlprim)
  add_test(NAME rlprim_smoke COMMAND rlprim_smoke)
endif()

# Python extension, built only when scikit-build-core drives the build.
if(SKBUILD)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE src/bindings.cpp)
  target_link_libraries(_core PRIVATE rlprim::rlprim)
  install(TARGETS _core DESTINATION rlprim)
endif()
