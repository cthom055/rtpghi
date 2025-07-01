cmake_minimum_required(VERSION 3.14)
project(
  rtpghi
  VERSION 0.1.0
  LANGUAGES CXX)

# Header-only library
add_library(rtpghi INTERFACE)
target_include_directories(
  rtpghi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(rtpghi INTERFACE cxx_std_11)

# Testing
if(PROJECT_IS_TOP_LEVEL)
  enable_testing()

  # Fetch Catch2
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)

  # Test executable
  add_executable(tests tests/test_basic.cpp)
  target_link_libraries(tests PRIVATE rtpghi Catch2::Catch2WithMain)

  # Auto-discover tests
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)

  # Example
  add_executable(example examples/basic_example.cpp)
  target_link_libraries(example rtpghi)
endif()
