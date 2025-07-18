cmake_minimum_required(VERSION 3.14)
project(
  rtpghi
  VERSION 0.1.0
  LANGUAGES CXX)

# Comprehensive warning configuration
function(set_project_warnings target_name)
  set(MSVC_WARNINGS
      /W4     # Baseline reasonable warnings
      /w14242 # 'identifier': conversion from 'type1' to 'type1', possible loss of data
      /w14254 # 'operator': conversion from 'type1:field_bits' to 'type1:field_bits', possible loss of data
      /w14263 # 'function': member function does not override any base class virtual member function
      /w14265 # 'classname': class has virtual functions, but destructor is not virtual
      /w14287 # 'operator': unsigned/negative constant mismatch
      /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside the for-loop scope
      /w14296 # 'operator': expression is always 'boolean_value'
      /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
      /w14545 # expression before comma evaluates to a function which is missing an argument list
      /w14546 # function call before comma missing argument list
      /w14547 # 'operator': operator before comma has no effect; expected operator with side-effect
      /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
      /w14555 # expression has no effect; expected expression with side-effect
      /w14619 # pragma warning: there is no warning number 'number'
      /w14640 # Enable warning on thread un-safe static member initialization
      /w14826 # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
      /w14905 # wide string literal cast to 'LPSTR'
      /w14906 # string literal cast to 'LPWSTR'
      /w14928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
      /WX     # Treat warnings as errors
  )

  set(CLANG_WARNINGS
      -Wall
      -Wextra                    # reasonable and standard
      -Wshadow                   # warn the user if a variable declaration shadows one from a parent context
      -Wnon-virtual-dtor         # warn the user if a class with virtual functions has a non-virtual destructor
      -Wold-style-cast           # warn for c-style casts
      -Wcast-align               # warn for potential performance problem casts
      -Wunused                   # warn on anything being unused
      -Woverloaded-virtual       # warn if you overload (not override) a virtual function
      -Wpedantic                 # warn if non-standard C++ is used
      -Wconversion               # warn on type conversions that may lose data
      -Wsign-conversion          # warn on sign conversions
      -Wnull-dereference         # warn if a null dereference is detected
      -Wdouble-promotion         # warn if float is implicit promoted to double
      -Wformat=2                 # warn on security issues around functions that format output (ie printf)
      -Wimplicit-fallthrough     # warn on statements that fallthrough without an explicit annotation
      -Wmisleading-indentation   # warn if indentation implies blocks where blocks do not exist
      -Werror                    # treat warnings as errors
  )

  set(GCC_WARNINGS
      ${CLANG_WARNINGS}
      -Wmisleading-indentation   # warn if indentation implies blocks where blocks do not exist
      -Wduplicated-cond          # warn if if / else chain has duplicated conditions
      -Wduplicated-branches      # warn if if / else branches have duplicated code
      -Wlogical-op               # warn about logical operations being used where bitwise were probably wanted
      -Wuseless-cast             # warn if you perform a cast to the same type
  )

  if(MSVC)
    set(PROJECT_WARNINGS_CXX ${MSVC_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(PROJECT_WARNINGS_CXX ${CLANG_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PROJECT_WARNINGS_CXX ${GCC_WARNINGS})
  else()
    message(AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
  endif()

  # Interface libraries work differently
  if(TARGET ${target_name})
    get_target_property(target_type ${target_name} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
      target_compile_options(${target_name} INTERFACE ${PROJECT_WARNINGS_CXX})
    else()
      target_compile_options(${target_name} PRIVATE ${PROJECT_WARNINGS_CXX})
    endif()
  endif()
endfunction()

# Header-only library
add_library(rtpghi INTERFACE)
target_include_directories(
  rtpghi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(rtpghi INTERFACE cxx_std_11)

# Apply warnings to library
set_project_warnings(rtpghi)

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
  add_executable(tests tests/test_basic.cpp tests/test_rtpghi_algorithm.cpp tests/test_rtpghi_realworld.cpp tests/test_rtpghi_advanced.cpp tests/test_gradients.cpp tests/test_benchmarks.cpp tests/test_heap.cpp tests/test_heap_benchmarks.cpp)
  target_link_libraries(tests PRIVATE rtpghi Catch2::Catch2WithMain)
  set_project_warnings(tests)

  # Auto-discover tests
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)

  # Examples
  add_executable(example examples/basic_example.cpp)
  target_link_libraries(example rtpghi)
  set_project_warnings(example)
  
  add_executable(complete_workflow examples/complete_workflow.cpp)
  target_link_libraries(complete_workflow rtpghi)
  set_project_warnings(complete_workflow)
endif()
