# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTPGHI is a header-only C++ library implementing the Real-Time Phase Gradient Heap Integration algorithm for phase vocoding. The library is currently under development with a complete API design and placeholder implementation.

## Build System & Commands

### Building and Testing
```bash
# Build project (creates tests and example executable)
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build

# Run tests with verbose output on failure
ctest --test-dir build --output-on-failure

# Run example
./build/example
```

### Code Formatting and Quality
```bash
# Format all C++ code (required before commits)
clang-format -i include/rtpghi/*.hpp tests/*.cpp examples/*.cpp

# Install and run pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Architecture

### Core Components
- **`include/rtpghi/rtpghi.hpp`** - Single header-only library containing the complete API
- **`rtpghi::FrameInput`** - Input data structure for FFT magnitudes, previous phases, and gradients
- **`rtpghi::FrameOutput`** - Output buffers for processed magnitudes and phases
- **`rtpghi::process()`** - Main processing function (currently placeholder implementation)

### Key Design Principles
- **Header-only library** for easy integration into other projects
- **Zero allocation** during processing for real-time compatibility
- **C++11 compatibility** for broad platform support
- **Comprehensive validation** with detailed error codes

### Development Status
- ✅ Complete API design and structure
- ✅ Comprehensive test suite using Catch2
- ✅ Cross-platform CMake build system
- 🚧 RTPGHI algorithm implementation (placeholder currently does simple phase integration)

## Code Style

The project enforces strict code style via clang-format:
- **Braces**: Always on separate lines (Microsoft style base)
- **Naming**: snake_case for variables and functions
- **Indentation**: 4 spaces, no tabs
- **Line length**: 120 characters maximum
- **Braces required**: Always use braces for if/else/for/while statements

## Testing

Tests use Catch2 framework and are automatically discovered by CMake. All tests are in `tests/test_basic.cpp` and cover:
- Valid input processing
- Input/output validation
- Size mismatch handling
- Error code validation

## Development Workflow

1. Install pre-commit hooks for automatic formatting
2. Write tests first for new functionality
3. Implement features in the header-only library
4. Run tests to verify implementation
5. Format code before committing