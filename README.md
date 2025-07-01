# RTPGHI - Real-Time Phase Gradient Heap Integration

A modern C++ implementation of the RTPGHI algorithm for phase vocoding, as
described in "Phase Vocoder Done Right" by Průša & Holighaus (2017).

## Overview

RTPGHI is a gradient-based phase vocoder algorithm that provides high-quality
time-stretching without the typical artifacts of traditional phase vocoders.
It uses phase gradient estimation and heap integration to maintain phase
coherence across frequency bins.

## Features

- **Header-only library** for easy integration
- **Zero allocation** during processing (real-time friendly)
- **C++11 compatible** for wide platform support
- **Cross-platform** (Windows, macOS, Linux)
- **Comprehensive testing** with Catch2

## Quick Start

### Basic Usage

```cpp
#include <rtpghi/rtpghi.hpp>
#include <vector>

// Setup your FFT data
const size_t fft_bins = 512;  // fft_size/2 + 1
std::vector<float> magnitudes(fft_bins);
std::vector<float> previous_phases(fft_bins);
std::vector<float> time_gradients(fft_bins);
std::vector<float> freq_gradients(fft_bins);

// Output buffers
std::vector<float> output_mags(fft_bins);
std::vector<float> output_phases(fft_bins);

// Process frame
rtpghi::FrameInput input{
    magnitudes.data(), previous_phases.data(),
    time_gradients.data(), freq_gradients.data(), fft_bins
};

rtpghi::FrameOutput output{
    output_mags.data(), output_phases.data(), fft_bins
};

auto result = rtpghi::process(input, output);
if (result == rtpghi::ErrorCode::OK) {
    // Use output_phases for inverse FFT
}
```

### CMake Integration

```cmake
# Add as subdirectory
add_subdirectory(rtpghi)
target_link_libraries(your_target rtpghi)
```

## Building

### Requirements

- CMake 3.14+
- C++11 compatible compiler
- Git (for fetching Catch2 during testing)

### Build Steps

```bash
git clone https://github.com/cthom055/rtpghi.git
cd rtpghi
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build

# Run example
./build/example
```

## API Reference

### Core Types

- `rtpghi::FrameInput` - Input frame data with gradients
- `rtpghi::FrameOutput` - Output buffers for processed data
- `rtpghi::ErrorCode` - Return codes for error handling

### Core Function

- `rtpghi::process(input, output)` - Process a single frame

## Algorithm Status

🚧 **Currently under development** 🚧

This library currently contains:

- ✅ Complete API design and structure
- ✅ Comprehensive test suite
- ✅ Cross-platform build system
- 🚧 RTPGHI algorithm implementation (in progress)

The placeholder implementation currently performs simple phase integration.
The full RTPGHI algorithm with heap integration will be implemented soon.

## Development

### Code Style

This project uses a consistent C++ code style enforced by clang-format:

- **Braces**: Always on their own lines
- **Naming**: snake_case for variables and functions
- **Indentation**: 4 spaces, no tabs
- **Line length**: 120 characters max

Format code before committing:

```bash
clang-format -i include/rtpghi/*.hpp tests/*.cpp examples/*.cpp
```

### Pre-commit Hooks

Install pre-commit hooks for automatic formatting and quality checks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically:

- Format C++ code with clang-format
- Check for trailing whitespace
- Validate YAML and Markdown files
- Format CMake files

### Running Tests

```bash
# Build and run tests
cmake -B build
cmake --build build
ctest --test-dir build

# Run with verbose output
ctest --test-dir build --output-on-failure
```

### Adding New Features

1. Write tests first in `tests/`
2. Implement feature in `include/rtpghi/`
3. Add example usage in `examples/`
4. Update documentation
5. Run tests and formatting before committing

## License

This project is licensed under the MIT License.

## References

Průša, Z., & Holighaus, N. (2017). Phase Vocoder Done Right. In *Proceedings
of the European Signal Processing Conference (EUSIPCO)* (pp. 1006-1010).
