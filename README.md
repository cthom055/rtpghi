# RTPGHI - Real-Time Phase Gradient Heap Integration

A header-only C++ library implementing the RTPGHI algorithm for high-quality phase vocoding and time-stretching.

## Quick Start

```cpp
#include <rtpghi/rtpghi.hpp>

// Configure processor
const size_t fft_bins = 513;  // 1024 FFT -> 513 bins
rtpghi::ProcessorConfig config(fft_bins);
rtpghi::Processor processor(config);

// Prepare input data
rtpghi::FrameInput input {
    magnitudes.data(),      // Current frame magnitudes
    prev_phases.data(),     // Previous frame phases  
    time_gradients.data(),  // Time gradients
    freq_gradients.data(),  // Frequency gradients
    nullptr,                // Previous time gradients (null for Forward Euler)
    fft_bins
};

rtpghi::FrameOutput output {
    out_mags.data(), out_phases.data(), fft_bins
};

// Process frame
if (processor.process(input, output) == rtpghi::ErrorCode::OK) {
    // Use output phases for inverse FFT
}
```

## Integration Methods

```cpp
// Forward Euler (default, faster)
rtpghi::ProcessorConfig euler_config(fft_bins);

// Trapezoidal (higher accuracy, requires previous gradients)
rtpghi::ProcessorConfig trap_config(fft_bins, 1e-6f, 12345, 
                                   rtpghi::IntegrationMethod::TRAPEZOIDAL);

// For trapezoidal, provide previous time gradients
rtpghi::FrameInput trap_input {
    magnitudes.data(), prev_phases.data(),
    time_gradients.data(), freq_gradients.data(),
    prev_time_gradients.data(),  // Required for trapezoidal
    fft_bins
};
```

## Gradient Calculation

```cpp
// Calculate time gradients between frames
rtpghi::calculate_time_gradients(
    prev_phases.data(), curr_phases.data(), fft_bins,
    time_step, rtpghi::GradientMethod::FORWARD, 
    time_gradients.data()
);

// Calculate frequency gradients within frame
rtpghi::calculate_freq_gradients(
    curr_phases.data(), fft_bins, freq_step,
    rtpghi::GradientMethod::CENTRAL, 
    freq_gradients.data()
);
```

## Building

### Requirements
- CMake 3.14+
- C++11 compatible compiler

### Build
```bash
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build

# Run examples
./build/example
./build/complete_workflow
```

### CMake Integration
```cmake
add_subdirectory(rtpghi)
target_link_libraries(your_target rtpghi)
```

## Key Features

- **Header-only** - Single include file
- **Zero allocation** - Real-time friendly processing
- **RAII design** - Automatic memory management
- **Multiple integration methods** - Forward Euler and Trapezoidal
- **Comprehensive validation** - Input validation and error handling

## API Reference

### Core Classes
- `rtpghi::ProcessorConfig` - Immutable processor configuration
- `rtpghi::Processor` - Main processing class with internal workspace
- `rtpghi::FrameInput` - Input frame data structure
- `rtpghi::FrameOutput` - Output frame data structure

### Gradient Methods
- `rtpghi::GradientMethod::FORWARD` - Forward difference
- `rtpghi::GradientMethod::BACKWARD` - Backward difference  
- `rtpghi::GradientMethod::CENTRAL` - Central difference (recommended)

### Integration Methods
- `rtpghi::IntegrationMethod::FORWARD_EULER` - Simple, fast (default)
- `rtpghi::IntegrationMethod::TRAPEZOIDAL` - Higher accuracy, requires previous gradients

## License

MIT License

## Citation

```
Z. Průša and N. Holighaus. Phase vocoder done right. In Proc. Eur. Signal Process. Conf. 
EUSIPCO, pages 1006–1010, Kos island, Greece, Aug. 2017.
```