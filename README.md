# RTPGHI - Real-Time Phase Gradient Heap Integration

A modern header-only C++ library implementing the RTPGHI algorithm for high-quality phase vocoding and time-stretching with enhanced gradient calculation APIs.

## Quick Start

```cpp
#include <rtpghi/rtpghi.hpp>
#include <complex>

// Configure processor
const size_t fft_bins = 513;  // 1024 FFT -> 513 bins
rtpghi::ProcessorConfig config(fft_bins);
rtpghi::Processor processor(config);

// Create complex spectrum from magnitude and phase data
std::vector<std::vector<std::complex<float>>> spectra(2);
spectra[0].resize(fft_bins);  // Previous frame
spectra[1].resize(fft_bins);  // Current frame

for (size_t i = 0; i < fft_bins; ++i) {
    spectra[0][i] = std::polar(magnitudes[i], prev_phases[i]);
    spectra[1][i] = std::polar(magnitudes[i], curr_phases[i]);
}

// Calculate gradients using enhanced API
std::vector<float> time_gradients(fft_bins);
std::vector<float> freq_gradients(fft_bins);

rtpghi::GradientOutput gradient_output {
    time_gradients.data(),
    freq_gradients.data(),
    fft_bins,  // time_frames
    fft_bins,  // freq_frames
    fft_bins
};

auto gradient_result = rtpghi::calculate_spectrum_gradients(
    spectra, time_step, freq_step, gradient_output,
    rtpghi::GradientMethod::FORWARD,
    rtpghi::GradientMethod::CENTRAL
);

if (gradient_result != rtpghi::ErrorCode::OK) {
    // Handle gradient calculation error
    return;
}

// Process frame with RTPGHI
rtpghi::FrameInput input {
    magnitudes.data(),
    prev_phases.data(),
    time_gradients.data(),
    freq_gradients.data(),
    nullptr,  // Previous time gradients (null for Forward Euler)
    fft_bins
};

rtpghi::FrameOutput output {
    out_mags.data(), out_phases.data(), fft_bins
};

auto process_result = processor.process(input, output);
if (process_result == rtpghi::ErrorCode::OK) {
    // Use output phases for inverse FFT
}
```

## Enhanced Gradient API

### Modern Matrix-Based Interface

The enhanced gradient API provides type-safe matrix views for convenient 2D data access:

```cpp
// Using GradientResult with managed storage
rtpghi::GradientResult result;
result.time_frames = num_frames;
result.freq_frames = num_frames; 
result.fft_bins = fft_bins;
result.time_data.resize(num_frames * fft_bins);
result.freq_data.resize(num_frames * fft_bins);

// Get matrix views for convenient 2D access
auto time_matrix = result.time_gradients();  // GradientMatrix view
auto freq_matrix = result.freq_gradients();  // GradientMatrix view

// Access data using familiar matrix notation
float time_grad = time_matrix(frame, bin);     // operator() access
float* freq_row = freq_matrix[frame];          // operator[] row access

// Iterate over data efficiently
for (size_t frame = 0; frame < time_matrix.rows(); ++frame) {
    float* row = time_matrix[frame];
    for (size_t bin = 0; bin < time_matrix.cols(); ++bin) {
        // Process row[bin]
    }
}
```

### Complex Spectrum Input

The enhanced API accepts complex spectrum data directly, eliminating manual phase extraction:

```cpp
// Direct complex spectrum processing
std::vector<std::vector<std::complex<float>>> complex_frames = {
    previous_complex_frame,  // std::vector<std::complex<float>>
    current_complex_frame    // std::vector<std::complex<float>>
};

// Gradients calculated automatically from complex data
auto result = rtpghi::calculate_spectrum_gradients(
    complex_frames, time_step, freq_step, output,
    rtpghi::GradientMethod::FORWARD,    // time gradient method
    rtpghi::GradientMethod::CENTRAL     // frequency gradient method
);
```

### Template-Based Flexibility

The API supports various complex number containers:

```cpp
// Raw pointer interface for C interop
const std::complex<float>* const* spectrum_ptrs = /* ... */;
rtpghi::calculate_spectrum_gradients(
    spectrum_ptrs, num_frames, fft_bins, 
    time_step, freq_step, output
);

// Template-based input for custom containers
rtpghi::ComplexSpectrumGradientInput<std::complex<double>> input {
    spectrum_data, num_frames, fft_bins,
    time_step, freq_step,
    rtpghi::GradientMethod::FORWARD,
    rtpghi::GradientMethod::CENTRAL
};
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

## Algorithm Overview

The RTPGHI algorithm reconstructs missing phase information through:

1. **Magnitude Classification** - Bins classified by significance threshold
2. **Heap-Based Propagation** - Phase estimation in magnitude-priority order  
3. **Time Integration** - Numerical integration of temporal phase evolution
4. **Frequency Propagation** - Local phase consistency within frames
5. **Random Assignment** - Random phases for insignificant bins

### Performance Characteristics

- **Time Complexity**: O(B log B) where B = significant bins
- **Space Complexity**: O(N) where N = total FFT bins
- **Real-time Safe**: Zero allocation during processing
- **Cache Friendly**: Workspace pre-allocated and reused

## Building

### Requirements
- CMake 3.14+
- C++11 compatible compiler
- Optional: Catch2 for testing (auto-downloaded)

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

- **Header-only** - Single include, no installation required
- **Modern C++** - [[nodiscard]], constexpr, type safety
- **Zero allocation** - Real-time friendly processing
- **RAII design** - Automatic resource management
- **Enhanced APIs** - Matrix views, complex input, template support
- **Comprehensive testing** - 154K+ test assertions covering edge cases
- **Cross-platform** - Windows, macOS, Linux support

## API Reference

### Core Classes
- `rtpghi::ProcessorConfig` - Immutable processor configuration
- `rtpghi::Processor` - Main RTPGHI algorithm implementation
- `rtpghi::GradientMatrix` - Lightweight 2D matrix view for gradient data
- `rtpghi::GradientResult` - Managed storage with matrix view access

### Input/Output Structures  
- `rtpghi::FrameInput` - RTPGHI processing input
- `rtpghi::FrameOutput` - RTPGHI processing output
- `rtpghi::GradientOutput` - Legacy gradient output structure
- `rtpghi::ComplexSpectrumGradientInput<T>` - Template-based complex input

### Gradient Methods
- `rtpghi::GradientMethod::FORWARD` - Forward finite difference
- `rtpghi::GradientMethod::BACKWARD` - Backward finite difference  
- `rtpghi::GradientMethod::CENTRAL` - Central finite difference (recommended)

### Integration Methods
- `rtpghi::IntegrationMethod::FORWARD_EULER` - Simple, fast (default)
- `rtpghi::IntegrationMethod::TRAPEZOIDAL` - Higher accuracy, smoother results

### Error Handling
All functions return `rtpghi::ErrorCode` with `[[nodiscard]]` attribute:
- `rtpghi::ErrorCode::OK` - Success
- `rtpghi::ErrorCode::INVALID_INPUT` - Invalid input parameters
- `rtpghi::ErrorCode::SIZE_MISMATCH` - Dimension mismatch
- `rtpghi::ErrorCode::INSUFFICIENT_MEMORY` - Memory allocation failure

## Examples

See `examples/` directory for complete usage demonstrations:
- `basic_example.cpp` - Simple single-frame processing
- `complete_workflow.cpp` - Multi-frame real-time processing with performance analysis

## License

MIT License

## Citation

```
Z. Průša and N. Holighaus. Phase vocoder done right. In Proc. Eur. Signal Process. Conf. 
EUSIPCO, pages 1006–1010, Kos island, Greece, Aug. 2017.
```

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`ctest --test-dir build`)
- Code follows existing style conventions  
- New features include comprehensive tests
- Documentation updated for API changes