# RTPGHI C++ Style Guide

This document outlines the coding conventions and style guidelines for the RTPGHI library.

## Naming Conventions

### Member Variables
- **Private/protected member variables** use `m_` prefix followed by snake_case
  ```cpp
  class Example {
  private:
      int m_sample_count;
      float m_frequency;
      std::vector<float> m_buffer;
  };
  ```
- **Rationale**: The single underscore suffix (`_`) is reserved for standard library implementations and should be avoided in user code

### Functions and Methods
- Use **snake_case** for all functions and methods
  ```cpp
  void process_frame();
  float calculate_magnitude();
  ```

### Classes and Structs
- Use **PascalCase** for class and struct names
  ```cpp
  class Processor;
  struct ProcessorConfig;
  ```

### Constants
- Use **UPPER_SNAKE_CASE** for compile-time constants
  ```cpp
  constexpr float DEFAULT_TOLERANCE = 1e-6f;
  constexpr size_t MAX_FFT_SIZE = 8192;
  ```

### Namespaces
- Use **lowercase** for namespace names
  ```cpp
  namespace rtpghi {
  namespace constants {
  ```

## Code Organization

### Header Files
- Use `#pragma once` instead of include guards
- Order includes: standard library, third-party libraries, project headers
- Prefer forward declarations when possible

### Classes
- Order of declarations: public, protected, private
- Within each section: types, constructors, methods, members
- Use explicit constructors for single-argument constructors

### Functions
- Keep functions short and focused (single responsibility)
- Use `noexcept` for functions that don't throw
- Mark getter methods as `const`

## Formatting

### Indentation
- Use **4 spaces** for indentation (no tabs)
- Brace style: opening braces on new line (Allman/Microsoft style)
  ```cpp
  if (condition)
  {
      // code
  }
  ```

### Line Length
- Maximum **120 characters** per line
- Break long lines at logical points

### Braces
- **Always use braces** for control structures (if/else/for/while)
  ```cpp
  // Good
  if (x > 0)
  {
      process();
  }
  
  // Bad
  if (x > 0)
      process();
  ```

## Best Practices

### RAII (Resource Acquisition Is Initialization)
- Use RAII for resource management
- Prefer `std::vector` and `std::unique_ptr` over raw arrays
- Avoid manual memory management

### Const Correctness
- Use `const` wherever possible
- Mark methods that don't modify state as `const`
- Use `constexpr` for compile-time constants

### Error Handling
- Use enum classes for error codes
- Validate inputs at API boundaries
- Throw exceptions in constructors for invalid configurations

### Performance
- Design for zero-allocation in real-time paths
- Use `noexcept` to enable optimizations
- Prefer stack allocation for small, fixed-size data

## Documentation

### Comments
- Use `///` for Doxygen-style documentation
- Document public APIs thoroughly
- Keep implementation comments minimal and meaningful

### File Headers
- Include copyright notice and license information
- Brief description of file purpose

## Example

```cpp
#pragma once
#include <vector>
#include <cstdint>

namespace rtpghi
{
    /// Configuration for audio processor
    struct ProcessorConfig
    {
        const size_t fft_bins;        ///< Number of FFT bins
        const float tolerance;        ///< Processing tolerance
        
        ProcessorConfig(size_t bins, float tol)
            : fft_bins(bins)
            , tolerance(tol)
        {
            if (fft_bins == 0)
            {
                throw std::invalid_argument("FFT bins must be positive");
            }
        }
    };
    
    /// Real-time audio processor
    class Processor
    {
    private:
        const ProcessorConfig m_config;
        std::vector<float> m_buffer;
        uint32_t m_frame_count;
        
    public:
        explicit Processor(const ProcessorConfig& config)
            : m_config(config)
            , m_buffer(config.fft_bins)
            , m_frame_count(0)
        {
        }
        
        void process(const float* input, float* output) noexcept;
        
        size_t get_frame_count() const noexcept 
        { 
            return m_frame_count; 
        }
    };
}
```

## Tools

- **clang-format**: Automated formatting (see .clang-format)
- **pre-commit**: Git hooks for consistent formatting
- Run `clang-format -i` before committing