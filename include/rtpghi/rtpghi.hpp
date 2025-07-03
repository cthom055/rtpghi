#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Ensure M_PI is available across platforms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rtpghi
{
    /// Mathematical and processing constants
    namespace constants
    {
        constexpr float PI = static_cast<float>(M_PI);
        constexpr float TWO_PI = 2.0f * PI;
        constexpr float DEFAULT_TOLERANCE = 1e-6f;
        constexpr uint32_t DEFAULT_RANDOM_SEED = 1;
        constexpr uint32_t LCG_MULTIPLIER = 1103515245;
        constexpr uint32_t LCG_INCREMENT = 12345;
        constexpr uint32_t LCG_MASK = 0x7fffffff;
    }  // namespace constants

    /// Integration method for time gradient processing
    enum class IntegrationMethod
    {
        FORWARD_EULER = 0,  ///< Simple forward Euler: phase += time_gradient
        TRAPEZOIDAL = 1     ///< Trapezoidal rule: phase += 0.5 * (prev_gradient + time_gradient)
    };

    /// Finite difference method for gradient calculation
    enum class GradientMethod
    {
        FORWARD = 0,   ///< Forward difference: grad[i] = (phase[i+1] - phase[i]) / step
        BACKWARD = 1,  ///< Backward difference: grad[i] = (phase[i] - phase[i-1]) / step
        CENTRAL = 2    ///< Central difference: grad[i] = (phase[i+1] - phase[i-1]) / (2*step)
    };

    /// Input data for gradient calculation
    struct GradientInput
    {
        const float* phases;    ///< Phase data [size]
        size_t size;            ///< Size of phase array
        float step_size;        ///< Step size (time step or frequency bin spacing)
        GradientMethod method;  ///< Finite difference method to use

        /// Validate input data
        bool is_valid() const noexcept
        {
            return phases && size > 0 && step_size > 0;
        }
    };

    /// Principal argument function for phase unwrapping
    inline float princarg(float x) noexcept
    {
        return x - constants::TWO_PI * std::round(x / constants::TWO_PI);
    }

    /// Heap entry for magnitude-priority processing
    struct HeapEntry
    {
        size_t bin_index;  ///< FFT bin index or offset index for frequency propagation
        float magnitude;   ///< Magnitude value for priority ordering
    };

    /// Max-heap implementation for magnitude-priority processing
    /// Provides efficient priority queue operations for RTPGHI algorithm
    class Heap
    {
    private:
        std::vector<HeapEntry> m_storage;  ///< Heap storage
        size_t m_size;                     ///< Current number of elements

        /// Bubble element up to maintain max-heap property
        void bubble_up(size_t idx) noexcept
        {
            while (idx > 0)
            {
                size_t parent = (idx - 1) / 2;
                if (m_storage[parent].magnitude >= m_storage[idx].magnitude)
                {
                    break;
                }
                std::swap(m_storage[parent], m_storage[idx]);
                idx = parent;
            }
        }

        /// Bubble element down to maintain max-heap property
        void bubble_down(size_t idx) noexcept
        {
            while (2 * idx + 1 < m_size)
            {
                size_t child = 2 * idx + 1;
                if (child + 1 < m_size && m_storage[child + 1].magnitude > m_storage[child].magnitude)
                {
                    child++;
                }
                if (m_storage[idx].magnitude >= m_storage[child].magnitude)
                {
                    break;
                }
                std::swap(m_storage[idx], m_storage[child]);
                idx = child;
            }
        }

    public:
        /// Constructor with capacity pre-allocation
        explicit Heap(size_t capacity = 0) : m_storage(capacity), m_size(0) {}

        /// Insert element into heap
        void insert(size_t bin_index, float magnitude) noexcept
        {
            if (m_size < m_storage.size())
            {
                m_storage[m_size] = { bin_index, magnitude };
                bubble_up(m_size);
                m_size++;
            }
        }

        /// Extract maximum element from heap
        HeapEntry extract_max() noexcept
        {
            HeapEntry result = m_storage[0];
            m_storage[0] = m_storage[--m_size];
            bubble_down(0);
            return result;
        }

        /// Check if heap is empty
        bool empty() const noexcept
        {
            return m_size == 0;
        }

        /// Clear heap (reset size to 0)
        void clear() noexcept
        {
            m_size = 0;
        }

        /// Get current size
        size_t size() const noexcept
        {
            return m_size;
        }

        /// Get capacity
        size_t capacity() const noexcept
        {
            return m_storage.size();
        }
    };

    /// Input frame data for RTPGHI processing
    struct FrameInput
    {
        const float* magnitudes;           ///< Current frame magnitudes [fft_bins]
        const float* previous_phases;      ///< Previous frame phases [fft_bins]
        const float* time_gradients;       ///< Current time phase gradients [fft_bins]
        const float* freq_gradients;       ///< Current frequency phase gradients [fft_bins]
        const float* prev_time_gradients;  ///< Previous time gradients [fft_bins] (nullptr for forward Euler)
        size_t fft_bins;                   ///< Number of FFT bins (fft_size/2 + 1)

        /// Validate input data pointers and size
        bool is_valid() const noexcept
        {
            return magnitudes && previous_phases && time_gradients && freq_gradients && fft_bins > 0;
        }
    };

    /// Output frame data for RTPGHI processing
    struct FrameOutput
    {
        float* magnitudes;  ///< Output magnitudes [fft_bins]
        float* phases;      ///< Output propagated phases [fft_bins]
        size_t fft_bins;    ///< Number of FFT bins (must match input)

        /// Validate output data pointers and size
        bool is_valid() const noexcept
        {
            return magnitudes && phases && fft_bins > 0;
        }
    };

    enum class ErrorCode
    {
        OK = 0,
        INVALID_INPUT,
        INVALID_OUTPUT,
        SIZE_MISMATCH
    };

    struct ProcessorConfig
    {
        const size_t fft_bins;                       ///< Number of FFT bins to process
        const float tolerance;                       ///< Relative tolerance for computing significant bins
        const uint32_t initial_random_seed;          ///< Initial seed for random phase generation
        const IntegrationMethod integration_method;  ///< Integration method for time gradients

        ProcessorConfig(size_t bins,
                        float tol = constants::DEFAULT_TOLERANCE,
                        uint32_t seed = constants::DEFAULT_RANDOM_SEED,
                        IntegrationMethod method = IntegrationMethod::FORWARD_EULER) :
            fft_bins(bins),
            tolerance(tol),
            initial_random_seed(seed),
            integration_method(method)
        {
            if (fft_bins == 0)
            {
                throw std::invalid_argument("FFT bins must be greater than 0");
            }
            if (tolerance <= 0.0f)
            {
                throw std::invalid_argument("Tolerance must be positive");
            }
        }
    };

    /// Calculate required heap storage size for given FFT bins
    inline size_t calculate_heap_size(size_t fft_bins) noexcept
    {
        return 2 * fft_bins;  // Time propagation + frequency propagation
    }

    /// Boolean array wrapper to avoid std::vector<bool> specialization issues
    class BoolArray
    {
    private:
        std::vector<char> m_storage;

    public:
        explicit BoolArray(size_t size = 0) : m_storage(size, 0) {}

        void resize(size_t size)
        {
            m_storage.resize(size, 0);
        }
        size_t size() const noexcept
        {
            return m_storage.size();
        }

        bool operator[](size_t index) const noexcept
        {
            return m_storage[index] != 0;
        }
        void set(size_t index, bool value) noexcept
        {
            if (index < m_storage.size())
            {
                m_storage[index] = value ? 1 : 0;
            }
        }

        void fill(bool value) noexcept
        {
            std::fill(m_storage.begin(), m_storage.end(), value ? 1 : 0);
        }

        char* data() noexcept
        {
            return m_storage.data();
        }
        const char* data() const noexcept
        {
            return m_storage.data();
        }
    };

    /// RTPGHI Processor with internal workspace
    /// Encapsulates the phase propagation algorithm and mutable state
    class Processor
    {
    private:
        const ProcessorConfig m_config;  ///< Immutable configuration
        Heap m_heap;                     ///< Priority heap for magnitude-ordered processing
        BoolArray m_significant_bins;    ///< Significant bin flags
        BoolArray m_done_mask;           ///< Processing completion mask
        uint32_t m_current_random_seed;  ///< Current random seed (mutable)

        /// Reset workspace for new frame processing
        void reset_workspace() noexcept
        {
            m_heap.clear();
            m_significant_bins.fill(false);
            m_done_mask.fill(false);
        }

    public:
        /// Construct processor with given configuration
        explicit Processor(const ProcessorConfig& config) :
            m_config(config),
            m_heap(calculate_heap_size(config.fft_bins)),
            m_significant_bins(config.fft_bins),
            m_done_mask(config.fft_bins),
            m_current_random_seed(config.initial_random_seed)
        {}

        /// Get configuration
        const ProcessorConfig& config() const noexcept
        {
            return m_config;
        }

        /// Process a frame using Real-Time Phase Gradient Heap Integration
        ErrorCode process(const FrameInput& input, FrameOutput& output);
    };

    inline ErrorCode Processor::process(const FrameInput& input, FrameOutput& output)
    {
        // Step 1: Input Validation
        if (!input.is_valid())
        {
            return ErrorCode::INVALID_INPUT;
        }
        if (!output.is_valid())
        {
            return ErrorCode::INVALID_OUTPUT;
        }
        if (input.fft_bins != output.fft_bins || input.fft_bins != m_config.fft_bins)
        {
            return ErrorCode::SIZE_MISMATCH;
        }

        // Validate integration method requirements
        if (m_config.integration_method == IntegrationMethod::TRAPEZOIDAL && !input.prev_time_gradients)
        {
            return ErrorCode::INVALID_INPUT;
        }

        const size_t fft_bins = input.fft_bins;
        const float tolerance = m_config.tolerance;
        uint32_t random_seed = m_current_random_seed;

        // Step 2: Reset and prepare workspace
        reset_workspace();

        // Step 3: Tolerance-based bin classification
        // Find maximum magnitude
        float max_magnitude = input.magnitudes[0];
        for (size_t m = 1; m < fft_bins; ++m)
        {
            if (input.magnitudes[m] > max_magnitude)
            {
                max_magnitude = input.magnitudes[m];
            }
        }

        // Calculate absolute tolerance and classify bins
        float abs_tolerance = tolerance * max_magnitude;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            m_significant_bins.set(m, input.magnitudes[m] > abs_tolerance);
        }

        // Step 4: Initialize heap and processing state
        size_t remaining_bins = 0;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            if (m_significant_bins[m])
            {
                remaining_bins++;
                m_heap.insert(m, input.magnitudes[m]);
            }
        }

        // Step 5: Heap-based phase propagation
        while (!m_heap.empty() && remaining_bins > 0)
        {
            HeapEntry current = m_heap.extract_max();
            size_t m = current.bin_index;

            // TIME PROPAGATION: From previous frame to current
            if (m < fft_bins && !m_done_mask[m] && m_significant_bins[m])
            {
                // Time integration (method depends on processor configuration)
                if (m_config.integration_method == IntegrationMethod::TRAPEZOIDAL)
                {
                    float gradient_avg = 0.5f * (input.prev_time_gradients[m] + input.time_gradients[m]);
                    output.phases[m] = input.previous_phases[m] + gradient_avg;
                }
                else
                {
                    output.phases[m] = input.previous_phases[m] + input.time_gradients[m];
                }
                m_done_mask.set(m, true);
                remaining_bins--;

                // Add to heap for frequency propagation (offset by fft_bins)
                m_heap.insert(m + fft_bins, input.magnitudes[m]);
            }
            // FREQUENCY PROPAGATION: Within current frame
            else if (current.bin_index >= fft_bins)
            {
                size_t src_bin = current.bin_index - fft_bins;

                if (m_done_mask[src_bin])
                {
                    // Propagate UP in frequency (m+1)
                    if (src_bin + 1 < fft_bins && !m_done_mask[src_bin + 1] && m_significant_bins[src_bin + 1])
                    {
                        float freq_grad_avg =
                            (input.freq_gradients[src_bin] + input.freq_gradients[src_bin + 1]) * 0.5f;
                        output.phases[src_bin + 1] = output.phases[src_bin] + freq_grad_avg;
                        m_done_mask.set(src_bin + 1, true);
                        remaining_bins--;
                        m_heap.insert((src_bin + 1) + fft_bins, input.magnitudes[src_bin + 1]);
                    }

                    // Propagate DOWN in frequency (m-1)
                    if (src_bin > 0 && !m_done_mask[src_bin - 1] && m_significant_bins[src_bin - 1])
                    {
                        float freq_grad_avg =
                            (input.freq_gradients[src_bin] + input.freq_gradients[src_bin - 1]) * 0.5f;
                        output.phases[src_bin - 1] = output.phases[src_bin] - freq_grad_avg;
                        m_done_mask.set(src_bin - 1, true);
                        remaining_bins--;
                        m_heap.insert((src_bin - 1) + fft_bins, input.magnitudes[src_bin - 1]);
                    }
                }
            }
        }

        // Step 6: Handle low-magnitude bins with random phase
        for (size_t m = 0; m < fft_bins; ++m)
        {
            if (!m_done_mask[m])
            {
                random_seed =
                    (random_seed * constants::LCG_MULTIPLIER + constants::LCG_INCREMENT) & constants::LCG_MASK;
                output.phases[m] = (static_cast<float>(random_seed) / static_cast<float>(constants::LCG_MASK)) * constants::TWO_PI;
            }
        }

        // Step 7: Copy magnitudes and apply phase unwrapping
        for (size_t m = 0; m < fft_bins; ++m)
        {
            output.magnitudes[m] = input.magnitudes[m];
            output.phases[m] = princarg(output.phases[m]);
        }

        // Update processor's random seed for next call
        m_current_random_seed = random_seed;

        return ErrorCode::OK;
    }

    /// Calculate phase gradients using finite differences
    ///
    /// Computes discrete derivatives of phase data using forward, backward, or central differences.
    /// Handles boundary conditions appropriately for each method.
    ///
    /// @param input Input phase data and gradient calculation parameters
    /// @param output_gradients Output buffer for calculated gradients [input.size]
    /// @return Error code indicating success or failure
    inline ErrorCode calculate_gradients(const GradientInput& input, float* output_gradients)
    {
        if (!input.is_valid() || !output_gradients)
        {
            return ErrorCode::INVALID_INPUT;
        }

        const float* phases = input.phases;
        const size_t size = input.size;
        const float step = input.step_size;

        switch (input.method)
        {
        case GradientMethod::FORWARD: {
            // Forward difference: grad[i] = (phase[i+1] - phase[i]) / step
            for (size_t i = 0; i < size; ++i)
            {
                if (i + 1 < size)
                {
                    // Use phase unwrapping for difference calculation
                    float phase_diff = princarg(phases[i + 1] - phases[i]);
                    output_gradients[i] = phase_diff / step;
                }
                else
                {
                    // Boundary condition: use backward difference for last point
                    float phase_diff = princarg(phases[i] - phases[i - 1]);
                    output_gradients[i] = phase_diff / step;
                }
            }
            break;
        }

        case GradientMethod::BACKWARD: {
            // Backward difference: grad[i] = (phase[i] - phase[i-1]) / step
            for (size_t i = 0; i < size; ++i)
            {
                if (i > 0)
                {
                    float phase_diff = princarg(phases[i] - phases[i - 1]);
                    output_gradients[i] = phase_diff / step;
                }
                else
                {
                    // Boundary condition: use forward difference for first point
                    float phase_diff = princarg(phases[i + 1] - phases[i]);
                    output_gradients[i] = phase_diff / step;
                }
            }
            break;
        }

        case GradientMethod::CENTRAL: {
            // Central difference: grad[i] = (phase[i+1] - phase[i-1]) / (2*step)
            for (size_t i = 0; i < size; ++i)
            {
                if (i > 0 && i + 1 < size)
                {
                    float phase_diff = princarg(phases[i + 1] - phases[i - 1]);
                    output_gradients[i] = phase_diff / (2.0f * step);
                }
                else if (i == 0)
                {
                    // Boundary condition: use forward difference for first point
                    float phase_diff = princarg(phases[i + 1] - phases[i]);
                    output_gradients[i] = phase_diff / step;
                }
                else  // i == size - 1
                {
                    // Boundary condition: use backward difference for last point
                    float phase_diff = princarg(phases[i] - phases[i - 1]);
                    output_gradients[i] = phase_diff / step;
                }
            }
            break;
        }

        default:
            return ErrorCode::INVALID_INPUT;
        }

        return ErrorCode::OK;
    }

    /// Calculate time gradients from consecutive phase frames
    ///
    /// Convenience function for calculating time derivatives between phase frames.
    ///
    /// @param prev_phases Previous frame phases [fft_bins]
    /// @param curr_phases Current frame phases [fft_bins]
    /// @param fft_bins Number of FFT bins
    /// @param time_step Time step between frames (typically hop_size / sample_rate)
    /// @param method Finite difference method
    /// @param output_gradients Output time gradients [fft_bins]
    /// @return Error code indicating success or failure
    inline ErrorCode calculate_time_gradients(const float* prev_phases,
                                              const float* curr_phases,
                                              size_t fft_bins,
                                              float time_step,
                                              GradientMethod /* method */,
                                              float* output_gradients)
    {
        if (!prev_phases || !curr_phases || !output_gradients || fft_bins == 0 || time_step <= 0)
        {
            return ErrorCode::INVALID_INPUT;
        }

        // For time gradients, we typically use forward/backward difference between frames
        for (size_t i = 0; i < fft_bins; ++i)
        {
            float phase_diff = princarg(curr_phases[i] - prev_phases[i]);
            output_gradients[i] = phase_diff / time_step;
        }

        return ErrorCode::OK;
    }

    /// Calculate frequency gradients within a single frame
    ///
    /// Convenience function for calculating frequency derivatives across FFT bins.
    ///
    /// @param phases Phase data for current frame [fft_bins]
    /// @param fft_bins Number of FFT bins
    /// @param freq_step Frequency step between bins (typically sample_rate / fft_size)
    /// @param method Finite difference method
    /// @param output_gradients Output frequency gradients [fft_bins]
    /// @return Error code indicating success or failure
    inline ErrorCode calculate_freq_gradients(const float* phases,
                                              size_t fft_bins,
                                              float freq_step,
                                              GradientMethod method,
                                              float* output_gradients)
    {
        if (!phases || !output_gradients || fft_bins == 0 || freq_step <= 0)
        {
            return ErrorCode::INVALID_INPUT;
        }

        GradientInput input { phases, fft_bins, freq_step, method };
        return calculate_gradients(input, output_gradients);
    }

}  // namespace rtpghi
