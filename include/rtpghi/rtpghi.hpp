#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace rtpghi
{
    /// Mathematical and processing constants
    namespace constants
    {
        constexpr float PI = 3.14159265358979323846f;
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

    /// Principal argument function for phase unwrapping
    inline float princarg(float x) noexcept
    {
        return x - constants::TWO_PI * std::round(x / constants::TWO_PI);
    }

    /// Robust phase wrapping implementation (for high-precision applications)
    template <typename T>
    inline bool approximatelyEqual(T a, T b, T epsilon = std::numeric_limits<T>::epsilon()) noexcept
    {
        return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
    }

    template <typename T> inline T wrappedModulo(T x, T y) noexcept
    {
        static_assert(!std::numeric_limits<T>::is_exact, "wrappedModulo: floating-point type expected");

        if (0. == y)
        {
            return x;
        }

        double m = x - y * floor(x / y);

        // handle boundary cases resulted from floating-point cut off:
        if (y > 0)  // modulo range: [0..y)
        {
            if (m >= y)  // Mod(-1e-16, 360.): m= 360.
            {
                return 0;
            }

            if (m < 0)
            {
                if (approximatelyEqual(static_cast<T>(y + m), y))
                {
                    return 0;  // just in case...
                }
                else
                {
                    return static_cast<T>(y + m);  // Mod(106.81415022205296, _TWO_PI): m= -1.421e-14
                }
            }
        }
        else  // modulo range: (y..0]
        {
            if (m <= y)  // Mod(1e-16, -360.): m= -360.
            {
                return 0;
            }

            if (m > 0)
            {
                if (approximatelyEqual(static_cast<T>(y + m), y))
                {
                    return 0;  // just in case...
                }
                else
                {
                    return static_cast<T>(y + m);  // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
                }
            }
        }

        return static_cast<T>(m);
    }

    /// Robust phase wrapping to [-π, π] range
    template <typename T> inline T wrapPi(T angle) noexcept
    {
        const auto result = wrappedModulo(angle + constants::PI, constants::TWO_PI) - constants::PI;
        return static_cast<T>(result);
    }

    /// Calculate unwrapped phase difference for gradient computation
    /// Returns the smallest angular distance between two phases
    inline float phaseDiff(float phase1, float phase0) noexcept
    {
        // Use princarg to get the wrapped difference in [-π, π]
        return princarg(phase1 - phase0);
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
        const float* current_magnitudes;   ///< Current frame magnitudes s(m,n) [fft_bins]
        const float* previous_magnitudes;  ///< Previous frame magnitudes s(m,n-1) [fft_bins]
        const float* previous_phases;      ///< Previous frame phases φs(m,n-1) [fft_bins]
        const float* time_gradients;       ///< Current time phase gradients (∆tφa)(m,n) [fft_bins]
        const float* freq_gradients;       ///< Current frequency phase gradients (∆fφa)(m,n) [fft_bins]
        const float*
            prev_time_gradients;    ///< Previous time gradients (∆tφa)(m,n-1) [fft_bins] (nullptr for forward Euler)
        float synthesis_time_step;  ///< Synthesis time step as
        float synthesis_freq_step;  ///< Synthesis frequency step bs
        size_t fft_bins;            ///< Number of FFT bins (fft_size/2 + 1)

        /// Validate input data pointers and size
        bool is_valid() const noexcept
        {
            return current_magnitudes && previous_magnitudes && previous_phases && time_gradients && freq_gradients &&
                   fft_bins > 0 && synthesis_time_step > 0.0f && synthesis_freq_step > 0.0f;
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
        const float synthesis_time_step;             ///< Synthesis time step as for scaling time integration
        const float synthesis_freq_step;             ///< Synthesis frequency step bs for scaling frequency integration

        ProcessorConfig(size_t bins,
                        float time_step = 1.0f,
                        float freq_step = 1.0f,
                        float tol = constants::DEFAULT_TOLERANCE,
                        uint32_t seed = constants::DEFAULT_RANDOM_SEED,
                        IntegrationMethod method = IntegrationMethod::FORWARD_EULER) :
            fft_bins(bins),
            tolerance(tol),
            initial_random_seed(seed),
            integration_method(method),
            synthesis_time_step(time_step),
            synthesis_freq_step(freq_step)
        {
            if (fft_bins == 0)
            {
                throw std::invalid_argument("FFT bins must be greater than 0");
            }
            if (tolerance <= 0.0f)
            {
                throw std::invalid_argument("Tolerance must be positive");
            }
            if (synthesis_time_step <= 0.0f)
            {
                throw std::invalid_argument("Synthesis time step must be positive");
            }
            if (synthesis_freq_step <= 0.0f)
            {
                throw std::invalid_argument("Synthesis frequency step must be positive");
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
            std::fill(m_storage.begin(), m_storage.end(), value ? static_cast<char>(1) : static_cast<char>(0));
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

    /// Real-Time Phase Gradient Heap Integration (RTPGHI) Processor
    ///
    /// The RTPGHI algorithm reconstructs missing phase information from magnitude-only spectrograms
    /// by propagating phase estimates using gradient information in a magnitude-priority order.
    ///
    /// ## Algorithm Overview
    ///
    /// RTPGHI operates by treating phase reconstruction as a constrained optimization problem:
    /// 1. **Magnitude Classification**: Frequency bins are classified by magnitude significance
    /// 2. **Heap-Based Propagation**: Phases are estimated in magnitude-priority order using a max-heap
    /// 3. **Time Integration**: Time gradients from previous frames are integrated using numerical methods
    /// 4. **Frequency Propagation**: Frequency gradients within current frame guide local phase consistency
    /// 5. **Random Assignment**: Insignificant bins receive random phases to minimize artifacts
    ///
    /// ## Processing Steps
    ///
    /// For each frame, the algorithm:
    /// 1. Identifies significant bins based on magnitude threshold (tolerance parameter)
    /// 2. Adds significant bins to a max-heap ordered by magnitude
    /// 3. Processes bins in order of decreasing magnitude:
    ///    - Integrates time gradient from previous frame
    ///    - Propagates from neighboring frequency bins when possible
    /// 4. Assigns random phases to remaining insignificant bins
    /// 5. Outputs reconstructed magnitude-phase spectrum
    ///
    /// ## Integration Methods
    ///
    /// - **Forward Euler**: Simple but can accumulate errors: `phase += time_gradient * dt`
    /// - **Trapezoidal**: Higher accuracy for smooth signals: `phase += 0.5 * (prev_grad + curr_grad) * dt`
    ///
    /// ## Performance Characteristics
    ///
    /// - **Time Complexity**: O(B log B) where B is number of significant bins
    /// - **Space Complexity**: O(N) where N is total FFT bins
    /// - **Real-time Safe**: No dynamic allocation after initialization
    /// - **Cache Friendly**: Workspace pre-allocated and reused across frames
    ///
    /// @see ProcessorConfig for configuration parameters
    /// @see IntegrationMethod for available integration strategies
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
        [[nodiscard]] ErrorCode process(const FrameInput& input, FrameOutput& output);
    };

    [[nodiscard]] inline ErrorCode Processor::process(const FrameInput& input, FrameOutput& output)
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

        // Validate synthesis step parameters match
        if (std::abs(input.synthesis_time_step - m_config.synthesis_time_step) > 1e-6f ||
            std::abs(input.synthesis_freq_step - m_config.synthesis_freq_step) > 1e-6f)
        {
            return ErrorCode::INVALID_INPUT;
        }

        const size_t fft_bins = input.fft_bins;
        const float tolerance = m_config.tolerance;
        uint32_t random_seed = m_current_random_seed;

        // Step 2: Reset and prepare workspace
        reset_workspace();

        // Step 3: Tolerance-based bin classification
        // Algorithm 1 Line 179: abstol ← tol · max(s(m,n) ∪ s(m,n-1))
        // Find maximum magnitude across both current and previous frames
        float max_magnitude = 0.0f;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            max_magnitude = std::max(max_magnitude, input.current_magnitudes[m]);
            max_magnitude = std::max(max_magnitude, input.previous_magnitudes[m]);
        }

        // Calculate absolute tolerance and classify bins
        // Algorithm 1 Line 180: Create set I = {m : s(m,n) > abstol}
        float abs_tolerance = tolerance * max_magnitude;

        // DEBUG: Track bin classification in detail for first few frames
        static int debug_frame = 0;
        debug_frame++;

        size_t significant_count = 0;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            bool is_significant = input.current_magnitudes[m] > abs_tolerance;
            m_significant_bins.set(m, is_significant);
            if (is_significant)
            {
                significant_count++;
            }
        }

        if (debug_frame <= 3)
        {
            printf("=== RTPGHI Frame %d ===\n", debug_frame);
            printf("Tolerance: relative=%.1e, max_mag=%.6f, abs_tolerance=%.1e\n",
                   static_cast<double>(tolerance),
                   static_cast<double>(max_magnitude),
                   static_cast<double>(abs_tolerance));
            printf("Significant bins: %zu/%zu (%.1f%%)\n",
                   significant_count,
                   fft_bins,
                   static_cast<double>(100.0f * significant_count / fft_bins));

            // Show first 20 bins classification
            printf("First 20 bins classification: ");
            for (size_t m = 0; m < 20 && m < fft_bins; ++m)
            {
                printf("%zu:%s ", m, (input.current_magnitudes[m] > abs_tolerance) ? "SIG" : "low");
            }
            printf("\n");
        }

        // Step 4: Initialize heap and processing state
        // Algorithm 1 Line 184: Insert (m, n-1) for m ∈ I into the heap
        size_t remaining_bins = 0;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            if (m_significant_bins[m])
            {
                remaining_bins++;
                // Insert previous frame coordinates with previous frame magnitudes for priority
                m_heap.insert(m, input.previous_magnitudes[m]);

                if (debug_frame <= 3 && remaining_bins <= 20)
                {
                    printf(
                        "  Heap init: bin %zu, prev_mag=%.6f\n", m, static_cast<double>(input.previous_magnitudes[m]));
                }
            }
        }

        if (debug_frame <= 3)
        {
            printf("  Heap initialized with %zu significant bins\n", remaining_bins);

            // DEBUG: Check input data for first 3 bins
            printf("  Input previous_phases (first 3): ");
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                printf("%.6f ", static_cast<double>(input.previous_phases[i]));
            }
            printf("\n");

            printf("  Input time_gradients (first 3): ");
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                printf("%.6f ", static_cast<double>(input.time_gradients[i]));
            }
            printf("\n");

            printf("  Input freq_gradients (first 3): ");
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                printf("%.6f ", static_cast<double>(input.freq_gradients[i]));
            }
            printf("\n");

            printf("  synthesis_time_step: %.6f\n", static_cast<double>(input.synthesis_time_step));
            printf("  synthesis_freq_step: %.6f\n", static_cast<double>(input.synthesis_freq_step));
        }

        // Step 5: Heap-based phase propagation
        size_t heap_iterations = 0;
        size_t time_propagations = 0;
        size_t freq_propagations = 0;

        while (!m_heap.empty() && remaining_bins > 0)
        {
            HeapEntry current = m_heap.extract_max();
            size_t m = current.bin_index;
            heap_iterations++;

            if (debug_frame <= 3 && heap_iterations <= 30)
            {
                printf("  Heap[%zu]: extracted bin_idx=%zu, magnitude=%.6f, remaining=%zu\n",
                       heap_iterations,
                       m,
                       static_cast<double>(current.magnitude),
                       remaining_bins);
            }

            // TIME PROPAGATION: From previous frame to current
            if (m < fft_bins && !m_done_mask[m] && m_significant_bins[m])
            {
                float old_phase = input.previous_phases[m];
                float time_gradient = input.time_gradients[m];

                if (debug_frame <= 3 && (m == 0 || m == 1 || m == 2))  // Focus on first 3 bins
                {
                    printf("    → TIME PROP bin %zu: prev_phase=%.6f + (dt=%.6f * grad=%.6f) = %.6f\n",
                           m,
                           static_cast<double>(old_phase),
                           static_cast<double>(input.synthesis_time_step),
                           static_cast<double>(time_gradient),
                           static_cast<double>(old_phase + input.synthesis_time_step * time_gradient));
                }

                // Algorithm 1 Lines 216-257: Time integration with synthesis step scaling
                // φs(mh, n) ← φs(mh, n-1) + as/2 * ((∆tφa)(mh, n-1) + (∆tφa)(mh, n))
                if (m_config.integration_method == IntegrationMethod::TRAPEZOIDAL)
                {
                    float gradient_avg = 0.5f * (input.prev_time_gradients[m] + input.time_gradients[m]);
                    output.phases[m] = old_phase + (input.synthesis_time_step * gradient_avg);
                }
                else
                {
                    // Forward Euler: φs(mh, n) ← φs(mh, n-1) + as * (∆tφa)(mh, n)
                    output.phases[m] = old_phase + (input.synthesis_time_step * time_gradient);
                }

                if (debug_frame <= 3 && (m == 0 || m == 1 || m == 2))  // Focus on first 3 bins
                {
                    printf("      STORED bin %zu: output.phases[%zu] = %.6f (before wrapping)\n",
                           m,
                           m,
                           static_cast<double>(output.phases[m]));
                }

                m_done_mask.set(m, true);
                remaining_bins--;
                time_propagations++;

                // Enable frequency propagation and debug any issues
                m_heap.insert(m + fft_bins, input.current_magnitudes[m]);
            }
            // FREQUENCY PROPAGATION: Enabled
            else if (current.bin_index >= fft_bins)
            {
                size_t src_bin = current.bin_index - fft_bins;

                if (debug_frame <= 3 && heap_iterations <= 30)
                {
                    printf(
                        "    → FREQ PROP from bin %zu (src_done=%s)\n", src_bin, m_done_mask[src_bin] ? "YES" : "NO");
                }

                if (m_done_mask[src_bin])
                {
                    // Algorithm 1 Line 269: Frequency propagation with synthesis step scaling
                    // φs(mh ± 1, n) ← φs(mh, n) ± bs/2 * ((∆fφa)(mh, n) + (∆fφa)(mh ± 1, n))

                    // Propagate UP in frequency (m+1)
                    if (src_bin + 1 < fft_bins && !m_done_mask[src_bin + 1] && m_significant_bins[src_bin + 1])
                    {
                        // Regular Euler integration: use source bin gradient only (no averaging)
                        float freq_grad = input.freq_gradients[src_bin];
                        // Scale down frequency gradients for 1x reconstruction to prevent excessive phase jumps
                        // Scale down frequency gradients for 1x reconstruction (empirically determined)\n float
                        // freq_scaling = 0.0003f;  // Fixes 3.14 rad -> 0.001 rad increments\n
                        float phase_increment = input.synthesis_freq_step * freq_grad;

                        // No limiting - debug raw frequency propagation behavior

                        output.phases[src_bin + 1] = wrapPi(output.phases[src_bin] + phase_increment);
                        m_done_mask.set(src_bin + 1, true);
                        remaining_bins--;
                        freq_propagations++;
                        m_heap.insert((src_bin + 1) + fft_bins, input.current_magnitudes[src_bin + 1]);

                        if (debug_frame <= 3 && freq_propagations <= 10)
                        {
                            printf("      FREQ UP: bin %zu: src_grad=%.6f, dest_grad=%.6f, grad_used=%.6f, "
                                   "freq_step=%.6f, increment=%.6f, phase=%.4f\n",
                                   src_bin + 1,
                                   static_cast<double>(input.freq_gradients[src_bin]),
                                   static_cast<double>(input.freq_gradients[src_bin + 1]),
                                   static_cast<double>(freq_grad),
                                   static_cast<double>(input.synthesis_freq_step),
                                   static_cast<double>(phase_increment),
                                   static_cast<double>(output.phases[src_bin + 1]));
                        }
                    }

                    // Propagate DOWN in frequency (m-1)
                    if (src_bin > 0 && !m_done_mask[src_bin - 1] && m_significant_bins[src_bin - 1])
                    {
                        // Regular Euler integration: use source bin gradient only (no averaging)
                        float freq_grad = input.freq_gradients[src_bin];
                        // Scale down frequency gradients for 1x reconstruction to prevent excessive phase jumps
                        // Scale down frequency gradients for 1x reconstruction (empirically determined)\n float
                        // freq_scaling = 0.0003f;  // Fixes 3.14 rad -> 0.001 rad increments\n
                        float phase_increment = input.synthesis_freq_step * freq_grad;

                        // No limiting - debug raw frequency propagation behavior

                        output.phases[src_bin - 1] = wrapPi(output.phases[src_bin] - phase_increment);
                        m_done_mask.set(src_bin - 1, true);
                        remaining_bins--;
                        freq_propagations++;
                        m_heap.insert((src_bin - 1) + fft_bins, input.current_magnitudes[src_bin - 1]);

                        if (debug_frame <= 3 && freq_propagations <= 10)
                        {
                            printf("      FREQ DOWN: bin %zu: src_grad=%.6f, dest_grad=%.6f, grad_used=%.6f, "
                                   "freq_step=%.6f, increment=%.6f, phase=%.4f\n",
                                   src_bin - 1,
                                   static_cast<double>(input.freq_gradients[src_bin]),
                                   static_cast<double>(input.freq_gradients[src_bin - 1]),
                                   static_cast<double>(freq_grad),
                                   static_cast<double>(input.synthesis_freq_step),
                                   static_cast<double>(phase_increment),
                                   static_cast<double>(output.phases[src_bin - 1]));
                        }
                    }
                }
            }
            else
            {
                // DEBUG: Track what gets skipped and why
                if (debug_frame <= 3 && heap_iterations <= 30)
                {
                    if (m < fft_bins)
                    {
                        printf("    → SKIP TIME bin %zu: done=%s, significant=%s\n",
                               m,
                               m_done_mask[m] ? "YES" : "NO",
                               m_significant_bins[m] ? "YES" : "NO");
                    }
                    else
                    {
                        printf("    → SKIP: unknown bin_idx=%zu\n", m);
                    }
                }
            }
        }

        if (debug_frame <= 3)
        {
            printf("  Final stats: %zu heap iterations, %zu time props, %zu freq props, %zu remaining\n",
                   heap_iterations,
                   time_propagations,
                   freq_propagations,
                   remaining_bins);
        }

        // Step 6: Handle low-magnitude bins with random phase
        size_t random_phase_count = 0;
        for (size_t m = 0; m < fft_bins; ++m)
        {
            if (!m_done_mask[m])
            {
                random_seed =
                    (random_seed * constants::LCG_MULTIPLIER + constants::LCG_INCREMENT) & constants::LCG_MASK;
                output.phases[m] =
                    (static_cast<float>(random_seed) / static_cast<float>(constants::LCG_MASK)) * constants::TWO_PI;
                random_phase_count++;

                if (debug_frame <= 3 && random_phase_count <= 20)
                {
                    printf("  RANDOM bin %zu: significant=%s, magnitude=%.6f\n",
                           m,
                           m_significant_bins[m] ? "YES" : "NO",
                           static_cast<double>(input.current_magnitudes[m]));
                }
            }
        }

        if (debug_frame <= 3)
        {
            printf("  Random phases assigned to %zu bins\n", random_phase_count);
        }

        // Step 7: Copy magnitudes and apply phase unwrapping
        for (size_t m = 0; m < fft_bins; ++m)
        {
            output.magnitudes[m] = input.current_magnitudes[m];

            if (debug_frame <= 3 && (m == 0 || m == 1 || m == 2))
            {
                float before_wrap = output.phases[m];
                output.phases[m] = wrapPi(output.phases[m]);  // Use wrapPi instead of princarg
                printf("  WRAP bin %zu: %.6f → %.6f\n",
                       m,
                       static_cast<double>(before_wrap),
                       static_cast<double>(output.phases[m]));
            }
            else
            {
                output.phases[m] = wrapPi(output.phases[m]);  // Use wrapPi instead of princarg
            }
        }

        // Update processor's random seed for next call
        m_current_random_seed = random_seed;

        return ErrorCode::OK;
    }

    /// Gradient dimension for 2D gradient calculations
    enum class GradientDimension
    {
        TIME,      ///< Calculate gradients across time (between frames)
        FREQUENCY  ///< Calculate gradients across frequency (between bins)
    };

    /// Enhanced input structure for complex spectrum batch processing
    template <typename ComplexType> struct ComplexSpectrumGradientInput
    {
        const ComplexType* const* spectra;  ///< Array of pointers to complex spectra [num_frames][fft_bins]
        size_t num_frames;                  ///< Number of time frames
        size_t fft_bins;                    ///< Number of frequency bins per frame
        float time_step;                    ///< Time between frames (hop_size / sample_rate)
        float freq_step;                    ///< Frequency step (sample_rate / fft_size)
        GradientMethod time_method;         ///< Method for time gradients (default: CENTRAL)
        GradientMethod freq_method;         ///< Method for frequency gradients (default: CENTRAL)

        /// Validate input data
        bool is_valid() const noexcept
        {
            return spectra && num_frames > 0 && fft_bins > 0 && time_step > 0.0f && freq_step > 0.0f;
        }
    };

    /// 2D matrix view for gradient data - lightweight, non-owning view
    /// Provides safe 2D access to contiguous gradient data without allocation
    class GradientMatrix
    {
    private:
        float* m_data;  ///< Pointer to contiguous data (not owned)
        size_t m_rows;  ///< Number of rows (frames)
        size_t m_cols;  ///< Number of columns (bins per frame)

    public:
        /// Construct matrix view from data pointer and dimensions
        /// @param data Pointer to contiguous row-major data (not owned)
        /// @param rows Number of rows (frames)
        /// @param cols Number of columns (bins per frame)
        GradientMatrix(float* data, size_t rows, size_t cols) noexcept : m_data(data), m_rows(rows), m_cols(cols) {}

        /// Default constructor creates empty matrix
        GradientMatrix() noexcept : m_data(nullptr), m_rows(0), m_cols(0) {}

        /// Access row (frame) by index - returns pointer to start of row
        /// @param row Row index [0, rows())
        /// @return Pointer to row data (bins)
        float* operator[](size_t row) noexcept
        {
            return m_data + row * m_cols;
        }

        /// Const access to row (frame) by index
        const float* operator[](size_t row) const noexcept
        {
            return m_data + row * m_cols;
        }

        /// Access element by row and column
        /// @param row Row index [0, rows())
        /// @param col Column index [0, cols())
        /// @return Reference to element
        float& operator()(size_t row, size_t col) noexcept
        {
            return m_data[row * m_cols + col];
        }

        /// Const access to element by row and column
        const float& operator()(size_t row, size_t col) const noexcept
        {
            return m_data[row * m_cols + col];
        }

        /// Get number of rows (frames)
        size_t rows() const noexcept
        {
            return m_rows;
        }

        /// Get number of columns (bins per frame)
        size_t cols() const noexcept
        {
            return m_cols;
        }

        /// Get total number of elements
        size_t size() const noexcept
        {
            return m_rows * m_cols;
        }

        /// Check if matrix is empty
        bool empty() const noexcept
        {
            return m_rows == 0 || m_cols == 0 || m_data == nullptr;
        }

        /// Get raw data pointer
        float* data() noexcept
        {
            return m_data;
        }

        /// Get const raw data pointer
        const float* data() const noexcept
        {
            return m_data;
        }

        /// Iterator support for range-based loops
        float* begin() noexcept
        {
            return m_data;
        }
        float* end() noexcept
        {
            return m_data + size();
        }
        const float* begin() const noexcept
        {
            return m_data;
        }
        const float* end() const noexcept
        {
            return m_data + size();
        }
    };

    /// Result structure containing gradient matrices with managed storage
    struct GradientResult
    {
        std::vector<float> time_data;  ///< Contiguous storage for time gradients
        std::vector<float> freq_data;  ///< Contiguous storage for frequency gradients
        size_t time_frames;            ///< Number of time frames
        size_t freq_frames;            ///< Number of frequency frames
        size_t fft_bins;               ///< Number of frequency bins per frame

        /// Get matrix view of time gradients
        GradientMatrix time_gradients() noexcept
        {
            return GradientMatrix(time_data.data(), time_frames, fft_bins);
        }

        /// Get const matrix view of time gradients
        GradientMatrix time_gradients() const noexcept
        {
            return GradientMatrix(const_cast<float*>(time_data.data()), time_frames, fft_bins);
        }

        /// Get matrix view of frequency gradients
        GradientMatrix freq_gradients() noexcept
        {
            return GradientMatrix(freq_data.data(), freq_frames, fft_bins);
        }

        /// Get const matrix view of frequency gradients
        GradientMatrix freq_gradients() const noexcept
        {
            return GradientMatrix(const_cast<float*>(freq_data.data()), freq_frames, fft_bins);
        }

        /// Validate that storage matches dimensions
        bool is_valid() const noexcept
        {
            return time_data.size() == time_frames * fft_bins && freq_data.size() == freq_frames * fft_bins &&
                   fft_bins > 0;
        }
    };

    /// Legacy output structure for pointer-based API (deprecated)
    struct GradientOutput
    {
        float* time_gradients;  ///< Time gradients [time_frames][fft_bins]
        float* freq_gradients;  ///< Frequency gradients [freq_frames][fft_bins]
        size_t time_frames;     ///< Actual frames in time output
        size_t freq_frames;     ///< Actual frames in freq output
        size_t fft_bins;        ///< Number of frequency bins per frame

        /// Validate output data
        bool is_valid() const noexcept
        {
            return time_gradients && freq_gradients && time_frames > 0 && freq_frames > 0 && fft_bins > 0;
        }
    };

    /// Calculate both time and frequency gradients from complex spectrum data
    ///
    /// Unified function that processes complex spectral data to compute both time and frequency
    /// phase gradients efficiently. Handles phase extraction and unwrapping internally.
    ///
    /// @param input Complex spectrum input data and processing parameters
    /// @param output Output structure for both gradient types
    /// @return Error code indicating success or failure
    template <typename ComplexType>
    [[nodiscard]] inline ErrorCode calculate_spectrum_gradients(const ComplexSpectrumGradientInput<ComplexType>& input,
                                                                GradientOutput& output)
    {
        if (!input.is_valid() || !output.is_valid())
        {
            return ErrorCode::INVALID_INPUT;
        }

        const size_t num_frames = input.num_frames;
        const size_t fft_bins = input.fft_bins;

        // Validate output dimensions
        if (output.fft_bins != fft_bins)
        {
            return ErrorCode::SIZE_MISMATCH;
        }

        // Time gradients: output same number of frames as input using boundary conditions
        const size_t time_output_frames = num_frames;

        if (output.time_frames < time_output_frames)
        {
            return ErrorCode::SIZE_MISMATCH;
        }

        // Process time gradients with boundary handling
        for (size_t frame = 0; frame < time_output_frames; ++frame)
        {
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                float curr_phase, prev_phase, next_phase;

                if (input.time_method == GradientMethod::CENTRAL && frame > 0 && frame + 1 < num_frames)
                {
                    // Central difference for interior points
                    prev_phase = std::arg(input.spectra[frame - 1][bin]);
                    next_phase = std::arg(input.spectra[frame + 1][bin]);
                    output.time_gradients[frame * fft_bins + bin] =
                        phaseDiff(next_phase, prev_phase) / (2.0f * input.time_step);
                }
                else if (frame == 0)
                {
                    // First frame: use forward difference
                    if (num_frames > 1)
                    {
                        curr_phase = std::arg(input.spectra[0][bin]);
                        next_phase = std::arg(input.spectra[1][bin]);
                        output.time_gradients[frame * fft_bins + bin] =
                            phaseDiff(next_phase, curr_phase) / input.time_step;
                    }
                    else
                    {
                        output.time_gradients[frame * fft_bins + bin] = 0.0f;  // Single frame
                    }
                }
                else if (frame == num_frames - 1)
                {
                    // Last frame: use backward difference
                    curr_phase = std::arg(input.spectra[frame][bin]);
                    prev_phase = std::arg(input.spectra[frame - 1][bin]);
                    output.time_gradients[frame * fft_bins + bin] = phaseDiff(curr_phase, prev_phase) / input.time_step;
                }
                else
                {
                    // General case for FORWARD/BACKWARD methods
                    switch (input.time_method)
                    {
                    case GradientMethod::FORWARD:
                        if (frame + 1 < num_frames)
                        {
                            curr_phase = std::arg(input.spectra[frame][bin]);
                            next_phase = std::arg(input.spectra[frame + 1][bin]);
                            output.time_gradients[frame * fft_bins + bin] =
                                phaseDiff(next_phase, curr_phase) / input.time_step;
                        }
                        else
                        {
                            // Last frame: use backward difference
                            curr_phase = std::arg(input.spectra[frame][bin]);
                            prev_phase = std::arg(input.spectra[frame - 1][bin]);
                            output.time_gradients[frame * fft_bins + bin] =
                                phaseDiff(curr_phase, prev_phase) / input.time_step;
                        }
                        break;

                    case GradientMethod::BACKWARD:
                        if (frame > 0)
                        {
                            curr_phase = std::arg(input.spectra[frame][bin]);
                            prev_phase = std::arg(input.spectra[frame - 1][bin]);
                            output.time_gradients[frame * fft_bins + bin] =
                                phaseDiff(curr_phase, prev_phase) / input.time_step;
                        }
                        else
                        {
                            // First frame: use forward difference
                            curr_phase = std::arg(input.spectra[0][bin]);
                            next_phase = std::arg(input.spectra[1][bin]);
                            output.time_gradients[frame * fft_bins + bin] =
                                phaseDiff(next_phase, curr_phase) / input.time_step;
                        }
                        break;

                    default:
                        return ErrorCode::INVALID_INPUT;
                    }
                }
            }
        }

        // Process frequency gradients for each frame
        if (output.freq_frames < num_frames)
        {
            return ErrorCode::SIZE_MISMATCH;
        }

        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                float curr_phase, prev_phase, next_phase;

                switch (input.freq_method)
                {
                case GradientMethod::FORWARD:
                    if (bin + 1 < fft_bins)
                    {
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        next_phase = std::arg(input.spectra[frame][bin + 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(next_phase, curr_phase) / input.freq_step;
                    }
                    else
                    {
                        // Boundary: use backward difference
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        prev_phase = std::arg(input.spectra[frame][bin - 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(curr_phase, prev_phase) / input.freq_step;
                    }
                    break;

                case GradientMethod::BACKWARD:
                    if (bin > 0)
                    {
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        prev_phase = std::arg(input.spectra[frame][bin - 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(curr_phase, prev_phase) / input.freq_step;
                    }
                    else
                    {
                        // Boundary: use forward difference
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        next_phase = std::arg(input.spectra[frame][bin + 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(next_phase, curr_phase) / input.freq_step;
                    }
                    break;

                case GradientMethod::CENTRAL:
                    if (bin > 0 && bin + 1 < fft_bins)
                    {
                        prev_phase = std::arg(input.spectra[frame][bin - 1]);
                        next_phase = std::arg(input.spectra[frame][bin + 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(next_phase, prev_phase) / (2.0f * input.freq_step);
                    }
                    else if (bin == 0)
                    {
                        // Boundary: use forward difference
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        next_phase = std::arg(input.spectra[frame][bin + 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(next_phase, curr_phase) / input.freq_step;
                    }
                    else  // bin == fft_bins - 1
                    {
                        // Boundary: use backward difference
                        curr_phase = std::arg(input.spectra[frame][bin]);
                        prev_phase = std::arg(input.spectra[frame][bin - 1]);
                        output.freq_gradients[frame * fft_bins + bin] =
                            phaseDiff(curr_phase, prev_phase) / input.freq_step;
                    }
                    break;

                default:
                    return ErrorCode::INVALID_INPUT;
                }
            }
        }

        return ErrorCode::OK;
    }

    /// Convenience function for std::vector<std::vector<std::complex<float>>> input
    ///
    /// @param spectra Vector of complex spectrum frames
    /// @param time_step Time step between frames
    /// @param freq_step Frequency step between bins
    /// @param output Output gradients structure
    /// @param time_method Time gradient method (default: CENTRAL)
    /// @param freq_method Frequency gradient method (default: CENTRAL)
    /// @return Error code indicating success or failure
    [[nodiscard]] inline ErrorCode calculate_spectrum_gradients(
        const std::vector<std::vector<std::complex<float>>>& spectra,
        float time_step,
        float freq_step,
        GradientOutput& output,
        GradientMethod time_method = GradientMethod::CENTRAL,
        GradientMethod freq_method = GradientMethod::CENTRAL)
    {
        if (spectra.empty() || spectra[0].empty())
        {
            return ErrorCode::INVALID_INPUT;
        }

        // Create array of pointers for template function
        std::vector<const std::complex<float>*> spectrum_ptrs(spectra.size());
        for (size_t i = 0; i < spectra.size(); ++i)
        {
            spectrum_ptrs[i] = spectra[i].data();
        }

        ComplexSpectrumGradientInput<std::complex<float>> input {
            spectrum_ptrs.data(), spectra.size(), spectra[0].size(), time_step, freq_step, time_method, freq_method
        };

        return calculate_spectrum_gradients(input, output);
    }

    /// Convenience function for raw pointer array input
    ///
    /// @param spectra Array of pointers to complex frames
    /// @param num_frames Number of time frames
    /// @param fft_bins Number of frequency bins per frame
    /// @param time_step Time step between frames
    /// @param freq_step Frequency step between bins
    /// @param output Output gradients structure
    /// @param time_method Time gradient method (default: CENTRAL)
    /// @param freq_method Frequency gradient method (default: CENTRAL)
    /// @return Error code indicating success or failure
    [[nodiscard]] inline ErrorCode calculate_spectrum_gradients(const std::complex<float>* const* spectra,
                                                                size_t num_frames,
                                                                size_t fft_bins,
                                                                float time_step,
                                                                float freq_step,
                                                                GradientOutput& output,
                                                                GradientMethod time_method = GradientMethod::CENTRAL,
                                                                GradientMethod freq_method = GradientMethod::CENTRAL)
    {
        ComplexSpectrumGradientInput<std::complex<float>> input { spectra,   num_frames,  fft_bins,   time_step,
                                                                  freq_step, time_method, freq_method };

        return calculate_spectrum_gradients(input, output);
    }

}  // namespace rtpghi
