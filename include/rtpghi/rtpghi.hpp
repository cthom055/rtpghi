#pragma once
#include <cstddef>

namespace rtpghi
{

    /// Input frame data for RTPGHI processing
    struct FrameInput
    {
        const float* magnitudes;       ///< Current frame magnitudes [fft_bins]
        const float* previous_phases;  ///< Previous frame phases [fft_bins]
        const float* time_gradients;   ///< Time phase gradients [fft_bins]
        const float* freq_gradients;   ///< Frequency phase gradients [fft_bins]
        size_t fft_bins;               ///< Number of FFT bins (fft_size/2 + 1)

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

    /// Error codes for RTPGHI processing
    enum class ErrorCode
    {
        OK = 0,          ///< Success
        INVALID_INPUT,   ///< Invalid input data
        INVALID_OUTPUT,  ///< Invalid output buffers
        SIZE_MISMATCH    ///< Input/output size mismatch
    };

    /// Process a frame using Real-Time Phase Gradient Heap Integration
    ///
    /// This is the core RTPGHI algorithm that propagates phases using gradient information.
    ///
    /// @param input Input frame data with gradients and previous phases
    /// @param output Output buffers for processed magnitudes and phases
    /// @return Error code indicating success or failure
    inline ErrorCode process(const FrameInput& input, FrameOutput& output)
    {
        // Validation
        if (!input.is_valid())
            return ErrorCode::INVALID_INPUT;
        if (!output.is_valid())
            return ErrorCode::INVALID_OUTPUT;
        if (input.fft_bins != output.fft_bins)
            return ErrorCode::SIZE_MISMATCH;

        // Copy magnitudes (pass-through for now)
        for (size_t i = 0; i < input.fft_bins; ++i)
        {
            output.magnitudes[i] = input.magnitudes[i];
        }

        // TODO: Implement RTPGHI phase propagation algorithm
        // For now, just copy previous phases as placeholder
        for (size_t i = 0; i < input.fft_bins; ++i)
        {
            output.phases[i] = input.previous_phases[i] + input.time_gradients[i];
        }

        return ErrorCode::OK;
    }

}  // namespace rtpghi
