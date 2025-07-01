#include <catch2/catch_test_macros.hpp>
#include <rtpghi/rtpghi.hpp>
#include <vector>

TEST_CASE("RTPGHI basic functionality", "[core]")
{
    const size_t fft_bins = 512;

    SECTION("Valid input processing")
    {
        // Setup test data
        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        rtpghi::FrameInput input { mags.data(), prev_phases.data(), time_grad.data(), freq_grad.data(), fft_bins };

        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(rtpghi::process(input, output) == rtpghi::ErrorCode::OK);

        // Check that magnitudes are passed through
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(out_mags[i] == 1.0f);
        }

        // Check that phases are modified (simple gradient integration)
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(out_phases[i] == 0.1f);  // prev_phase + time_gradient
        }
    }

    SECTION("Input validation")
    {
        std::vector<float> data(fft_bins, 0.0f);
        std::vector<float> output_data(fft_bins);

        // Null pointer in input
        rtpghi::FrameInput bad_input { nullptr, data.data(), data.data(), data.data(), fft_bins };
        rtpghi::FrameOutput output { output_data.data(), output_data.data(), fft_bins };

        REQUIRE(rtpghi::process(bad_input, output) == rtpghi::ErrorCode::INVALID_INPUT);
    }

    SECTION("Output validation")
    {
        std::vector<float> data(fft_bins, 0.0f);

        rtpghi::FrameInput input { data.data(), data.data(), data.data(), data.data(), fft_bins };

        // Null pointer in output
        rtpghi::FrameOutput bad_output { nullptr, data.data(), fft_bins };

        REQUIRE(rtpghi::process(input, bad_output) == rtpghi::ErrorCode::INVALID_OUTPUT);
    }

    SECTION("Size mismatch")
    {
        std::vector<float> data(fft_bins, 0.0f);

        rtpghi::FrameInput input { data.data(), data.data(), data.data(), data.data(), fft_bins };

        // Wrong output size
        rtpghi::FrameOutput output {
            data.data(), data.data(), fft_bins / 2  // Wrong size
        };

        REQUIRE(rtpghi::process(input, output) == rtpghi::ErrorCode::SIZE_MISMATCH);
    }

    SECTION("Input validation methods")
    {
        std::vector<float> data(fft_bins, 0.0f);

        // Valid input
        rtpghi::FrameInput valid_input { data.data(), data.data(), data.data(), data.data(), fft_bins };
        REQUIRE(valid_input.is_valid() == true);

        // Invalid input (null pointer)
        rtpghi::FrameInput invalid_input { nullptr, data.data(), data.data(), data.data(), fft_bins };
        REQUIRE(invalid_input.is_valid() == false);

        // Invalid input (zero size)
        rtpghi::FrameInput zero_size_input { data.data(), data.data(), data.data(), data.data(), 0 };
        REQUIRE(zero_size_input.is_valid() == false);
    }
}
