#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <rtpghi/rtpghi.hpp>
#include <vector>

using Catch::Approx;

TEST_CASE("Enhanced Spectrum Gradient API", "[gradients][spectrum][enhanced]")
{
    const float eps = 1e-5f;

    SECTION("Complex spectrum to time gradients")
    {
        // Create simple complex spectra with linear phase evolution
        const size_t num_frames = 3;
        const size_t fft_bins = 4;
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        
        // Initialize each frame
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                // Phase increases linearly with time: phase = frame * 0.1
                float magnitude = 1.0f;
                float phase = static_cast<float>(frame) * 0.1f;
                spectra[frame][bin] = std::polar(magnitude, phase);
            }
        }

        // Prepare output buffers - time gradients now same size as input frames
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);

        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        float time_step = 0.1f;
        float freq_step = 100.0f;

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, time_step, freq_step, output,
            rtpghi::GradientMethod::FORWARD,
            rtpghi::GradientMethod::FORWARD
        ) == rtpghi::ErrorCode::OK);

        // Check time gradients: should be 0.1 / 0.1 = 1.0 for all bins (interior points)
        // First and last frames use boundary conditions but should still be 1.0 for linear phase
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                float gradient = time_gradients[frame * fft_bins + bin];
                REQUIRE(gradient == Approx(1.0f).epsilon(eps));
            }
        }
    }

    SECTION("Complex spectrum to frequency gradients")
    {
        // Create complex spectra with linear phase across frequency
        const size_t num_frames = 2;
        const size_t fft_bins = 5;
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                // Phase increases linearly with frequency: phase = bin * 0.2
                float magnitude = 1.0f;
                float phase = static_cast<float>(bin) * 0.2f;
                spectra[frame][bin] = std::polar(magnitude, phase);
            }
        }

        // Prepare output buffers - time gradients same size as input frames  
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);

        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        float time_step = 0.1f;
        float freq_step = 100.0f;

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, time_step, freq_step, output,
            rtpghi::GradientMethod::FORWARD,
            rtpghi::GradientMethod::FORWARD
        ) == rtpghi::ErrorCode::OK);

        // Check frequency gradients: should be 0.2 / 100.0 = 0.002 for interior bins
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            for (size_t bin = 0; bin < fft_bins - 1; ++bin) // Skip last bin (boundary condition)
            {
                float gradient = freq_gradients[frame * fft_bins + bin];
                REQUIRE(gradient == Approx(0.002f).epsilon(eps));
            }
        }
    }

    SECTION("Raw pointer interface")
    {
        const size_t num_frames = 3;
        const size_t fft_bins = 4;
        
        // Create complex spectra using raw pointers
        std::vector<std::complex<float>> frame0 = {
            std::polar(1.0f, 0.0f), std::polar(1.0f, 0.0f), 
            std::polar(1.0f, 0.0f), std::polar(1.0f, 0.0f)
        };
        std::vector<std::complex<float>> frame1 = {
            std::polar(1.0f, 0.1f), std::polar(1.0f, 0.1f), 
            std::polar(1.0f, 0.1f), std::polar(1.0f, 0.1f)
        };
        std::vector<std::complex<float>> frame2 = {
            std::polar(1.0f, 0.2f), std::polar(1.0f, 0.2f), 
            std::polar(1.0f, 0.2f), std::polar(1.0f, 0.2f)
        };

        const std::complex<float>* spectra_ptrs[] = {
            frame0.data(), frame1.data(), frame2.data()
        };

        // Prepare output - time gradients same size as input
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);

        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        float time_step = 0.1f;
        float freq_step = 100.0f;

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra_ptrs, num_frames, fft_bins, time_step, freq_step, output,
            rtpghi::GradientMethod::FORWARD, rtpghi::GradientMethod::FORWARD
        ) == rtpghi::ErrorCode::OK);

        // Time gradients should be 0.1 / 0.1 = 1.0
        for (size_t i = 0; i < time_gradients.size(); ++i)
        {
            REQUIRE(time_gradients[i] == Approx(1.0f).epsilon(eps));
        }
    }

    SECTION("Central difference method")
    {
        const size_t num_frames = 5;
        const size_t fft_bins = 3;
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        
        // Quadratic phase evolution: phase = frame^2 * 0.01
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                float phase = static_cast<float>(frame * frame) * 0.01f;
                spectra[frame][bin] = std::polar(1.0f, phase);
            }
        }

        // Time gradients now have same number of frames as input
        const size_t expected_time_frames = num_frames;
        std::vector<float> time_gradients(expected_time_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);

        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            expected_time_frames,
            num_frames,
            fft_bins
        };

        float time_step = 1.0f;
        float freq_step = 100.0f;

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, time_step, freq_step, output,
            rtpghi::GradientMethod::CENTRAL,
            rtpghi::GradientMethod::FORWARD
        ) == rtpghi::ErrorCode::OK);

        // For quadratic phase f(t) = t^2 * 0.01, derivative f'(t) = 2t * 0.01
        // Frame 0: uses forward difference at t=0: (f(1) - f(0)) / 1 = (0.01 - 0.0) / 1 = 0.01  
        // Frame 1: uses central difference at t=1: (f(2) - f(0)) / 2 = (0.04 - 0.0) / 2 = 0.02
        // Frame 2: uses central difference at t=2: (f(3) - f(1)) / 2 = (0.09 - 0.01) / 2 = 0.04
        // Frame 3: uses central difference at t=3: (f(4) - f(2)) / 2 = (0.16 - 0.04) / 2 = 0.06
        // Frame 4: uses backward difference at t=4: (f(4) - f(3)) / 1 = (0.16 - 0.09) / 1 = 0.07

        REQUIRE(time_gradients[0] == Approx(0.01f).epsilon(eps)); // Frame 0 (t=0, forward)
        REQUIRE(time_gradients[fft_bins] == Approx(0.02f).epsilon(eps)); // Frame 1 (t=1, central)
        REQUIRE(time_gradients[2 * fft_bins] == Approx(0.04f).epsilon(eps)); // Frame 2 (t=2, central)
        REQUIRE(time_gradients[3 * fft_bins] == Approx(0.06f).epsilon(eps)); // Frame 3 (t=3, central)
        REQUIRE(time_gradients[4 * fft_bins] == Approx(0.07f).epsilon(eps)); // Frame 4 (t=4, backward)
    }

    SECTION("Error handling")
    {
        std::vector<std::vector<std::complex<float>>> empty_spectra;
        std::vector<float> gradients(10);

        rtpghi::GradientOutput output{
            gradients.data(),
            gradients.data(),
            1,
            1,
            5
        };

        // Empty input should fail
        REQUIRE(rtpghi::calculate_spectrum_gradients(
            empty_spectra, 0.1f, 100.0f, output
        ) == rtpghi::ErrorCode::INVALID_INPUT);

        // Create valid input but invalid output
        std::vector<std::vector<std::complex<float>>> valid_spectra(2, std::vector<std::complex<float>>(4));
        
        rtpghi::GradientOutput bad_output{
            nullptr,  // Invalid pointer
            gradients.data(),
            1,
            1,
            4
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            valid_spectra, 0.1f, 100.0f, bad_output
        ) == rtpghi::ErrorCode::INVALID_INPUT);
    }
}