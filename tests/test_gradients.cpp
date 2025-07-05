#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <rtpghi/rtpghi.hpp>
#include <vector>

using Catch::Approx;

TEST_CASE("Phase Gradient Calculations", "[gradients]")
{
    const float pi = 3.14159265359f;
    const float eps = 1e-5f;

    SECTION("Forward difference method")
    {
        // Test with a simple linear phase: phase[i] = 0.1 * i
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f };
        const size_t num_frames = phases.size();
        const size_t fft_bins = 1;
        float step = 1.0f;

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);
        
        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output,
            rtpghi::GradientMethod::FORWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // Forward difference: grad[i] = (phase[i+1] - phase[i]) / step
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (last point uses backward difference)
        for (size_t i = 0; i < num_frames; ++i)
        {
            REQUIRE(time_gradients[i * fft_bins + 0] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Backward difference method")
    {
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f };
        const size_t num_frames = phases.size();
        const size_t fft_bins = 1;
        float step = 1.0f;

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);
        
        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output,
            rtpghi::GradientMethod::BACKWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // Backward difference: grad[i] = (phase[i] - phase[i-1]) / step
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (first point uses forward difference)
        for (size_t i = 0; i < num_frames; ++i)
        {
            REQUIRE(time_gradients[i * fft_bins + 0] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Central difference method")
    {
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f };
        const size_t num_frames = phases.size();
        const size_t fft_bins = 1;
        float step = 1.0f;

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);
        
        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output,
            rtpghi::GradientMethod::CENTRAL, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // Central difference: grad[i] = (phase[i+1] - phase[i-1]) / (2*step)
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (boundaries use forward/backward)
        for (size_t i = 0; i < num_frames; ++i)
        {
            REQUIRE(time_gradients[i * fft_bins + 0] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Phase unwrapping in gradient calculation")
    {
        // Test phases that wrap around 2π boundary
        std::vector<float> phases = { -pi + 0.1f, -pi + 0.2f, pi - 0.1f, -pi + 0.1f };
        const size_t num_frames = phases.size();
        const size_t fft_bins = 1;
        float step = 1.0f;

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);
        
        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output,
            rtpghi::GradientMethod::FORWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // Should handle phase wrapping correctly
        REQUIRE(time_gradients[0] == Approx(0.1f).epsilon(eps));  // Normal difference
        REQUIRE(std::abs(time_gradients[1]) < pi);                // Should not be ~2π
        REQUIRE(std::abs(time_gradients[2]) < pi);                // Should handle wrap-around
    }

    SECTION("Non-uniform step sizes")
    {
        std::vector<float> phases = { 0.0f, 0.5f, 1.0f, 1.5f };
        const size_t num_frames = phases.size();
        const size_t fft_bins = 1;
        float step = 0.5f;  // Different step size

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(num_frames);
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers
        std::vector<float> time_gradients(num_frames * fft_bins);
        std::vector<float> freq_gradients(num_frames * fft_bins);
        
        rtpghi::GradientOutput output{
            time_gradients.data(),
            freq_gradients.data(),
            num_frames,
            num_frames,
            fft_bins
        };

        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output,
            rtpghi::GradientMethod::FORWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // With step=0.5, gradient should be (0.5 - 0.0) / 0.5 = 1.0
        REQUIRE(time_gradients[0] == Approx(1.0f).epsilon(eps));
        REQUIRE(time_gradients[1] == Approx(1.0f).epsilon(eps));
        REQUIRE(time_gradients[2] == Approx(1.0f).epsilon(eps));
    }

    SECTION("Quadratic phase test for central difference accuracy")
    {
        // Test f(x) = x^2, f'(x) = 2x
        // Central difference should be more accurate than forward/backward
        const size_t N = 5;
        const size_t fft_bins = 1;
        std::vector<float> phases(N);
        std::vector<float> grad_forward(N), grad_backward(N), grad_central(N);
        float step = 0.1f;

        // Generate quadratic phase: phase[i] = (i*step)^2
        for (size_t i = 0; i < N; ++i)
        {
            float x = static_cast<float>(i) * step;
            phases[i] = x * x;
        }

        // Convert phases to complex spectrum format
        std::vector<std::vector<std::complex<float>>> spectra(N);
        for (size_t frame = 0; frame < N; ++frame)
        {
            spectra[frame].resize(fft_bins);
            spectra[frame][0] = std::polar(1.0f, phases[frame]);
        }

        // Prepare output buffers for all methods
        std::vector<float> time_gradients_forward(N * fft_bins);
        std::vector<float> freq_gradients_forward(N * fft_bins);
        std::vector<float> time_gradients_backward(N * fft_bins);
        std::vector<float> freq_gradients_backward(N * fft_bins);
        std::vector<float> time_gradients_central(N * fft_bins);
        std::vector<float> freq_gradients_central(N * fft_bins);
        
        rtpghi::GradientOutput output_forward{
            time_gradients_forward.data(),
            freq_gradients_forward.data(),
            N, N, fft_bins
        };
        rtpghi::GradientOutput output_backward{
            time_gradients_backward.data(),
            freq_gradients_backward.data(),
            N, N, fft_bins
        };
        rtpghi::GradientOutput output_central{
            time_gradients_central.data(),
            freq_gradients_central.data(),
            N, N, fft_bins
        };

        // Calculate gradients with all methods
        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output_forward,
            rtpghi::GradientMethod::FORWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);
        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output_backward,
            rtpghi::GradientMethod::BACKWARD, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);
        REQUIRE(rtpghi::calculate_spectrum_gradients(
            spectra, step, 1.0f, output_central,
            rtpghi::GradientMethod::CENTRAL, rtpghi::GradientMethod::CENTRAL
        ) == rtpghi::ErrorCode::OK);

        // For middle points, central difference should be most accurate
        // True derivative at x=0.2 is 2*0.2 = 0.4
        size_t mid_point = 2;
        float x_mid = static_cast<float>(mid_point) * step;
        float true_derivative = 2.0f * x_mid;

        INFO("True derivative at x=" << x_mid << " is " << true_derivative);
        INFO("Forward difference: " << time_gradients_forward[mid_point]);
        INFO("Backward difference: " << time_gradients_backward[mid_point]);
        INFO("Central difference: " << time_gradients_central[mid_point]);

        // Central difference should be closest to true value
        float error_central = std::abs(time_gradients_central[mid_point] - true_derivative);
        float error_forward = std::abs(time_gradients_forward[mid_point] - true_derivative);
        float error_backward = std::abs(time_gradients_backward[mid_point] - true_derivative);

        REQUIRE(error_central <= error_forward);
        REQUIRE(error_central <= error_backward);
    }
}