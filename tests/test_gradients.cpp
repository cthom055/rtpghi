#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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
        std::vector<float> gradients(phases.size());
        float step = 1.0f;

        rtpghi::GradientInput input { phases.data(), phases.size(), step, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // Forward difference: grad[i] = (phase[i+1] - phase[i]) / step
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (last point uses backward difference)
        for (size_t i = 0; i < phases.size(); ++i)
        {
            REQUIRE(gradients[i] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Backward difference method")
    {
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f };
        std::vector<float> gradients(phases.size());
        float step = 1.0f;

        rtpghi::GradientInput input { phases.data(), phases.size(), step, rtpghi::GradientMethod::BACKWARD };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // Backward difference: grad[i] = (phase[i] - phase[i-1]) / step
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (first point uses forward difference)
        for (size_t i = 0; i < phases.size(); ++i)
        {
            REQUIRE(gradients[i] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Central difference method")
    {
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f };
        std::vector<float> gradients(phases.size());
        float step = 1.0f;

        rtpghi::GradientInput input { phases.data(), phases.size(), step, rtpghi::GradientMethod::CENTRAL };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // Central difference: grad[i] = (phase[i+1] - phase[i-1]) / (2*step)
        // Expected: [0.1, 0.1, 0.1, 0.1, 0.1] (boundaries use forward/backward)
        for (size_t i = 0; i < phases.size(); ++i)
        {
            REQUIRE(gradients[i] == Approx(0.1f).epsilon(eps));
        }
    }

    SECTION("Phase unwrapping in gradient calculation")
    {
        // Test phases that wrap around 2π boundary
        std::vector<float> phases = { -pi + 0.1f, -pi + 0.2f, pi - 0.1f, -pi + 0.1f };
        std::vector<float> gradients(phases.size());
        float step = 1.0f;

        rtpghi::GradientInput input { phases.data(), phases.size(), step, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // Should handle phase wrapping correctly
        REQUIRE(gradients[0] == Approx(0.1f).epsilon(eps));  // Normal difference
        REQUIRE(std::abs(gradients[1]) < pi);                // Should not be ~2π
        REQUIRE(std::abs(gradients[2]) < pi);                // Should handle wrap-around
    }

    SECTION("Non-uniform step sizes")
    {
        std::vector<float> phases = { 0.0f, 0.5f, 1.0f, 1.5f };
        std::vector<float> gradients(phases.size());
        float step = 0.5f;  // Different step size

        rtpghi::GradientInput input { phases.data(), phases.size(), step, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // With step=0.5, gradient should be (0.5 - 0.0) / 0.5 = 1.0
        REQUIRE(gradients[0] == Approx(1.0f).epsilon(eps));
        REQUIRE(gradients[1] == Approx(1.0f).epsilon(eps));
        REQUIRE(gradients[2] == Approx(1.0f).epsilon(eps));
    }

    SECTION("Quadratic phase test for central difference accuracy")
    {
        // Test f(x) = x^2, f'(x) = 2x
        // Central difference should be more accurate than forward/backward
        const size_t N = 5;
        std::vector<float> phases(N);
        std::vector<float> grad_forward(N), grad_backward(N), grad_central(N);
        float step = 0.1f;

        // Generate quadratic phase: phase[i] = (i*step)^2
        for (size_t i = 0; i < N; ++i)
        {
            float x = static_cast<float>(i) * step;
            phases[i] = x * x;
        }

        // Calculate gradients with all methods
        rtpghi::GradientInput input_forward { phases.data(), N, step, rtpghi::GradientMethod::FORWARD };
        rtpghi::GradientInput input_backward { phases.data(), N, step, rtpghi::GradientMethod::BACKWARD };
        rtpghi::GradientInput input_central { phases.data(), N, step, rtpghi::GradientMethod::CENTRAL };

        REQUIRE(rtpghi::calculate_gradients(input_forward, grad_forward.data()) == rtpghi::ErrorCode::OK);
        REQUIRE(rtpghi::calculate_gradients(input_backward, grad_backward.data()) == rtpghi::ErrorCode::OK);
        REQUIRE(rtpghi::calculate_gradients(input_central, grad_central.data()) == rtpghi::ErrorCode::OK);

        // For middle points, central difference should be most accurate
        // True derivative at x=0.2 is 2*0.2 = 0.4
        size_t mid_point = 2;
        float x_mid = static_cast<float>(mid_point) * step;
        float true_derivative = 2.0f * x_mid;

        INFO("True derivative at x=" << x_mid << " is " << true_derivative);
        INFO("Forward difference: " << grad_forward[mid_point]);
        INFO("Backward difference: " << grad_backward[mid_point]);
        INFO("Central difference: " << grad_central[mid_point]);

        // Central difference should be closest to true value
        float error_central = std::abs(grad_central[mid_point] - true_derivative);
        float error_forward = std::abs(grad_forward[mid_point] - true_derivative);
        float error_backward = std::abs(grad_backward[mid_point] - true_derivative);

        REQUIRE(error_central <= error_forward);
        REQUIRE(error_central <= error_backward);
    }
}

TEST_CASE("Time Gradient Calculations", "[gradients][time]")
{
    const float eps = 1e-6f;

    SECTION("Simple time evolution")
    {
        const size_t fft_bins = 4;
        std::vector<float> prev_phases = { 0.0f, 0.1f, 0.2f, 0.3f };
        std::vector<float> curr_phases = { 0.05f, 0.15f, 0.25f, 0.35f };
        std::vector<float> time_gradients(fft_bins);
        float time_step = 0.1f;

        REQUIRE(rtpghi::calculate_time_gradients(prev_phases.data(), curr_phases.data(), fft_bins,
                                               time_step, rtpghi::GradientMethod::FORWARD,
                                               time_gradients.data()) == rtpghi::ErrorCode::OK);

        // Expected: (0.05 - 0.0) / 0.1 = 0.5 for all bins
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(time_gradients[i] == Approx(0.5f).epsilon(eps));
        }
    }

    SECTION("Phase wrap handling in time gradients")
    {
        const size_t fft_bins = 2;
        const float pi = 3.14159265359f;
        std::vector<float> prev_phases = { pi - 0.1f, -pi + 0.1f };
        std::vector<float> curr_phases = { -pi + 0.1f, pi - 0.1f };
        std::vector<float> time_gradients(fft_bins);
        float time_step = 1.0f;

        REQUIRE(rtpghi::calculate_time_gradients(prev_phases.data(), curr_phases.data(), fft_bins,
                                               time_step, rtpghi::GradientMethod::FORWARD,
                                               time_gradients.data()) == rtpghi::ErrorCode::OK);

        // Should handle phase wrapping - gradients should be small, not ~2π
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(std::abs(time_gradients[i]) < 1.0f);  // Should be wrapped properly
        }
    }
}

TEST_CASE("Frequency Gradient Calculations", "[gradients][frequency]")
{
    const float eps = 1e-6f;

    SECTION("Linear frequency sweep")
    {
        // Phase increases linearly with frequency
        const size_t fft_bins = 5;
        std::vector<float> phases = { 0.0f, 0.2f, 0.4f, 0.6f, 0.8f };
        std::vector<float> freq_gradients(fft_bins);
        float freq_step = 100.0f;  // Hz per bin

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                               rtpghi::GradientMethod::CENTRAL,
                                               freq_gradients.data()) == rtpghi::ErrorCode::OK);

        // Expected gradient: 0.2 / 100 = 0.002 rad/Hz
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(freq_gradients[i] == Approx(0.002f).epsilon(eps));
        }
    }

    SECTION("Comparison of gradient methods for frequency")
    {
        const size_t fft_bins = 7;
        std::vector<float> phases(fft_bins);
        std::vector<float> grad_forward(fft_bins), grad_backward(fft_bins), grad_central(fft_bins);
        float freq_step = 50.0f;

        // Generate sinusoidal phase pattern
        for (size_t i = 0; i < fft_bins; ++i)
        {
            phases[i] = 0.5f * std::sin(2.0f * 3.14159f * static_cast<float>(i) / static_cast<float>(fft_bins));
        }

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                               rtpghi::GradientMethod::FORWARD,
                                               grad_forward.data()) == rtpghi::ErrorCode::OK);

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                               rtpghi::GradientMethod::BACKWARD,
                                               grad_backward.data()) == rtpghi::ErrorCode::OK);

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                               rtpghi::GradientMethod::CENTRAL,
                                               grad_central.data()) == rtpghi::ErrorCode::OK);

        // All methods should produce finite results
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(std::isfinite(grad_forward[i]));
            REQUIRE(std::isfinite(grad_backward[i]));
            REQUIRE(std::isfinite(grad_central[i]));
        }

        // For smooth functions, central difference should be different from forward/backward
        // (indicating higher accuracy for interior points)
        bool methods_differ = false;
        for (size_t i = 1; i < fft_bins - 1; ++i)
        {
            if (std::abs(grad_central[i] - grad_forward[i]) > eps ||
                std::abs(grad_central[i] - grad_backward[i]) > eps)
            {
                methods_differ = true;
                break;
            }
        }
        REQUIRE(methods_differ);  // Different methods should give different results
    }
}

TEST_CASE("Gradient API Error Handling", "[gradients][error]")
{
    SECTION("Invalid input validation")
    {
        std::vector<float> phases = { 0.0f, 0.1f, 0.2f };
        std::vector<float> gradients(phases.size());

        // Null phases pointer
        rtpghi::GradientInput bad_input1 { nullptr, phases.size(), 1.0f, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(bad_input1, gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        // Zero size
        rtpghi::GradientInput bad_input2 { phases.data(), 0, 1.0f, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(bad_input2, gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        // Zero or negative step size
        rtpghi::GradientInput bad_input3 { phases.data(), phases.size(), 0.0f, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(bad_input3, gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        // Null output
        rtpghi::GradientInput valid_input { phases.data(), phases.size(), 1.0f, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(valid_input, nullptr) == rtpghi::ErrorCode::INVALID_INPUT);
    }

    SECTION("Time gradient error handling")
    {
        const size_t fft_bins = 3;
        std::vector<float> phases(fft_bins, 0.0f);
        std::vector<float> gradients(fft_bins);

        // Null input pointers
        REQUIRE(rtpghi::calculate_time_gradients(nullptr, phases.data(), fft_bins, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        REQUIRE(rtpghi::calculate_time_gradients(phases.data(), nullptr, fft_bins, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        REQUIRE(rtpghi::calculate_time_gradients(phases.data(), phases.data(), fft_bins, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               nullptr) == rtpghi::ErrorCode::INVALID_INPUT);

        // Zero bins or time step
        REQUIRE(rtpghi::calculate_time_gradients(phases.data(), phases.data(), 0, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        REQUIRE(rtpghi::calculate_time_gradients(phases.data(), phases.data(), fft_bins, 0.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);
    }

    SECTION("Frequency gradient error handling")
    {
        const size_t fft_bins = 3;
        std::vector<float> phases(fft_bins, 0.0f);
        std::vector<float> gradients(fft_bins);

        // Null input pointers
        REQUIRE(rtpghi::calculate_freq_gradients(nullptr, fft_bins, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               nullptr) == rtpghi::ErrorCode::INVALID_INPUT);

        // Zero bins or frequency step
        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), 0, 1.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);

        REQUIRE(rtpghi::calculate_freq_gradients(phases.data(), fft_bins, 0.0f,
                                               rtpghi::GradientMethod::FORWARD,
                                               gradients.data()) == rtpghi::ErrorCode::INVALID_INPUT);
    }

    SECTION("Single element arrays")
    {
        std::vector<float> phases = { 1.5f };
        std::vector<float> gradients(1);

        rtpghi::GradientInput input { phases.data(), 1, 1.0f, rtpghi::GradientMethod::FORWARD };
        REQUIRE(rtpghi::calculate_gradients(input, gradients.data()) == rtpghi::ErrorCode::OK);

        // For single element, should handle gracefully (result may be 0 or undefined)
        REQUIRE(std::isfinite(gradients[0]));
    }
}