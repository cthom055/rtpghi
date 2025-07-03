#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <rtpghi/rtpghi.hpp>
#include <vector>

using Catch::Approx;

// No helper function needed - create processor directly

TEST_CASE("RTPGHI Phase Propagation Algorithm", "[algorithm][failing]")
{
    const size_t fft_bins = 32;  // Small size for detailed testing
    const float tolerance = 1e-6f;
    const float pi = 3.14159265359f;

    SECTION("Heap-based magnitude-priority processing")
    {
        // Setup: High magnitude bins should be processed first
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        // Create magnitude pattern: peak at bin 15, low elsewhere
        for (size_t i = 0; i < fft_bins; ++i)
        {
            mags[i] = (i == 15) ? 1.0f : 0.01f;  // Single strong peak
        }

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Algorithm should process high-magnitude bins first
        // The peak bin (15) should have phase = prev_phase[15] + time_grad[15] = 0 + 0.1 = 0.1
        REQUIRE(out_phases[15] == Approx(0.1f).epsilon(1e-6));

        // FAILING: Frequency propagation should spread from peak to neighbors
        // Neighbors should get phases propagated from the peak using freq gradients
        // This test will fail because current implementation doesn't do heap-based propagation
        INFO("Current implementation does simple time integration, not heap-based propagation");
        REQUIRE(out_phases[14] != out_phases[15]);  // Should be different due to freq propagation
        REQUIRE(out_phases[16] != out_phases[15]);  // Should be different due to freq propagation
        
        // No cleanup needed with RAII
    }

    SECTION("Tolerance-based bin classification")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create magnitude distribution: max = 1.0, some below tolerance
        float max_mag = 1.0f;
        float abs_tolerance = tolerance * max_mag;  // 1e-6

        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i < 10)
            {
                mags[i] = 0.5f;  // Above tolerance
            }
            else if (i < 20)
            {
                mags[i] = max_mag;  // Maximum magnitude
            }
            else
            {
                mags[i] = abs_tolerance * 0.5f;  // Below tolerance
            }
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Bins below tolerance should get random phase, not gradient integration
        // Current implementation applies time gradient to all bins
        for (size_t i = 20; i < fft_bins; ++i)
        {
            INFO("Bin " << i << " is below tolerance, should have random phase");
            REQUIRE(out_phases[i] != Approx(0.1f).epsilon(1e-6));  // Should not be time gradient result
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Frequency-domain phase propagation")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create scenario where frequency propagation occurs
        // High magnitude in center, lower around it to trigger propagation
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i == 16)
            {
                mags[i] = 2.0f;  // High magnitude seed
            }
            else if (i >= 14 && i <= 18)
            {
                mags[i] = 1.0f;  // Medium magnitude for propagation
            }
            else
            {
                mags[i] = 0.005f;  // Low magnitude (below tolerance)
            }
            freq_grad[i] = 0.02f;  // Constant frequency gradient
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Check that high-magnitude bin (16) gets time integration
        REQUIRE(out_phases[16] == Approx(0.1f).epsilon(1e-6));

        // Check frequency propagation to adjacent significant bins
        float expected_15 = out_phases[16] - freq_grad[16];  // Downward propagation
        float expected_17 = out_phases[16] + freq_grad[16];  // Upward propagation

        INFO("Center bin (16) phase: " << out_phases[16]);
        INFO("Expected bin 15 phase: " << expected_15 << ", actual: " << out_phases[15]);
        INFO("Expected bin 17 phase: " << expected_17 << ", actual: " << out_phases[17]);

        // Allow some tolerance for phase wrapping
        REQUIRE(std::abs(out_phases[15] - expected_15) < 0.1f);
        REQUIRE(std::abs(out_phases[17] - expected_17) < 0.1f);
        
        // No cleanup needed with RAII
    }

    SECTION("Bidirectional frequency propagation")
    {
        std::vector<float> mags(fft_bins, 0.01f);  // Mostly low magnitude
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.0f);
        std::vector<float> freq_grad(fft_bins, 0.02f);  // Constant frequency gradient
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Single high-magnitude bin in the middle to seed propagation
        size_t seed_bin = fft_bins / 2;
        mags[seed_bin] = 1.0f;
        prev_phases[seed_bin] = pi / 4;  // Non-zero starting phase

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Phase should propagate both up and down from seed bin
        // Upward propagation (increasing frequency)
        for (size_t i = seed_bin; i < fft_bins - 1; ++i)
        {
            float expected_phase_up = out_phases[i] + freq_grad[i];
            INFO("Upward propagation from bin " << i << " to " << (i + 1));
            REQUIRE(out_phases[i + 1] == Approx(expected_phase_up).epsilon(1e-5));
        }

        // Downward propagation (decreasing frequency)
        for (size_t i = seed_bin; i > 0; --i)
        {
            float expected_phase_down = out_phases[i] - freq_grad[i - 1];
            INFO("Downward propagation from bin " << i << " to " << (i - 1));
            REQUIRE(out_phases[i - 1] == Approx(expected_phase_down).epsilon(1e-5));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Heap priority ordering")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create specific magnitude ordering to test heap priority
        std::vector<std::pair<size_t, float>> mag_order = {
            { 5, 1.0f },   // Highest priority
            { 10, 0.8f },  // Second priority
            { 15, 0.6f },  // Third priority
            { 20, 0.4f }   // Fourth priority
        };

        // Fill with low magnitudes, then set specific high values
        std::fill(mags.begin(), mags.end(), 0.01f);
        for (auto& pair : mag_order)
        {
            mags[pair.first] = pair.second;
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: High magnitude bins should be processed before low magnitude bins
        // This means their phases should reflect time integration first
        for (auto& pair : mag_order)
        {
            size_t bin = pair.first;
            float expected_phase = prev_phases[bin] + time_grad[bin];
            INFO("High-magnitude bin " << bin << " should be processed with time integration");
            REQUIRE(out_phases[bin] == Approx(expected_phase).epsilon(1e-6));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Random phase assignment for low-magnitude bins")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        float max_mag = 1.0f;
        float abs_tolerance = tolerance * max_mag;

        // Most bins below tolerance, few above
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i < 3)
            {
                mags[i] = max_mag;  // Above tolerance
            }
            else
            {
                mags[i] = abs_tolerance * 0.1f;  // Well below tolerance
            }
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Low-magnitude bins should have random phases, not gradient integration
        // Check that phases are in valid range [-π, π] and not the gradient result
        for (size_t i = 3; i < fft_bins; ++i)
        {
            INFO("Bin " << i << " should have random phase assignment");
            REQUIRE(out_phases[i] >= -pi);
            REQUIRE(out_phases[i] <= pi);
            // Should NOT be the simple time gradient result
            REQUIRE(out_phases[i] != Approx(0.1f).epsilon(1e-6));
        }

        // Verify random phases are actually different (not all the same)
        bool phases_differ = false;
        for (size_t i = 4; i < fft_bins; ++i)
        {
            if (std::abs(out_phases[i] - out_phases[3]) > 1e-6)
            {
                phases_differ = true;
                break;
            }
        }
        REQUIRE(phases_differ);  // Random phases should vary
        
        // No cleanup needed with RAII
    }

    SECTION("Forward Euler integration for time gradients")
    {
        std::vector<float> mags(fft_bins);  // Varying magnitudes
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Setup with isolated bins - only test a few bins to avoid frequency propagation
        std::fill(mags.begin(), mags.end(), 0.001f);  // Very low magnitude (below tolerance)
        std::fill(prev_phases.begin(), prev_phases.end(), 0.0f);
        std::fill(time_grad.begin(), time_grad.end(), 0.0f);

        // Only set up a few isolated bins for testing
        std::vector<size_t> test_bins = { 0, 10, 20, 30 };  // Well separated bins
        for (size_t idx = 0; idx < test_bins.size() && test_bins[idx] < fft_bins; ++idx)
        {
            size_t i = test_bins[idx];
            mags[i] = 1.0f;  // High magnitude (significant)
            prev_phases[i] = 0.1f * i;
            time_grad[i] = 0.05f * (i + 1);  // Varying time gradients
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (not needed for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Check forward Euler integration with principal argument wrapping for test bins only
        for (size_t idx = 0; idx < test_bins.size() && test_bins[idx] < fft_bins; ++idx)
        {
            size_t i = test_bins[idx];
            float simple_result = prev_phases[i] + time_grad[i];
            // Apply same principal argument wrapping as algorithm
            float wrapped_result = simple_result - 2.0f * pi * std::round(simple_result / (2.0f * pi));

            INFO("Bin " << i << " forward Euler: " << simple_result);
            INFO("Wrapped result: " << wrapped_result);
            INFO("Actual result: " << out_phases[i]);

            REQUIRE(out_phases[i] == Approx(wrapped_result).epsilon(1e-6));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Trapezoidal integration for time gradients")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> prev_time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Setup with isolated bins
        std::fill(mags.begin(), mags.end(), 0.001f);
        std::fill(prev_phases.begin(), prev_phases.end(), 0.0f);
        std::fill(time_grad.begin(), time_grad.end(), 0.0f);
        std::fill(prev_time_grad.begin(), prev_time_grad.end(), 0.0f);

        // Only set up a few isolated bins for testing
        std::vector<size_t> test_bins = { 0, 10, 20, 30 };
        for (size_t idx = 0; idx < test_bins.size() && test_bins[idx] < fft_bins; ++idx)
        {
            size_t i = test_bins[idx];
            mags[i] = 1.0f;  // High magnitude (significant)
            prev_phases[i] = 0.1f * i;
            prev_time_grad[i] = 0.03f * (i + 1);  // Previous time gradients
            time_grad[i] = 0.07f * (i + 1);       // Current time gradients (different from previous)
        }

        // Create processor configured for trapezoidal integration
        rtpghi::ProcessorConfig config(fft_bins, rtpghi::constants::DEFAULT_TOLERANCE, 
                                      rtpghi::constants::DEFAULT_RANDOM_SEED,
                                      rtpghi::IntegrationMethod::TRAPEZOIDAL);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),
                                   prev_phases.data(),
                                   time_grad.data(),
                                   freq_grad.data(),
                                   prev_time_grad.data(),  // Previous time gradients for trapezoidal integration
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Check trapezoidal integration: phase += 0.5 * (prev_grad + curr_grad)
        for (size_t idx = 0; idx < test_bins.size() && test_bins[idx] < fft_bins; ++idx)
        {
            size_t i = test_bins[idx];
            float grad_avg = 0.5f * (prev_time_grad[i] + time_grad[i]);
            float expected_result = prev_phases[i] + grad_avg;
            // Apply same principal argument wrapping as algorithm
            float wrapped_result = expected_result - 2.0f * pi * std::round(expected_result / (2.0f * pi));

            INFO("Bin " << i << " prev_grad: " << prev_time_grad[i] << " curr_grad: " << time_grad[i]);
            INFO("Average gradient: " << grad_avg);
            INFO("Expected result: " << expected_result);
            INFO("Wrapped result: " << wrapped_result);
            INFO("Actual result: " << out_phases[i]);

            REQUIRE(out_phases[i] == Approx(wrapped_result).epsilon(1e-6));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Integration method validation")
    {
        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Test trapezoidal integration without previous gradients (should fail validation)
        rtpghi::FrameInput invalid_input { mags.data(),      prev_phases.data(),
                                           time_grad.data(), freq_grad.data(),
                                           nullptr,  // No previous gradients provided
                                           fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        // Create processor configured for trapezoidal integration
        rtpghi::ProcessorConfig config(fft_bins, rtpghi::constants::DEFAULT_TOLERANCE,
                                     rtpghi::constants::DEFAULT_RANDOM_SEED,
                                     rtpghi::IntegrationMethod::TRAPEZOIDAL);
        rtpghi::Processor processor(config);

        // Should return INVALID_INPUT because trapezoidal needs previous gradients
        REQUIRE(processor.process(invalid_input, output) == rtpghi::ErrorCode::INVALID_INPUT);

        // Test forward Euler without previous gradients (should succeed)
        rtpghi::FrameInput valid_input { mags.data(),      prev_phases.data(),
                                         time_grad.data(), freq_grad.data(),
                                         nullptr,  // No previous gradients needed for forward Euler
                                         fft_bins };

        // Create a new processor configured for Forward Euler (default)
        rtpghi::ProcessorConfig euler_config(fft_bins);
        rtpghi::Processor euler_processor(euler_config);

        REQUIRE(euler_processor.process(valid_input, output) == rtpghi::ErrorCode::OK);
        
        // No cleanup needed with RAII
    }

    SECTION("Phase propagation order verification")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.0f);
        std::vector<float> freq_grad(fft_bins, 0.1f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create magnitude pattern: two peaks for testing propagation order
        std::fill(mags.begin(), mags.end(), 0.01f);
        mags[8] = 1.0f;   // First peak (higher magnitude)
        mags[24] = 0.9f;  // Second peak (lower magnitude)

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Higher magnitude bin should be processed first, then propagate to neighbors
        // The algorithm should process bin 8 first (magnitude 1.0) then bin 24 (magnitude 0.9)
        // This affects the final phase values due to different propagation paths

        // Bin 8 should be processed first with time integration (0 + 0 = 0)
        REQUIRE(out_phases[8] == Approx(0.0f).epsilon(1e-6));

        // Bin 24 should be processed after bin 8, but current implementation doesn't show this
        // In real RTPGHI, the processing order would affect neighboring bins differently
        INFO("Current implementation processes all bins independently");
        INFO("Real RTPGHI would show propagation order effects");
        
        // No cleanup needed with RAII
    }
}

TEST_CASE("RTPGHI Complex Algorithm Scenarios", "[algorithm][failing][complex]")
{
    const size_t fft_bins = 64;
    const float tolerance = 1e-5f;

    SECTION("Multi-peak magnitude distribution")
    {
        std::vector<float> mags(fft_bins, 0.01f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.05f);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Multiple peaks at different magnitudes
        mags[10] = 1.0f;  // Highest peak
        mags[30] = 0.8f;  // Second peak
        mags[50] = 0.6f;  // Third peak

        // Frequency gradients that vary across spectrum
        for (size_t i = 0; i < fft_bins; ++i)
        {
            freq_grad[i] = 0.02f * std::sin(2.0f * 3.14159f * i / fft_bins);
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Multiple peaks should create different propagation patterns
        // Phase at each peak should reflect time integration
        REQUIRE(out_phases[10] == Approx(0.05f).epsilon(1e-6));  // Highest peak processed first
        REQUIRE(out_phases[30] == Approx(0.05f).epsilon(1e-6));  // Second peak
        REQUIRE(out_phases[50] == Approx(0.05f).epsilon(1e-6));  // Third peak

        // FAILING: Regions between peaks should show frequency propagation effects
        // This will fail because current implementation doesn't do inter-peak propagation
        INFO("Real RTPGHI would show complex propagation patterns between peaks");
        
        // No cleanup needed with RAII
    }

    SECTION("Phase unwrapping and principal argument")
    {
        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins, 3.0f);  // Large gradient causing phase wrap
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Previous phases near 2π boundary
        const float pi = 3.14159265359f;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            prev_phases[i] = 2.0f * pi - 0.1f;  // Near 2π
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: Phases should be wrapped to [-π, π] using principal argument
        for (size_t i = 0; i < fft_bins; ++i)
        {
            float raw_phase = prev_phases[i] + time_grad[i];  // ≈ 2π - 0.1 + 3.0 ≈ 8.18
            float expected_wrapped = raw_phase - 2.0f * pi * std::round(raw_phase / (2.0f * pi));

            INFO("Bin " << i << " raw phase: " << raw_phase);
            INFO("Expected wrapped phase: " << expected_wrapped);
            INFO("Actual phase: " << out_phases[i]);

            REQUIRE(out_phases[i] >= -pi);
            REQUIRE(out_phases[i] <= pi);
            REQUIRE(out_phases[i] == Approx(expected_wrapped).epsilon(1e-5));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Extreme magnitude ratios")
    {
        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Extreme dynamic range: very high peak, very low background
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i == fft_bins / 2)
            {
                mags[i] = 1000.0f;  // Very high peak
            }
            else
            {
                mags[i] = 1e-8f;  // Very low background
            }
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // FAILING: With extreme dynamic range, most bins should get random phase
        float max_mag = *std::max_element(mags.begin(), mags.end());
        float abs_tolerance = tolerance * max_mag;  // 1e-5 * 1000 = 0.01

        size_t significant_bin_count = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (mags[i] > abs_tolerance)
            {
                significant_bin_count++;
            }
        }

        INFO("Maximum magnitude: " << max_mag);
        INFO("Absolute tolerance: " << abs_tolerance);
        INFO("Significant bins: " << significant_bin_count);

        // Only the peak bin should be significant
        REQUIRE(significant_bin_count == 1);

        // FAILING: Non-significant bins should not have gradient-integrated phases
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i != fft_bins / 2)
            {  // Not the peak bin
                INFO("Bin " << i << " magnitude " << mags[i] << " is below tolerance");
                REQUIRE(out_phases[i] != Approx(0.1f).epsilon(1e-6));  // Should not be time gradient
            }
        }
        
        // No cleanup needed with RAII
    }
}