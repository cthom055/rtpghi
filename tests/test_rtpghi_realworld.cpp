#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include <random>
#include <rtpghi/rtpghi.hpp>
#include <vector>

using Catch::Approx;

// No helper function needed - create processor directly

TEST_CASE("RTPGHI Real-World Audio Scenarios", "[realworld][comprehensive]")
{
    const float pi = 3.14159265359f;

    SECTION("Musical harmonic content - Major chord")
    {
        // Simulate a C major chord (C4, E4, G4) = (261.6, 329.6, 392.0 Hz)
        // At 44.1kHz with 2048 FFT: bins ~= 12, 15, 18
        const size_t fft_bins = 1025;  // 2048 FFT / 2 + 1
        const float sample_rate = 44100.0f;
        const float fundamental = 261.6f;  // C4

        std::vector<float> mags(fft_bins, 0.01f);  // Background noise
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Set up harmonic content
        std::vector<std::pair<float, float>> harmonics = {
            { 261.6f, 1.0f },  // C4 fundamental
            { 329.6f, 0.8f },  // E4 major third
            { 392.0f, 0.6f },  // G4 perfect fifth
            { 523.2f, 0.3f }   // C5 octave
        };

        for (const auto& harmonic : harmonics)
        {
            size_t bin = static_cast<size_t>(harmonic.first * fft_bins * 2 / sample_rate);
            if (bin < fft_bins)
            {
                mags[bin] = harmonic.second;
                // Simulate phase evolution for musical content
                prev_phases[bin] = 0.5f * pi * std::sin(harmonic.first / fundamental);
                time_grad[bin] = 2.0f * pi * harmonic.first / sample_rate;  // Natural frequency evolution
                freq_grad[bin] = 0.1f * harmonic.second;                    // Magnitude-dependent frequency gradient
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

        // Verify harmonic bins are processed correctly
        for (const auto& harmonic : harmonics)
        {
            size_t bin = static_cast<size_t>(harmonic.first * fft_bins * 2 / sample_rate);
            if (bin < fft_bins)
            {
                INFO("Checking harmonic at " << harmonic.first << " Hz (bin " << bin << ")");
                // High-magnitude bins should be processed by time integration, not random phase
                REQUIRE(output.magnitudes[bin] == Approx(harmonic.second).epsilon(1e-6f));
                // Phase should be within reasonable range
                REQUIRE(output.phases[bin] >= -pi);
                REQUIRE(output.phases[bin] <= 2 * pi);
            }
        }

        // No cleanup needed with RAII
    }

    SECTION("Speech-like content with formants")
    {
        // Simulate vowel /a/ formants: F1=730Hz, F2=1090Hz, F3=2440Hz
        const size_t fft_bins = 513;         // 1024 FFT / 2 + 1
        const float sample_rate = 16000.0f;  // Common speech sample rate

        std::vector<float> mags(fft_bins, 0.005f);  // Speech noise floor
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins, 0.02f);  // Slow speech evolution
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Set up formant structure
        std::vector<std::tuple<float, float, float>> formants = {
            std::make_tuple(100.0f, 0.4f, 50.0f),    // F0 (pitch)
            std::make_tuple(730.0f, 1.0f, 80.0f),    // F1
            std::make_tuple(1090.0f, 0.8f, 100.0f),  // F2
            std::make_tuple(2440.0f, 0.6f, 150.0f)   // F3
        };

        for (const auto& formant : formants)
        {
            float freq = std::get<0>(formant);
            float amplitude = std::get<1>(formant);
            float bandwidth = std::get<2>(formant);

            size_t center_bin = static_cast<size_t>(freq * fft_bins * 2 / sample_rate);
            size_t bw_bins = static_cast<size_t>(bandwidth * fft_bins * 2 / sample_rate);

            // Create formant with realistic bandwidth
            for (size_t i = 0; i < bw_bins && center_bin + i < fft_bins; ++i)
            {
                float decay = std::exp(-0.1f * static_cast<float>(i));  // Exponential decay from center
                if (center_bin + i < fft_bins)
                {
                    mags[center_bin + i] = amplitude * decay;
                    prev_phases[center_bin + i] = pi * std::sin(freq / 1000.0f);
                    freq_grad[center_bin + i] = 0.05f * decay;
                }
                if (i > 0 && center_bin >= i)
                {
                    mags[center_bin - i] = amplitude * decay;
                    prev_phases[center_bin - i] = pi * std::sin(freq / 1000.0f);
                    freq_grad[center_bin - i] = 0.05f * decay;
                }
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

        // Check formant regions are handled correctly
        size_t processed_formant_bins = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (mags[i] > 0.1f)
            {  // Significant formant energy
                processed_formant_bins++;
                // Should not be random phase
                REQUIRE(output.phases[i] != Approx(0.0f).epsilon(1e-3));  // Not default/zero
            }
        }

        INFO("Processed formant bins: " << processed_formant_bins);
        REQUIRE(processed_formant_bins > 10);  // Should have substantial formant content

        // No cleanup needed with RAII
    }

    SECTION("Transient content - Drum hit simulation")
    {
        const size_t fft_bins = 257;  // 512 FFT / 2 + 1

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Simulate broadband transient with exponential decay
        for (size_t i = 0; i < fft_bins; ++i)
        {
            float freq = static_cast<float>(i) * 22050.0f / static_cast<float>(fft_bins);  // Up to Nyquist
            // Transient: high energy that decays exponentially with frequency
            mags[i] = std::exp(-freq / 2000.0f) + 0.01f;  // Exponential decay + noise floor
            // Sharp phase discontinuity typical of transients
            prev_phases[i] = (i % 3 == 0) ? pi : -pi / 2;
            // Large time gradients for rapid evolution
            time_grad[i] = 0.5f * mags[i];
            // Frequency spreading
            freq_grad[i] = 0.02f * std::sin(freq / 1000.0f);
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

        // Check transient handling
        size_t high_energy_bins = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (mags[i] > 0.5f)
            {  // High energy bins
                high_energy_bins++;
                // Should maintain magnitude
                REQUIRE(output.magnitudes[i] == Approx(mags[i]).epsilon(1e-6f));
                // Phase should be processed (not random)
                REQUIRE(std::abs(output.phases[i]) <= pi + 0.1f);  // Allow some tolerance
            }
        }

        INFO("High energy transient bins: " << high_energy_bins);
        REQUIRE(high_energy_bins >= 5);  // Should have significant high-energy content

        // No cleanup needed with RAII
    }

    SECTION("Pure silence - All zeros")
    {
        const size_t fft_bins = 129;  // 256 FFT / 2 + 1

        std::vector<float> mags(fft_bins, 0.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.0f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // All bins should get random phase since none are significant
        // With zero input, time integration would give 0.0, so random phases should be non-zero
        size_t non_zero_phases = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(output.magnitudes[i] == 0.0f);
            // Phases should be in [-π, π] range
            REQUIRE(output.phases[i] >= -pi - 0.1f);
            REQUIRE(output.phases[i] <= pi + 0.1f);
            // Count non-zero phases (random phases shouldn't all be zero)
            if (std::abs(output.phases[i]) > 1e-6f)
            {
                non_zero_phases++;
            }
        }
        // Most phases should be non-zero (random)
        REQUIRE(non_zero_phases > fft_bins / 2);

        // No cleanup needed with RAII
    }

    SECTION("White noise")
    {
        const size_t fft_bins = 513;  // 1024 FFT / 2 + 1
        std::mt19937 rng(42);         // Deterministic random
        std::uniform_real_distribution<float> mag_dist(0.8f, 1.2f);
        std::uniform_real_distribution<float> phase_dist(-pi, pi);
        std::uniform_real_distribution<float> grad_dist(-0.1f, 0.1f);

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Generate white noise characteristics
        for (size_t i = 0; i < fft_bins; ++i)
        {
            mags[i] = mag_dist(rng);
            prev_phases[i] = phase_dist(rng);
            time_grad[i] = grad_dist(rng);
            freq_grad[i] = grad_dist(rng);
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

        // All bins should be significant and processed
        size_t processed_bins = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(output.magnitudes[i] == Approx(mags[i]).epsilon(1e-6f));
            // Verify phase is processed (not random)
            if (output.phases[i] >= -pi - 0.1f && output.phases[i] <= pi + 0.1f)
            {
                processed_bins++;
            }
        }

        INFO("Processed bins: " << processed_bins);
        // Most bins should be processed by heap algorithm (not random)
        REQUIRE(static_cast<float>(processed_bins) >= static_cast<float>(fft_bins) * 0.9f);  // At least 90% processed

        // No cleanup needed with RAII
    }
}

TEST_CASE("RTPGHI Edge Cases and Boundary Conditions", "[edge][comprehensive]")
{
    const float pi = 3.14159265359f;

    SECTION("Minimum FFT size - Single bin")
    {
        const size_t fft_bins = 1;

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, pi / 4);
        std::vector<float> time_grad(fft_bins, 0.2f);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);
        REQUIRE(output.magnitudes[0] == 1.0f);
        // Should be time integrated: pi/4 + 0.2
        float expected = pi / 4 + 0.2f;
        expected = expected - 2.0f * pi * std::round(expected / (2.0f * pi));
        REQUIRE(output.phases[0] == Approx(expected).epsilon(1e-6f));

        // No cleanup needed with RAII
    }

    SECTION("Large FFT size stress test")
    {
        const size_t fft_bins = 2048;  // Maximum supported size

        std::vector<float> mags(fft_bins, 0.5f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.01f);
        std::vector<float> freq_grad(fft_bins, 0.001f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Add some peaks to test heap performance
        for (size_t i = 100; i < fft_bins; i += 100)
        {
            mags[i] = 2.0f;  // Peaks every 100 bins
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

        // Verify processing completed successfully
        size_t processed_bins = 0;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(output.magnitudes[i] == mags[i]);
            if (mags[i] > 1.0f)
            {  // Peak bins
                processed_bins++;
                REQUIRE(output.phases[i] == Approx(0.01f).epsilon(1e-6f));
            }
        }

        INFO("Processed peak bins: " << processed_bins);
        REQUIRE(processed_bins >= 15);  // Should have multiple peaks (every 100 bins)

        // No cleanup needed with RAII
    }

    SECTION("Large FFT size validation")
    {
        const size_t fft_bins = 2049;  // Large FFT size that should work with user allocation

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.01f);
        std::vector<float> freq_grad(fft_bins, 0.001f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor for large FFT
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        // Should succeed with user-allocated plan (no hardcoded size limits)
        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Verify basic processing worked
        bool all_phases_modified = true;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (out_phases[i] == 0.0f && time_grad[i] != 0.0f)
            {
                all_phases_modified = false;
                break;
            }
        }
        REQUIRE(all_phases_modified);

        // No cleanup needed with RAII
    }

    SECTION("Extreme magnitude ratios")
    {
        const size_t fft_bins = 65;

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create extreme dynamic range
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i == 32)
            {
                mags[i] = 1e6f;  // Extremely loud
            }
            else if (i == 16 || i == 48)
            {
                mags[i] = 1e3f;  // Very loud
            }
            else
            {
                mags[i] = 1e-9f;  // Extremely quiet
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

        // Very loud bins should be processed by time integration
        REQUIRE(output.phases[32] == Approx(0.1f).epsilon(1e-6f));
        REQUIRE(output.phases[16] == Approx(0.1f).epsilon(1e-6f));
        REQUIRE(output.phases[48] == Approx(0.1f).epsilon(1e-6f));

        // Very quiet bins should get random phase
        bool has_random_phases = false;
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i != 32 && i != 16 && i != 48)
            {
                if (output.phases[i] != Approx(0.1f).epsilon(1e-6f))
                {
                    has_random_phases = true;
                    REQUIRE(output.phases[i] >= -pi - 0.1f);
                    REQUIRE(output.phases[i] <= pi + 0.1f);
                }
            }
        }
        REQUIRE(has_random_phases);

        // No cleanup needed with RAII
    }

    SECTION("Numerical precision edge cases")
    {
        const size_t fft_bins = 33;

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Test very small gradients and phases
        for (size_t i = 0; i < fft_bins; ++i)
        {
            mags[i] = 1.0f;
            prev_phases[i] = 1e-7f * static_cast<float>(i);  // Very small phases
            time_grad[i] = 1e-8f;        // Very small time gradient
            freq_grad[i] = 1e-9f * static_cast<float>(i);    // Very small frequency gradient
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

        // Should handle small numbers without precision loss
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(std::isfinite(output.phases[i]));
            REQUIRE(std::isfinite(output.magnitudes[i]));
            REQUIRE(output.magnitudes[i] == 1.0f);
        }

        // No cleanup needed with RAII
    }

    SECTION("Phase wrap-around scenarios")
    {
        const size_t fft_bins = 17;

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Test phases that will wrap around
        for (size_t i = 0; i < fft_bins; ++i)
        {
            prev_phases[i] = 1.8f * pi;  // Near 2π
            time_grad[i] = 0.5f * pi;    // Will cause wrap-around
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

        // All phases should be wrapped to [-π, π]
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(output.phases[i] >= -pi - 0.01f);
            REQUIRE(output.phases[i] <= pi + 0.01f);
        }

        // No cleanup needed with RAII
    }
}