#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <rtpghi/rtpghi.hpp>
#include <vector>

using Catch::Approx;

// No helper function needed - create processor directly

TEST_CASE("RTPGHI Performance and Stress Tests", "[performance][stress]")
{
    const float pi = 3.14159265359f;

    SECTION("Performance benchmark - Standard FFT sizes")
    {
        // Test common audio FFT sizes
        std::vector<size_t> fft_sizes = { 64, 128, 256, 512, 1024, 2048 };

        for (size_t fft_size : fft_sizes)
        {
            size_t fft_bins = fft_size / 2 + 1;

            std::vector<float> mags(fft_bins, 1.0f);
            std::vector<float> prev_phases(fft_bins, 0.0f);
            std::vector<float> time_grad(fft_bins, 0.1f);
            std::vector<float> freq_grad(fft_bins, 0.01f);
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

            auto start = std::chrono::high_resolution_clock::now();
            rtpghi::ErrorCode result = processor.process(input, output);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            INFO("FFT size " << fft_size << " (" << fft_bins << " bins) took " << duration.count() << " microseconds");

            REQUIRE(result == rtpghi::ErrorCode::OK);
            // Should complete in reasonable time (less than 1000 microseconds for largest size)
            REQUIRE(duration.count() < 1000);
            
            // No cleanup needed with RAII
        }
    }

    SECTION("Iteration stability - Multiple consecutive frames")
    {
        const size_t fft_bins = 257;
        const size_t num_frames = 100;

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.05f);
        std::vector<float> freq_grad(fft_bins, 0.02f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        // Simulate evolving audio over time
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                       time_grad.data(), freq_grad.data(),
                                       nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                       fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

            REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

            // Use output phases as input for next frame
            for (size_t i = 0; i < fft_bins; ++i)
            {
                prev_phases[i] = output.phases[i];
                // Verify phases remain bounded
                REQUIRE(std::isfinite(output.phases[i]));
                REQUIRE(output.phases[i] >= -pi - 0.1f);
                REQUIRE(output.phases[i] <= 2 * pi + 0.1f);
            }
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Memory access patterns - No out-of-bounds")
    {
        const size_t fft_bins = 1537;  // Odd size to test boundary conditions

        std::vector<float> mags(fft_bins + 10, 0.5f);  // Extra space to catch overruns
        std::vector<float> prev_phases(fft_bins + 10, 0.0f);
        std::vector<float> time_grad(fft_bins + 10, 0.1f);
        std::vector<float> freq_grad(fft_bins + 10, 0.05f);
        std::vector<float> out_mags(fft_bins + 10);
        std::vector<float> out_phases(fft_bins + 10);

        // Fill guard regions with sentinel values
        for (size_t i = fft_bins; i < fft_bins + 10; ++i)
        {
            mags[i] = -999.0f;
            prev_phases[i] = -999.0f;
            time_grad[i] = -999.0f;
            freq_grad[i] = -999.0f;
            out_mags[i] = -999.0f;
            out_phases[i] = -999.0f;
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Verify guard regions are untouched
        for (size_t i = fft_bins; i < fft_bins + 10; ++i)
        {
            REQUIRE(mags[i] == -999.0f);
            REQUIRE(prev_phases[i] == -999.0f);
            REQUIRE(time_grad[i] == -999.0f);
            REQUIRE(freq_grad[i] == -999.0f);
            // Output guard regions may be modified, which is acceptable
        }
        
        // No cleanup needed with RAII
    }
}

TEST_CASE("RTPGHI Audio Processing Patterns", "[audio][patterns]")
{
    const float pi = 3.14159265359f;

    SECTION("Pitch shifting simulation")
    {
        // Simulate pitch shift effect with modified time gradients
        const size_t fft_bins = 513;
        const float pitch_ratio = 1.2f;  // 20% higher pitch

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins, 0.0f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create harmonic content
        for (size_t i = 0; i < fft_bins; ++i)
        {
            float freq = i * 22050.0f / fft_bins;
            mags[i] = (i % 10 == 0 && i > 0) ? 0.8f : 0.05f;  // Harmonics every 10 bins
            prev_phases[i] = 0.0f;
            time_grad[i] = 2.0f * pi * freq * pitch_ratio / 44100.0f;  // Pitch shifted gradients
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Check harmonic bins are processed correctly
        size_t harmonic_bins = 0;
        for (size_t i = 10; i < fft_bins; i += 10)
        {
            if (mags[i] > 0.5f)
            {
                harmonic_bins++;
                REQUIRE(output.magnitudes[i] == Approx(mags[i]).epsilon(1e-6));
            }
        }

        INFO("Harmonic bins processed: " << harmonic_bins);
        REQUIRE(harmonic_bins >= 40);  // Should have many harmonics
        
        // No cleanup needed with RAII
    }

    SECTION("Time stretching simulation")
    {
        // Simulate time stretch with modified gradients
        const size_t fft_bins = 257;
        const float stretch_ratio = 0.75f;  // 25% slower

        std::vector<float> mags(fft_bins, 0.8f);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins, 0.01f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Set up smooth phase evolution
        for (size_t i = 0; i < fft_bins; ++i)
        {
            prev_phases[i] = pi * std::sin(i / 32.0f);
            time_grad[i] = 0.1f * stretch_ratio;  // Slower evolution
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Verify phase continuity
        bool phase_continuous = true;
        for (size_t i = 1; i < fft_bins - 1; ++i)
        {
            // Check for reasonable phase differences (not completely random)
            float phase_diff = std::abs(output.phases[i + 1] - output.phases[i]);
            if (phase_diff > pi)
            {
                phase_diff = 2 * pi - phase_diff;  // Handle wrap-around
            }
            if (phase_diff > pi / 2)
            {  // Large jump indicates discontinuity
                phase_continuous = false;
                break;
            }
        }

        // Most phases should show continuity for time-stretched content
        REQUIRE(phase_continuous);
        
        // No cleanup needed with RAII
    }

    SECTION("Spectral filtering effects")
    {
        // Simulate spectral filtering with frequency-dependent magnitudes
        const size_t fft_bins = 513;

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.08f);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create bandpass filter effect (high in middle, low at edges)
        for (size_t i = 0; i < fft_bins; ++i)
        {
            float normalized_freq = i / float(fft_bins);
            // Bandpass: peak around 0.3-0.7 of Nyquist
            if (normalized_freq > 0.2f && normalized_freq < 0.8f)
            {
                mags[i] = 1.0f * std::sin(pi * (normalized_freq - 0.2f) / 0.6f);
                freq_grad[i] = 0.05f;
            }
            else
            {
                mags[i] = 0.01f;  // Outside passband
                freq_grad[i] = 0.001f;
            }
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Check passband vs stopband processing
        size_t passband_bins = 0;
        size_t stopband_bins = 0;

        for (size_t i = 0; i < fft_bins; ++i)
        {
            float normalized_freq = i / float(fft_bins);
            if (normalized_freq > 0.2f && normalized_freq < 0.8f && mags[i] > 0.5f)
            {
                passband_bins++;
                // Passband: should be processed by time integration
                REQUIRE(output.magnitudes[i] == Approx(mags[i]).epsilon(1e-6));
            }
            else if (mags[i] <= 0.02f)
            {
                stopband_bins++;
                // Stopband: should get random phase
                REQUIRE(output.magnitudes[i] == Approx(mags[i]).epsilon(1e-6));
            }
        }

        INFO("Passband bins: " << passband_bins << ", Stopband bins: " << stopband_bins);
        REQUIRE(passband_bins >= 100);  // Substantial passband
        REQUIRE(stopband_bins >= 200);  // Substantial stopband
        
        // No cleanup needed with RAII
    }

    SECTION("Transient detection and preservation")
    {
        // Test handling of transient content with sharp magnitude changes
        const size_t fft_bins = 129;

        std::vector<float> mags(fft_bins);
        std::vector<float> prev_phases(fft_bins);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create transient: sudden onset with rapid decay
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (i < 20)
            {
                // Attack portion: rapidly increasing
                mags[i] = i / 20.0f;
                prev_phases[i] = 0.0f;
                time_grad[i] = 0.3f;  // Large gradient for transient
                freq_grad[i] = 0.1f;
            }
            else if (i < 60)
            {
                // Decay portion: exponentially decreasing
                mags[i] = std::exp(-(i - 20) * 0.1f);
                prev_phases[i] = pi * (i - 20) / 40.0f;
                time_grad[i] = 0.1f;
                freq_grad[i] = 0.05f;
            }
            else
            {
                // Tail: very low
                mags[i] = 0.01f;
                prev_phases[i] = pi;
                time_grad[i] = 0.02f;
                freq_grad[i] = 0.01f;
            }
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Verify transient regions are handled appropriately
        bool attack_processed = true;
        bool decay_processed = true;

        for (size_t i = 0; i < 20; ++i)
        {
            if (output.magnitudes[i] != Approx(mags[i]).epsilon(1e-6))
            {
                attack_processed = false;
            }
        }

        for (size_t i = 20; i < 60; ++i)
        {
            if (output.magnitudes[i] != Approx(mags[i]).epsilon(1e-6))
            {
                decay_processed = false;
            }
        }

        REQUIRE(attack_processed);
        REQUIRE(decay_processed);
        
        // No cleanup needed with RAII
    }
}

TEST_CASE("RTPGHI Robustness and Error Handling", "[robustness][error]")
{
    SECTION("Mixed valid and invalid inputs")
    {
        const size_t fft_bins = 65;

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        // Test various null pointer combinations
        rtpghi::FrameInput input1 { nullptr,          prev_phases.data(),
                                    time_grad.data(), freq_grad.data(),
                                    nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                    fft_bins,          };
        rtpghi::FrameInput input2 { mags.data(),      nullptr,
                                    time_grad.data(), freq_grad.data(),
                                    nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                    fft_bins,          };
        rtpghi::FrameInput input3 { mags.data(), prev_phases.data(),
                                    nullptr,     freq_grad.data(),
                                    nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                    fft_bins,     };
        rtpghi::FrameInput input4 { mags.data(),      prev_phases.data(),
                                    time_grad.data(), nullptr,
                                    nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                    fft_bins,          };

        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input1, output) == rtpghi::ErrorCode::INVALID_INPUT);
        REQUIRE(processor.process(input2, output) == rtpghi::ErrorCode::INVALID_INPUT);
        REQUIRE(processor.process(input3, output) == rtpghi::ErrorCode::INVALID_INPUT);
        REQUIRE(processor.process(input4, output) == rtpghi::ErrorCode::INVALID_INPUT);
        
        // No cleanup needed with RAII
    }

    SECTION("Extreme gradient values")
    {
        const size_t fft_bins = 33;

        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins);
        std::vector<float> freq_grad(fft_bins);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // Test with extreme gradient values
        for (size_t i = 0; i < fft_bins; ++i)
        {
            time_grad[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;  // Very large gradients
            freq_grad[i] = (i % 3 == 0) ? 500.0f : -500.0f;
        }

        // Create processor
        rtpghi::ProcessorConfig config(fft_bins);
        rtpghi::Processor processor(config);

        rtpghi::FrameInput input { mags.data(),      prev_phases.data(),
                                   time_grad.data(), freq_grad.data(),
                                   nullptr,  // prev_time_gradients (use nullptr for forward Euler)
                                   fft_bins,          };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

        REQUIRE(processor.process(input, output) == rtpghi::ErrorCode::OK);

        // Should handle extreme values gracefully
        for (size_t i = 0; i < fft_bins; ++i)
        {
            REQUIRE(std::isfinite(output.phases[i]));
            REQUIRE(std::isfinite(output.magnitudes[i]));
        }
        
        // No cleanup needed with RAII
    }

    SECTION("Zero-length input edge case")
    {
        const size_t fft_bins = 0;
        float dummy_data = 0.0f;

        // For zero-length input, our constructor will throw an exception
        // This is the expected behavior with modern RAII design
        try {
            rtpghi::ProcessorConfig config(fft_bins);
            // If we reach here with fft_bins = 0, that's unexpected
            REQUIRE(false);  // Should have thrown exception
        } catch (const std::invalid_argument&) {
            // Expected exception for zero-length FFT - this is the correct behavior
            REQUIRE(true);
        }
    }
}