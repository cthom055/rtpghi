#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <rtpghi/rtpghi.hpp>
#include <vector>

// No helper function needed - create processor directly

TEST_CASE("RTPGHI Performance Benchmarks", "[!benchmark]")
{
    const float pi = 3.14159265359f;
    std::mt19937 rng(12345);  // Deterministic random for consistent benchmarks
    std::uniform_real_distribution<float> mag_dist(0.1f, 2.0f);
    std::uniform_real_distribution<float> phase_dist(-pi, pi);
    std::uniform_real_distribution<float> grad_dist(-0.5f, 0.5f);

    SECTION("RTPGHI Processing - Various FFT Sizes")
    {
        // Test with 512 FFT (257 bins)
        BENCHMARK("RTPGHI 512 FFT")
        {
            const size_t fft_bins = 257;
            std::vector<float> mags(fft_bins);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            for (size_t i = 0; i < fft_bins; ++i)
            {
                mags[i] = mag_dist(rng);
                prev_phases[i] = phase_dist(rng);
                time_grad[i] = grad_dist(rng);
                freq_grad[i] = grad_dist(rng) * 0.1f;
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // Test with 1024 FFT (513 bins)
        BENCHMARK("RTPGHI 1024 FFT")
        {
            const size_t fft_bins = 513;
            std::vector<float> mags(fft_bins);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            for (size_t i = 0; i < fft_bins; ++i)
            {
                mags[i] = mag_dist(rng);
                prev_phases[i] = phase_dist(rng);
                time_grad[i] = grad_dist(rng);
                freq_grad[i] = grad_dist(rng) * 0.1f;
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // Test with 2048 FFT (1025 bins)
        BENCHMARK("RTPGHI 2048 FFT")
        {
            const size_t fft_bins = 1025;
            std::vector<float> mags(fft_bins);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            for (size_t i = 0; i < fft_bins; ++i)
            {
                mags[i] = mag_dist(rng);
                prev_phases[i] = phase_dist(rng);
                time_grad[i] = grad_dist(rng);
                freq_grad[i] = grad_dist(rng) * 0.1f;
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };

            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };
    }

    SECTION("Gradient Calculation Methods")
    {
        const size_t size = 1024;
        std::vector<float> phases(size);
        std::vector<float> gradients(size);

        for (size_t i = 0; i < size; ++i)
        {
            phases[i] = phase_dist(rng);
        }

        BENCHMARK("Gradient Forward Difference 1024")
        {
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::FORWARD };
            return rtpghi::calculate_gradients(input, gradients.data());
        };

        BENCHMARK("Gradient Backward Difference 1024")
        {
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::BACKWARD };
            return rtpghi::calculate_gradients(input, gradients.data());
        };

        BENCHMARK("Gradient Central Difference 1024")
        {
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::CENTRAL };
            return rtpghi::calculate_gradients(input, gradients.data());
        };
    }

    SECTION("Integration Methods")
    {
        const size_t fft_bins = 513;
        std::vector<float> mags(fft_bins, 1.0f);
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> prev_time_grad(fft_bins, 0.08f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        BENCHMARK("Forward Euler Integration")
        {
            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        BENCHMARK("Trapezoidal Integration")
        {
            // Create processor configured for trapezoidal integration
            rtpghi::ProcessorConfig config(fft_bins, rtpghi::constants::DEFAULT_TOLERANCE,
                                         rtpghi::constants::DEFAULT_RANDOM_SEED,
                                         rtpghi::IntegrationMethod::TRAPEZOIDAL);
            rtpghi::Processor processor(config);

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      prev_time_grad.data(), fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };
    }

    SECTION("Memory Access Patterns")
    {
        const size_t fft_bins = 1024;
        std::vector<float> prev_phases(fft_bins, 0.0f);
        std::vector<float> time_grad(fft_bins, 0.1f);
        std::vector<float> freq_grad(fft_bins, 0.05f);
        std::vector<float> out_mags(fft_bins);
        std::vector<float> out_phases(fft_bins);

        // All bins significant (sequential access)
        BENCHMARK("Sequential Memory Access")
        {
            std::vector<float> mags(fft_bins, 1.0f);  // All significant
            
            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);
            
            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // Sparse significant bins
        BENCHMARK("Sparse Memory Access")
        {
            std::vector<float> mags(fft_bins, 0.001f);
            for (size_t i = 0; i < fft_bins; i += 10)
            {
                mags[i] = 1.0f;  // Every 10th bin significant
            }
            
            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);
            
            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // Mostly random phase assignment
        BENCHMARK("Random Phase Assignment")
        {
            std::vector<float> mags(fft_bins, 0.0001f);
            for (size_t i = 0; i < 10; ++i)
            {
                mags[i] = 1.0f;  // Only first 10 bins significant
            }
            
            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);
            
            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };
    }

    SECTION("Real-world Audio Scenarios")
    {
        const size_t fft_bins = 513;
        const float sample_rate = 44100.0f;

        // Musical content with harmonics
        BENCHMARK("Musical Content Processing")
        {
            std::vector<float> mags(fft_bins, 0.05f);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins, 0.01f);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            // Add harmonic content
            for (size_t i = 0; i < fft_bins; ++i)
            {
                float freq = static_cast<float>(i) * sample_rate / (2 * static_cast<float>(fft_bins));
                if (i > 0 && (i % 20 == 0 || i % 33 == 0))
                {
                    mags[i] = 0.8f * std::exp(-freq / 8000.0f);
                }
                prev_phases[i] = 2.0f * pi * freq * 0.01f;
                time_grad[i] = 2.0f * pi * freq / sample_rate;
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // Transient content
        BENCHMARK("Transient Content Processing")
        {
            std::vector<float> mags(fft_bins);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            // Broadband transient with exponential decay
            for (size_t i = 0; i < fft_bins; ++i)
            {
                float freq = static_cast<float>(i) * sample_rate / (2 * static_cast<float>(fft_bins));
                mags[i] = std::exp(-freq / 2000.0f) + 0.01f;
                prev_phases[i] = (i % 3 == 0) ? pi : -pi / 2;
                time_grad[i] = 0.5f * mags[i];
                freq_grad[i] = 0.02f * std::sin(freq / 1000.0f);
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };

        // White noise
        BENCHMARK("White Noise Processing")
        {
            std::vector<float> mags(fft_bins);
            std::vector<float> prev_phases(fft_bins);
            std::vector<float> time_grad(fft_bins);
            std::vector<float> freq_grad(fft_bins);
            std::vector<float> out_mags(fft_bins);
            std::vector<float> out_phases(fft_bins);

            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);

            for (size_t i = 0; i < fft_bins; ++i)
            {
                mags[i] = mag_dist(rng);
                prev_phases[i] = phase_dist(rng);
                time_grad[i] = grad_dist(rng) * 0.1f;
                freq_grad[i] = grad_dist(rng) * 0.1f;
            }

            rtpghi::FrameInput input { mags.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };
    }
}

TEST_CASE("Gradient Calculation Scaling", "[!benchmark]")
{
    const float pi = 3.14159265359f;
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> phase_dist(-pi, pi);

    SECTION("Gradient calculation scaling with size")
    {
        // Small size
        BENCHMARK("Gradient 128 bins")
        {
            const size_t size = 128;
            std::vector<float> phases(size);
            std::vector<float> gradients(size);
            
            for (size_t i = 0; i < size; ++i)
            {
                phases[i] = phase_dist(rng);
            }
            
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::CENTRAL };
            return rtpghi::calculate_gradients(input, gradients.data());
        };

        // Medium size
        BENCHMARK("Gradient 512 bins")
        {
            const size_t size = 512;
            std::vector<float> phases(size);
            std::vector<float> gradients(size);
            
            for (size_t i = 0; i < size; ++i)
            {
                phases[i] = phase_dist(rng);
            }
            
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::CENTRAL };
            return rtpghi::calculate_gradients(input, gradients.data());
        };

        // Large size
        BENCHMARK("Gradient 2048 bins")
        {
            const size_t size = 2048;
            std::vector<float> phases(size);
            std::vector<float> gradients(size);
            
            for (size_t i = 0; i < size; ++i)
            {
                phases[i] = phase_dist(rng);
            }
            
            rtpghi::GradientInput input { phases.data(), size, 1.0f, rtpghi::GradientMethod::CENTRAL };
            return rtpghi::calculate_gradients(input, gradients.data());
        };
    }
}

TEST_CASE("Time vs Frequency Gradient Performance", "[!benchmark]")
{
    const size_t fft_bins = 513;
    const float pi = 3.14159265359f;
    std::mt19937 rng(99999);
    std::uniform_real_distribution<float> phase_dist(-pi, pi);

    std::vector<float> prev_phases(fft_bins);
    std::vector<float> curr_phases(fft_bins);
    std::vector<float> gradients(fft_bins);

    for (size_t i = 0; i < fft_bins; ++i)
    {
        prev_phases[i] = phase_dist(rng);
        curr_phases[i] = phase_dist(rng);
    }

    SECTION("Time vs Frequency gradient comparison")
    {
        BENCHMARK("Time Gradients (between frames)")
        {
            return rtpghi::calculate_time_gradients(prev_phases.data(), curr_phases.data(),
                                                   fft_bins, 0.01f, 
                                                   rtpghi::GradientMethod::FORWARD,
                                                   gradients.data());
        };

        BENCHMARK("Frequency Gradients (within frame)")
        {
            return rtpghi::calculate_freq_gradients(curr_phases.data(), fft_bins, 100.0f,
                                                   rtpghi::GradientMethod::CENTRAL,
                                                   gradients.data());
        };
    }
}

TEST_CASE("Complete Workflow Performance", "[!benchmark]")
{
    const size_t fft_bins = 513;
    const float sample_rate = 44100.0f;
    const float time_step = 512.0f / sample_rate;
    const float freq_step = sample_rate / (2 * fft_bins);
    const float pi = 3.14159265359f;

    std::vector<float> magnitudes(fft_bins);
    std::vector<float> prev_phases(fft_bins);
    std::vector<float> curr_phases(fft_bins);
    std::vector<float> time_grad(fft_bins);
    std::vector<float> freq_grad(fft_bins);
    std::vector<float> out_mags(fft_bins);
    std::vector<float> out_phases(fft_bins);

    // Initialize with realistic data
    for (size_t i = 0; i < fft_bins; ++i)
    {
        float freq = static_cast<float>(i) * freq_step;
        magnitudes[i] = (i % 20 == 0) ? 0.8f : 0.05f;
        prev_phases[i] = 2.0f * pi * freq * 0.01f;
        curr_phases[i] = 2.0f * pi * freq * 0.02f;
    }

    SECTION("Complete spectrogram-to-RTPGHI workflow")
    {
        BENCHMARK("Complete Workflow")
        {
            // Step 1: Calculate time gradients
            rtpghi::calculate_time_gradients(prev_phases.data(), curr_phases.data(),
                                           fft_bins, time_step, 
                                           rtpghi::GradientMethod::FORWARD,
                                           time_grad.data());

            // Step 2: Calculate frequency gradients
            rtpghi::calculate_freq_gradients(curr_phases.data(), fft_bins, freq_step,
                                           rtpghi::GradientMethod::CENTRAL,
                                           freq_grad.data());

            // Step 3: Apply RTPGHI
            // Create processor outside timed section
            rtpghi::ProcessorConfig config(fft_bins);
            rtpghi::Processor processor(config);
            
            rtpghi::FrameInput input { magnitudes.data(), prev_phases.data(),
                                      time_grad.data(), freq_grad.data(),
                                      nullptr, fft_bins };
            rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
            
            auto result = processor.process(input, output);
            
            // No cleanup needed with RAII
            
            return result;
        };
    }
}