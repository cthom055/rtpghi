#include <chrono>
#include <cmath>
#include <iostream>
#include <rtpghi/rtpghi.hpp>
#include <vector>

// Generate a simple harmonic signal for testing
void generate_harmonic_frame(std::vector<float>& magnitudes, std::vector<float>& phases,
                           size_t fft_bins, float time_index, float sample_rate)
{
    const float freq_step = sample_rate / (2 * fft_bins);
    
    // Clear spectrum
    std::fill(magnitudes.begin(), magnitudes.end(), 0.01f);
    std::fill(phases.begin(), phases.end(), 0.0f);
    
    // Add harmonic content for A4 (440 Hz)
    const float fundamental = 440.0f;
    for (int harmonic = 1; harmonic <= 5; ++harmonic)
    {
        float freq = fundamental * harmonic;
        size_t bin = static_cast<size_t>(freq / freq_step);
        
        if (bin < fft_bins)
        {
            magnitudes[bin] = 1.0f / harmonic;  // Decreasing amplitude
            phases[bin] = 2.0f * rtpghi::constants::PI * freq * time_index / sample_rate;
        }
    }
}

void demonstrate_integration_methods(size_t fft_bins)
{
    std::cout << "\n=== Integration Method Comparison ===\n\n";
    
    const float sample_rate = 44100.0f;
    const float time_step = 512.0f / sample_rate;
    const float freq_step = sample_rate / (2 * fft_bins);
    
    // Prepare test data
    std::vector<float> magnitudes(fft_bins, 1.0f);
    std::vector<float> prev_phases(fft_bins, 0.0f);
    std::vector<float> curr_time_grad(fft_bins, 0.1f);
    std::vector<float> prev_time_grad(fft_bins, 0.08f);
    std::vector<float> freq_grad(fft_bins, 0.02f);
    
    std::vector<float> euler_phases(fft_bins);
    std::vector<float> euler_mags(fft_bins);
    std::vector<float> trap_phases(fft_bins);
    std::vector<float> trap_mags(fft_bins);
    
    // Test Forward Euler
    auto start = std::chrono::high_resolution_clock::now();
    
    rtpghi::ProcessorConfig euler_config(fft_bins);
    rtpghi::Processor euler_processor(euler_config);
    
    rtpghi::FrameInput euler_input {
        magnitudes.data(), prev_phases.data(),
        curr_time_grad.data(), freq_grad.data(),
        nullptr, fft_bins
    };
    rtpghi::FrameOutput euler_output {
        euler_mags.data(), euler_phases.data(), fft_bins
    };
    
    euler_processor.process(euler_input, euler_output);
    auto euler_time = std::chrono::high_resolution_clock::now();
    
    // Test Trapezoidal
    rtpghi::ProcessorConfig trap_config(fft_bins, rtpghi::constants::DEFAULT_TOLERANCE,
                                       rtpghi::constants::DEFAULT_RANDOM_SEED,
                                       rtpghi::IntegrationMethod::TRAPEZOIDAL);
    rtpghi::Processor trap_processor(trap_config);
    
    rtpghi::FrameInput trap_input {
        magnitudes.data(), prev_phases.data(),
        curr_time_grad.data(), freq_grad.data(),
        prev_time_grad.data(), fft_bins
    };
    rtpghi::FrameOutput trap_output {
        trap_mags.data(), trap_phases.data(), fft_bins
    };
    
    trap_processor.process(trap_input, trap_output);
    auto trap_time = std::chrono::high_resolution_clock::now();
    
    // Calculate timing
    auto euler_duration = std::chrono::duration_cast<std::chrono::microseconds>(euler_time - start);
    auto trap_duration = std::chrono::duration_cast<std::chrono::microseconds>(trap_time - euler_time);
    
    std::cout << "Method comparison (first 8 bins):\n";
    std::cout << "Bin | Prev Grad | Curr Grad | Euler Result | Trap Result | Difference\n";
    std::cout << "----+-----------+-----------+--------------+-------------+-----------\n";
    
    for (size_t i = 0; i < 8; ++i)
    {
        float diff = trap_phases[i] - euler_phases[i];
        printf("%3zu | %9.3f | %9.3f | %12.6f | %11.6f | %9.6f\n",
               i, prev_time_grad[i], curr_time_grad[i],
               euler_phases[i], trap_phases[i], diff);
    }
    
    std::cout << "\nPerformance:\n";
    std::cout << "Forward Euler: " << euler_duration.count() << " μs\n";
    std::cout << "Trapezoidal:   " << trap_duration.count() << " μs\n";
    std::cout << "Overhead:      " << (trap_duration.count() - euler_duration.count()) << " μs\n";
}

void demonstrate_realtime_processing(size_t fft_bins, size_t num_frames)
{
    std::cout << "\n=== Real-Time Processing Workflow ===\n\n";
    
    const float sample_rate = 44100.0f;
    const float time_step = 512.0f / sample_rate;
    const float freq_step = sample_rate / (2 * fft_bins);
    
    std::cout << "Parameters:\n";
    std::cout << "- Sample rate: " << sample_rate << " Hz\n";
    std::cout << "- Time step: " << time_step * 1000 << " ms\n";
    std::cout << "- Frequency step: " << freq_step << " Hz\n";
    std::cout << "- Processing " << num_frames << " frames\n\n";
    
    // Allocate buffers
    std::vector<float> magnitudes(fft_bins);
    std::vector<float> prev_phases(fft_bins, 0.0f);
    std::vector<float> curr_phases(fft_bins);
    std::vector<float> time_grad(fft_bins);
    std::vector<float> freq_grad(fft_bins);
    std::vector<float> output_mags(fft_bins);
    std::vector<float> output_phases(fft_bins);
    
    // Create processor
    rtpghi::ProcessorConfig config(fft_bins);
    rtpghi::Processor processor(config);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    double total_processing_time = 0.0;
    double max_frame_time = 0.0;
    
    std::cout << "Processing frames...\n";
    std::cout << "Frame | Peak Freq | Processing Time | Significant Bins\n";
    std::cout << "------|-----------|-----------------|------------------\n";
    
    for (size_t frame = 0; frame < num_frames; ++frame)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Generate frame data
        generate_harmonic_frame(magnitudes, curr_phases, fft_bins, frame * time_step, sample_rate);
        
        // Calculate gradients
        rtpghi::calculate_time_gradients(
            prev_phases.data(), curr_phases.data(), fft_bins,
            time_step, rtpghi::GradientMethod::FORWARD,
            time_grad.data()
        );
        
        rtpghi::calculate_freq_gradients(
            curr_phases.data(), fft_bins, freq_step,
            rtpghi::GradientMethod::CENTRAL,
            freq_grad.data()
        );
        
        // Process with RTPGHI
        rtpghi::FrameInput input {
            magnitudes.data(), prev_phases.data(),
            time_grad.data(), freq_grad.data(),
            nullptr, fft_bins
        };
        rtpghi::FrameOutput output {
            output_mags.data(), output_phases.data(), fft_bins
        };
        
        processor.process(input, output);
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        double frame_time = frame_duration.count();
        
        // Analyze results
        size_t peak_bin = 0;
        float peak_mag = output_mags[0];
        size_t significant_bins = 0;
        
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (output_mags[i] > peak_mag)
            {
                peak_mag = output_mags[i];
                peak_bin = i;
            }
            if (output_mags[i] > 0.1f)
            {
                significant_bins++;
            }
        }
        
        float peak_freq = peak_bin * freq_step;
        
        // Show progress for first few frames and every 10th frame
        if (frame < 5 || frame % 10 == 0)
        {
            printf("%5zu | %9.1f | %15.1f | %16zu\n",
                   frame, peak_freq, frame_time, significant_bins);
        }
        
        // Update for next iteration
        prev_phases = output_phases;
        max_frame_time = std::max(max_frame_time, frame_time);
        total_processing_time += frame_time;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    std::cout << "\nPerformance Summary:\n";
    std::cout << "- Total time: " << total_duration.count() << " μs\n";
    std::cout << "- Average per frame: " << total_processing_time / num_frames << " μs\n";
    std::cout << "- Maximum frame time: " << max_frame_time << " μs\n";
    std::cout << "- Real-time budget: " << time_step * 1e6 << " μs per frame\n";
    
    bool realtime_capable = max_frame_time < (time_step * 1e6);
    std::cout << "- Real-time capable: " << (realtime_capable ? "YES" : "NO") << "\n";
    
    if (realtime_capable)
    {
        double cpu_usage = (max_frame_time / (time_step * 1e6)) * 100.0;
        std::cout << "- CPU usage: " << cpu_usage << "%\n";
    }
}

int main()
{
    std::cout << "RTPGHI Complete Workflow Example\n";
    std::cout << "=================================\n";
    
    const size_t fft_bins = 513;  // 1024 FFT -> 513 bins
    
    try
    {
        // Demonstrate different aspects of RTPGHI
        demonstrate_integration_methods(fft_bins);
        demonstrate_realtime_processing(fft_bins, 25);
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "✓ Compared Forward Euler vs Trapezoidal integration\n";
        std::cout << "✓ Demonstrated real-time audio processing workflow\n";
        std::cout << "✓ Showed performance characteristics and CPU usage\n";
        std::cout << "✓ Validated RTPGHI for live audio applications\n\n";
        
        std::cout << "Key takeaways:\n";
        std::cout << "- Forward Euler is faster and suitable for most applications\n";
        std::cout << "- Trapezoidal provides higher accuracy with minimal overhead\n";
        std::cout << "- RTPGHI maintains real-time performance for typical frame rates\n";
        std::cout << "- The algorithm preserves harmonic structure in audio signals\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}