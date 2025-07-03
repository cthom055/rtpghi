#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <rtpghi/rtpghi.hpp>
#include <vector>

/// Simulate FFT magnitude spectrum for a complex audio signal
void generate_audio_spectrum(std::vector<float>& magnitudes, std::vector<float>& phases, 
                           size_t fft_bins, float time_index)
{
    const float pi = 3.14159265359f;
    const float sample_rate = 44100.0f;
    
    // Clear spectrum
    std::fill(magnitudes.begin(), magnitudes.end(), 0.01f);  // Noise floor
    std::fill(phases.begin(), phases.end(), 0.0f);
    
    // Add harmonic content (simulating musical notes)
    std::vector<float> fundamentals = { 220.0f, 440.0f, 660.0f };  // A3, A4, E5
    
    for (float fundamental : fundamentals)
    {
        // Add harmonics for each fundamental
        for (int harmonic = 1; harmonic <= 5; ++harmonic)
        {
            float freq = fundamental * harmonic;
            size_t bin = static_cast<size_t>(freq * fft_bins * 2 / sample_rate);
            
            if (bin < fft_bins)
            {
                float amplitude = 1.0f / harmonic;  // Decreasing harmonic amplitude
                magnitudes[bin] += amplitude * (0.8f + 0.2f * std::sin(time_index * 0.1f));
                phases[bin] = 2.0f * pi * freq * time_index / sample_rate;
            }
        }
    }
    
    // Add some broadband transient content
    if (static_cast<int>(time_index) % 50 == 0)  // Periodic transients
    {
        for (size_t i = fft_bins / 4; i < 3 * fft_bins / 4; ++i)
        {
            magnitudes[i] += 0.3f * std::exp(-static_cast<float>(i - fft_bins/2) / 50.0f);
        }
    }
}

void demonstrate_gradient_methods(size_t fft_bins)
{
    std::cout << "\n=== Phase Gradient Calculation Methods ===\n\n";
    
    // Create test phase data with known derivatives
    std::vector<float> phases(fft_bins);
    std::vector<float> time_gradients(fft_bins);
    std::vector<float> freq_gradients_forward(fft_bins);
    std::vector<float> freq_gradients_backward(fft_bins);
    std::vector<float> freq_gradients_central(fft_bins);
    
    // Generate quadratic phase for testing: phase(x) = 0.5 * x^2
    // True derivative: phase'(x) = x
    const float freq_step = 100.0f;  // Hz per bin
    for (size_t i = 0; i < fft_bins; ++i)
    {
        float x = i * freq_step / 1000.0f;  // Normalize
        phases[i] = 0.5f * x * x;
        time_gradients[i] = 0.05f;  // Constant time evolution
    }
    
    // Calculate gradients using different methods
    rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step, 
                                   rtpghi::GradientMethod::FORWARD, freq_gradients_forward.data());
    rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                   rtpghi::GradientMethod::BACKWARD, freq_gradients_backward.data());
    rtpghi::calculate_freq_gradients(phases.data(), fft_bins, freq_step,
                                   rtpghi::GradientMethod::CENTRAL, freq_gradients_central.data());
    
    std::cout << "Frequency Gradient Methods Comparison (first 10 bins):\n";
    std::cout << "Bin | Phase  | True Grad | Forward  | Backward | Central  | Central Error\n";
    std::cout << "----+--------+-----------+----------+----------+----------+--------------\n";
    
    for (size_t i = 0; i < std::min(size_t(10), fft_bins); ++i)
    {
        float x = i * freq_step / 1000.0f;
        float true_gradient = x / freq_step;  // True derivative normalized by step
        float central_error = std::abs(freq_gradients_central[i] - true_gradient);
        
        printf("%3zu | %6.3f | %9.6f | %8.6f | %8.6f | %8.6f | %12.8f\n",
               i, phases[i], true_gradient, 
               freq_gradients_forward[i], freq_gradients_backward[i], freq_gradients_central[i],
               central_error);
    }
    
    std::cout << "\nNote: Central difference method typically provides higher accuracy\n";
    std::cout << "for smooth functions, as evidenced by lower error values.\n";
}

void demonstrate_integration_methods(size_t fft_bins)
{
    std::cout << "\n=== Integration Method Comparison ===\n\n";
    
    std::vector<float> magnitudes(fft_bins, 1.0f);
    std::vector<float> prev_phases(fft_bins, 0.0f);
    std::vector<float> time_grad_current(fft_bins);
    std::vector<float> time_grad_previous(fft_bins);
    std::vector<float> freq_grad(fft_bins, 0.0f);
    
    std::vector<float> out_mags_euler(fft_bins);
    std::vector<float> out_phases_euler(fft_bins);
    std::vector<float> out_mags_trap(fft_bins);
    std::vector<float> out_phases_trap(fft_bins);
    
    // Setup test gradients with varying time evolution
    for (size_t i = 0; i < fft_bins; ++i)
    {
        time_grad_previous[i] = 0.1f + 0.05f * std::sin(i / 10.0f);
        time_grad_current[i] = 0.12f + 0.04f * std::sin(i / 10.0f);
        prev_phases[i] = 0.5f * i / fft_bins;  // Linear starting phases
    }
    
    // Test Forward Euler
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create processor for Euler method
    rtpghi::ProcessorConfig euler_config(fft_bins);
    rtpghi::Processor euler_processor(euler_config);
    
    rtpghi::FrameInput euler_input { magnitudes.data(), prev_phases.data(),
                                   time_grad_current.data(), freq_grad.data(),
                                   nullptr, fft_bins };
    rtpghi::FrameOutput euler_output { out_mags_euler.data(), out_phases_euler.data(), fft_bins };
    
    euler_processor.process(euler_input, euler_output);
    
    // No cleanup needed with RAII
    
    auto euler_time = std::chrono::high_resolution_clock::now();
    
    // Test Trapezoidal
    // Create processor configured for Trapezoidal method
    rtpghi::ProcessorConfig trap_config(fft_bins, rtpghi::constants::DEFAULT_TOLERANCE,
                                       rtpghi::constants::DEFAULT_RANDOM_SEED,
                                       rtpghi::IntegrationMethod::TRAPEZOIDAL);
    rtpghi::Processor trap_processor(trap_config);
    
    rtpghi::FrameInput trap_input { magnitudes.data(), prev_phases.data(),
                                  time_grad_current.data(), freq_grad.data(),
                                  time_grad_previous.data(), fft_bins };
    rtpghi::FrameOutput trap_output { out_mags_trap.data(), out_phases_trap.data(), fft_bins };
    
    trap_processor.process(trap_input, trap_output);
    
    // No cleanup needed with RAII
    
    auto trap_time = std::chrono::high_resolution_clock::now();
    
    auto euler_duration = std::chrono::duration_cast<std::chrono::microseconds>(euler_time - start_time);
    auto trap_duration = std::chrono::duration_cast<std::chrono::microseconds>(trap_time - euler_time);
    
    std::cout << "Integration Method Comparison (first 10 bins):\n";
    std::cout << "Bin | Prev Phase | Prev Grad | Curr Grad | Euler Result | Trapez Result | Difference\n";
    std::cout << "----+------------+-----------+-----------+--------------+---------------+-----------\n";
    
    for (size_t i = 0; i < std::min(size_t(10), fft_bins); ++i)
    {
        float difference = out_phases_trap[i] - out_phases_euler[i];
        
        printf("%3zu | %10.6f | %9.6f | %9.6f | %12.6f | %13.6f | %9.6f\n",
               i, prev_phases[i], time_grad_previous[i], time_grad_current[i],
               out_phases_euler[i], out_phases_trap[i], difference);
    }
    
    std::cout << "\nPerformance:\n";
    std::cout << "Forward Euler: " << euler_duration.count() << " μs\n";
    std::cout << "Trapezoidal:   " << trap_duration.count() << " μs\n";
    std::cout << "Overhead:      " << (trap_duration.count() - euler_duration.count()) << " μs ("
              << std::fixed << std::setprecision(1) 
              << (100.0 * trap_duration.count() / euler_duration.count() - 100.0) << "% slower)\n";
}

void demonstrate_realtime_workflow(size_t fft_bins, size_t num_frames)
{
    std::cout << "\n=== Real-Time Audio Processing Workflow ===\n\n";
    
    const float sample_rate = 44100.0f;
    const size_t hop_size = 512;
    const float time_step = hop_size / sample_rate;
    const float freq_step = sample_rate / (2 * fft_bins);
    
    std::cout << "Audio Parameters:\n";
    std::cout << "Sample Rate: " << sample_rate << " Hz\n";
    std::cout << "FFT Size: " << (fft_bins - 1) * 2 << " (-> " << fft_bins << " bins)\n";
    std::cout << "Hop Size: " << hop_size << " samples\n";
    std::cout << "Frame Rate: " << sample_rate / hop_size << " Hz\n";
    std::cout << "Time per Frame: " << time_step * 1000 << " ms\n\n";
    
    // Storage for multi-frame processing
    std::vector<float> magnitudes(fft_bins);
    std::vector<float> prev_phases(fft_bins, 0.0f);
    std::vector<float> curr_phases(fft_bins);
    std::vector<float> prev_time_grad(fft_bins, 0.0f);
    std::vector<float> curr_time_grad(fft_bins);
    std::vector<float> freq_grad(fft_bins);
    std::vector<float> out_mags(fft_bins);
    std::vector<float> out_phases(fft_bins);
    
    // Create processor for real-time workflow
    rtpghi::ProcessorConfig realtime_config(fft_bins);
    rtpghi::Processor realtime_processor(realtime_config);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    double max_frame_time = 0.0;
    double total_processing_time = 0.0;
    
    std::cout << "Processing " << num_frames << " frames...\n";
    std::cout << "Frame | Peak Bin | Peak Mag | Time (μs) | Significant Bins | Phase Range\n";
    std::cout << "------|----------|----------|-----------|------------------|------------\n";
    
    for (size_t frame = 0; frame < num_frames; ++frame)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Step 1: Generate/receive new spectrum
        generate_audio_spectrum(magnitudes, curr_phases, fft_bins, static_cast<float>(frame));
        
        // Step 2: Calculate time gradients from previous frame
        if (frame > 0)
        {
            rtpghi::calculate_time_gradients(prev_phases.data(), curr_phases.data(), fft_bins,
                                           time_step, rtpghi::GradientMethod::FORWARD, curr_time_grad.data());
        }
        else
        {
            std::fill(curr_time_grad.begin(), curr_time_grad.end(), 0.0f);
        }
        
        // Step 3: Calculate frequency gradients within current frame
        rtpghi::calculate_freq_gradients(curr_phases.data(), fft_bins, freq_step,
                                       rtpghi::GradientMethod::CENTRAL, freq_grad.data());
        
        // Step 4: Apply RTPGHI processing
        rtpghi::FrameInput input { magnitudes.data(), prev_phases.data(),
                                 curr_time_grad.data(), freq_grad.data(),
                                 frame > 0 ? prev_time_grad.data() : nullptr, fft_bins };
        rtpghi::FrameOutput output { out_mags.data(), out_phases.data(), fft_bins };
        
        realtime_processor.process(input, output);
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        double frame_time_us = frame_duration.count();
        
        // Analysis of results
        size_t peak_bin = 0;
        float peak_mag = out_mags[0];
        size_t significant_bins = 0;
        float min_phase = out_phases[0], max_phase = out_phases[0];
        
        for (size_t i = 0; i < fft_bins; ++i)
        {
            if (out_mags[i] > peak_mag)
            {
                peak_mag = out_mags[i];
                peak_bin = i;
            }
            if (out_mags[i] > 0.1f) significant_bins++;
            if (out_phases[i] < min_phase) min_phase = out_phases[i];
            if (out_phases[i] > max_phase) max_phase = out_phases[i];
        }
        
        // Show progress every 10 frames or for first few frames
        if (frame < 5 || frame % 10 == 0)
        {
            printf("%5zu | %8zu | %8.3f | %9.1f | %16zu | [%6.3f, %6.3f]\n",
                   frame, peak_bin, peak_mag, frame_time_us, significant_bins, min_phase, max_phase);
        }
        
        // Update for next frame
        prev_phases = out_phases;
        prev_time_grad = curr_time_grad;
        
        max_frame_time = std::max(max_frame_time, frame_time_us);
        total_processing_time += frame_time_us;
    }
    
    // No cleanup needed with RAII
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    std::cout << "\nProcessing Statistics:\n";
    std::cout << "Total time: " << total_duration.count() << " μs\n";
    std::cout << "Processing time: " << std::fixed << std::setprecision(1) << total_processing_time << " μs\n";
    std::cout << "Average per frame: " << total_processing_time / num_frames << " μs\n";
    std::cout << "Maximum frame time: " << max_frame_time << " μs\n";
    std::cout << "Real-time budget: " << time_step * 1e6 << " μs per frame\n";
    
    bool realtime_capable = max_frame_time < (time_step * 1e6);
    std::cout << "Real-time capable: " << (realtime_capable ? "YES" : "NO") << "\n";
    if (realtime_capable)
    {
        std::cout << "CPU usage: " << (max_frame_time / (time_step * 1e6)) * 100.0 << "%\n";
    }
}

int main()
{
    std::cout << "RTPGHI Complete Workflow Demonstration\n";
    std::cout << "======================================\n";
    
    const size_t fft_bins = 513;  // 1024 FFT -> 513 bins
    
    try
    {
        // Demonstrate different aspects of the RTPGHI workflow
        demonstrate_gradient_methods(fft_bins);
        demonstrate_integration_methods(fft_bins);
        demonstrate_realtime_workflow(fft_bins, 50);
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "This example demonstrated:\n";
        std::cout << "1. Phase gradient calculation using forward, backward, and central differences\n";
        std::cout << "2. Integration method comparison (Forward Euler vs. Trapezoidal)\n";
        std::cout << "3. Complete real-time audio processing workflow with RTPGHI\n";
        std::cout << "4. Performance analysis and real-time capability assessment\n\n";
        
        std::cout << "The RTPGHI algorithm successfully processes audio spectrogram data by:\n";
        std::cout << "- Computing phase gradients from spectral evolution\n";
        std::cout << "- Applying heap-based magnitude-priority phase propagation\n";
        std::cout << "- Handling phase unwrapping and tolerance-based bin classification\n";
        std::cout << "- Maintaining real-time performance for typical audio frame rates\n";
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}