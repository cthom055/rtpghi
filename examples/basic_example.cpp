#include <cmath>
#include <iostream>
#include <rtpghi/rtpghi.hpp>
#include <vector>

int main()
{
    std::cout << "RTPGHI Basic Example\n";
    std::cout << "===================\n\n";
    
    std::cout << "This example demonstrates basic usage of the RTPGHI library for\n";
    std::cout << "Real-Time Phase Gradient Heap Integration audio processing.\n\n";

    // Setup parameters for typical audio processing
    const size_t fft_bins = 256;  // FFT size / 2 + 1 (512 FFT -> 256 bins)
    const float sample_rate = 44100.0f;
    const size_t hop_size = 256;
    
    std::cout << "Audio Parameters:\n";
    std::cout << "- FFT Size: " << (fft_bins - 1) * 2 << " samples\n";
    std::cout << "- FFT Bins: " << fft_bins << " (positive frequencies only)\n";
    std::cout << "- Sample Rate: " << sample_rate << " Hz\n";
    std::cout << "- Hop Size: " << hop_size << " samples\n\n";

    // Create test data representing a typical audio spectrum
    std::vector<float> magnitudes(fft_bins);
    std::vector<float> previous_phases(fft_bins);
    std::vector<float> time_gradients(fft_bins);
    std::vector<float> freq_gradients(fft_bins);

    // Initialize with realistic audio data
    const float pi = 3.14159265359f;
    for (size_t i = 0; i < fft_bins; ++i)
    {
        float freq = i * sample_rate / (2 * fft_bins);  // Frequency of this bin
        
        // Simulate musical harmonic content
        if (i > 0 && (i % 20 == 0 || i % 33 == 0))
        {
            magnitudes[i] = 1.0f * std::exp(-freq / 4000.0f);  // Harmonics with high-freq rolloff
        }
        else
        {
            magnitudes[i] = 0.05f;  // Background noise level
        }
        
        previous_phases[i] = 2.0f * pi * freq * 0.01f;  // Phase evolution from time
        time_gradients[i] = 2.0f * pi * freq / sample_rate;  // Natural frequency evolution
        freq_gradients[i] = 0.02f * magnitudes[i];  // Magnitude-dependent freq spreading
    }

    // Output buffers
    std::vector<float> output_magnitudes(fft_bins);
    std::vector<float> output_phases(fft_bins);

    // Demonstrate gradient calculation
    std::cout << "Step 1: Calculate frequency gradients using central differences\n";
    std::vector<float> calculated_freq_grad(fft_bins);
    float freq_step = sample_rate / (2 * fft_bins);
    
    rtpghi::calculate_freq_gradients(previous_phases.data(), fft_bins, freq_step,
                                   rtpghi::GradientMethod::CENTRAL, calculated_freq_grad.data());
    
    std::cout << "         Frequency step: " << freq_step << " Hz per bin\n";
    std::cout << "         Gradients calculated for " << fft_bins << " bins\n\n";

    // Setup RTPGHI processor with RAII-managed workspace
    std::cout << "Step 2: Apply RTPGHI processing with Forward Euler integration\n";
    std::cout << "         Using processor-based design with automatic memory management\n\n";
    
    // Configure and create processor
    rtpghi::ProcessorConfig config(fft_bins, 1e-6f, 12345);
    rtpghi::Processor processor(config);
    
    std::cout << "         FFT bins: " << config.fft_bins << "\n";
    std::cout << "         Tolerance: " << config.tolerance << "\n";
    std::cout << "         Random seed: " << config.initial_random_seed << "\n";
    
    // No validation needed - constructor would have thrown if invalid
    
    rtpghi::FrameInput input { magnitudes.data(),
                               previous_phases.data(),
                               time_gradients.data(),
                               calculated_freq_grad.data(),  // Use calculated gradients
                               nullptr,  // prev_time_gradients (nullptr for Forward Euler)
                               fft_bins };

    rtpghi::FrameOutput output { output_magnitudes.data(), output_phases.data(), fft_bins };
    
    // Process frame
    auto result = processor.process(input, output);
    
    if (result != rtpghi::ErrorCode::OK)
    {
        std::cerr << "Error: Processing failed with code " << static_cast<int>(result) << std::endl;
        return 1;
    }
    
    std::cout << "         Processing successful!\n";
    std::cout << "         Processor can be reused for subsequent frames without reallocation\n\n";
    
    // No manual cleanup needed - RAII handles memory management

    // Analyze and display results
    std::cout << "Step 3: Analysis of results\n";
    
    // Find significant bins and phase statistics
    size_t significant_bin_count = 0;
    size_t peak_bin = 0;
    float peak_magnitude = 0.0f;
    float min_phase = output_phases[0], max_phase = output_phases[0];
    
    for (size_t i = 0; i < fft_bins; ++i)
    {
        if (output_magnitudes[i] > 0.1f) significant_bin_count++;
        if (output_magnitudes[i] > peak_magnitude)
        {
            peak_magnitude = output_magnitudes[i];
            peak_bin = i;
        }
        if (output_phases[i] < min_phase) min_phase = output_phases[i];
        if (output_phases[i] > max_phase) max_phase = output_phases[i];
    }
    
    float peak_frequency = peak_bin * sample_rate / (2 * fft_bins);
    
    std::cout << "         Significant bins: " << significant_bin_count << " / " << fft_bins << "\n";
    std::cout << "         Peak magnitude: " << peak_magnitude << " at bin " << peak_bin 
              << " (" << peak_frequency << " Hz)\n";
    std::cout << "         Phase range: [" << min_phase << ", " << max_phase << "] radians\n\n";

    // Show detailed results for significant bins
    std::cout << "Detailed results (first 10 bins):\n";
    std::cout << "Bin | Freq (Hz) | Input Mag | Output Mag | Input Phase | Output Phase | Phase Diff\n";
    std::cout << "----+-----------+-----------+------------+-------------+--------------+-----------\n";

    for (size_t i = 0; i < 10; ++i)
    {
        float freq = i * sample_rate / (2 * fft_bins);
        float phase_diff = output_phases[i] - previous_phases[i];
        
        printf("%3zu | %9.1f | %9.3f | %10.3f | %11.3f | %12.3f | %9.3f\n",
               i, freq,
               magnitudes[i],
               output_magnitudes[i],
               previous_phases[i],
               output_phases[i],
               phase_diff);
    }

    std::cout << "\nRTPGHI Algorithm Summary:\n";
    std::cout << "- Applied magnitude-priority heap-based phase propagation\n";
    std::cout << "- Processed significant bins using time and frequency gradients\n";
    std::cout << "- Applied random phase assignment to low-magnitude bins\n";
    std::cout << "- Maintained phase continuity through principal argument wrapping\n";
    std::cout << "- Preserved spectral magnitude information\n\n";
    
    std::cout << "For more comprehensive examples, run:\n";
    std::cout << "  ./complete_workflow  (demonstrates full spectrogram processing workflow)\n";

    return 0;
}
