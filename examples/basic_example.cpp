#include <cmath>
#include <iostream>
#include <rtpghi/rtpghi.hpp>
#include <vector>

int main()
{
    std::cout << "RTPGHI Basic Example\n";
    std::cout << "====================\n\n";

    // Audio processing parameters
    const size_t fft_bins = 513;  // 1024 FFT -> 513 bins
    const float sample_rate = 44100.0f;
    const float time_step = 512.0f / sample_rate;  // ~11.6ms hop
    const float freq_step = sample_rate / 1024.0f;  // ~43Hz per bin

    std::cout << "Processing a single audio frame:\n";
    std::cout << "- FFT bins: " << fft_bins << "\n";
    std::cout << "- Sample rate: " << sample_rate << " Hz\n";
    std::cout << "- Time step: " << time_step * 1000 << " ms\n\n";

    // Create realistic test data
    std::vector<float> magnitudes(fft_bins);
    std::vector<float> prev_phases(fft_bins);
    std::vector<float> curr_phases(fft_bins);
    std::vector<float> time_gradients(fft_bins);
    std::vector<float> freq_gradients(fft_bins);

    // Simulate harmonic content (musical note)
    const float fundamental = 440.0f;  // A4
    for (size_t i = 0; i < fft_bins; ++i)
    {
        float freq = i * freq_step;
        
        // Add harmonics at multiples of fundamental
        bool is_harmonic = false;
        for (int h = 1; h <= 8; ++h)
        {
            if (std::abs(freq - h * fundamental) < freq_step / 2)
            {
                magnitudes[i] = 1.0f / h;  // Decreasing harmonic strength
                is_harmonic = true;
                break;
            }
        }
        
        if (!is_harmonic)
        {
            magnitudes[i] = 0.01f;  // Background noise
        }
        
        // Phase evolution
        prev_phases[i] = 0.0f;
        curr_phases[i] = 2.0f * rtpghi::constants::PI * freq * time_step;
    }

    std::cout << "Step 1: Calculate gradients\n";
    
    // Calculate time gradients (phase change between frames)
    rtpghi::calculate_time_gradients(
        prev_phases.data(), curr_phases.data(), fft_bins,
        time_step, rtpghi::GradientMethod::FORWARD,
        time_gradients.data()
    );
    
    // Calculate frequency gradients (phase change across bins)
    rtpghi::calculate_freq_gradients(
        curr_phases.data(), fft_bins, freq_step,
        rtpghi::GradientMethod::CENTRAL,
        freq_gradients.data()
    );
    
    std::cout << "         Time and frequency gradients calculated\n\n";

    std::cout << "Step 2: Configure RTPGHI processor\n";
    
    // Create processor with default Forward Euler integration
    rtpghi::ProcessorConfig config(fft_bins);
    rtpghi::Processor processor(config);
    
    std::cout << "         Integration method: Forward Euler\n";
    std::cout << "         Tolerance: " << config.tolerance << "\n\n";

    std::cout << "Step 3: Process frame\n";
    
    // Setup input and output
    std::vector<float> output_mags(fft_bins);
    std::vector<float> output_phases(fft_bins);
    
    rtpghi::FrameInput input {
        magnitudes.data(),
        prev_phases.data(),
        time_gradients.data(),
        freq_gradients.data(),
        nullptr,  // No previous gradients needed for Forward Euler
        fft_bins
    };
    
    rtpghi::FrameOutput output {
        output_mags.data(),
        output_phases.data(),
        fft_bins
    };

    // Process the frame
    auto result = processor.process(input, output);
    
    if (result != rtpghi::ErrorCode::OK)
    {
        std::cerr << "Error: Processing failed\n";
        return 1;
    }
    
    std::cout << "         Processing successful!\n\n";

    std::cout << "Step 4: Analyze results\n";
    
    // Find harmonics and analyze output
    size_t harmonic_bins = 0;
    size_t processed_bins = 0;
    float peak_magnitude = 0;
    size_t peak_bin = 0;
    
    for (size_t i = 0; i < fft_bins; ++i)
    {
        if (magnitudes[i] > 0.1f)
        {
            harmonic_bins++;
        }
        if (output_mags[i] > 0.1f)
        {
            processed_bins++;
        }
        if (output_mags[i] > peak_magnitude)
        {
            peak_magnitude = output_mags[i];
            peak_bin = i;
        }
    }
    
    float peak_freq = peak_bin * freq_step;
    
    std::cout << "         Input harmonics: " << harmonic_bins << " bins\n";
    std::cout << "         Processed bins: " << processed_bins << " bins\n";
    std::cout << "         Peak: " << peak_magnitude << " at " << peak_freq << " Hz\n\n";

    // Show first few harmonic results
    std::cout << "Harmonic analysis:\n";
    std::cout << "Bin | Freq (Hz) | Input Mag | Output Mag | Phase Change\n";
    std::cout << "----+-----------+-----------+------------+-------------\n";
    
    for (size_t i = 0; i < fft_bins && i < 50; ++i)
    {
        if (magnitudes[i] > 0.1f)  // Only show significant bins
        {
            float freq = i * freq_step;
            float phase_change = output_phases[i] - prev_phases[i];
            
            printf("%3zu | %9.1f | %9.3f | %10.3f | %11.6f\n",
                   i, freq, magnitudes[i], output_mags[i], phase_change);
        }
    }

    std::cout << "\nThe RTPGHI algorithm:\n";
    std::cout << "✓ Preserved magnitude information\n";
    std::cout << "✓ Computed coherent phases using gradient information\n";
    std::cout << "✓ Applied heap-based phase propagation for quality\n";
    std::cout << "✓ Ready for inverse FFT and audio reconstruction\n\n";

    std::cout << "Next steps:\n";
    std::cout << "- Use output phases with magnitudes for inverse FFT\n";
    std::cout << "- Run './complete_workflow' for multi-frame processing\n";

    return 0;
}