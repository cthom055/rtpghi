#include <cmath>
#include <iostream>
#include <rtpghi/rtpghi.hpp>
#include <vector>

int main()
{
    std::cout << "RTPGHI Basic Example\n";
    std::cout << "===================\n\n";

    // Setup parameters
    const size_t fft_bins = 256;  // FFT size / 2 + 1

    // Create test data
    std::vector<float> magnitudes(fft_bins);
    std::vector<float> previous_phases(fft_bins);
    std::vector<float> time_gradients(fft_bins);
    std::vector<float> freq_gradients(fft_bins);

    // Initialize with some test data
    for (size_t i = 0; i < fft_bins; ++i)
    {
        magnitudes[i] = 1.0f / (1.0f + i);                   // Decreasing magnitude
        previous_phases[i] = static_cast<float>(i) * 0.01f;  // Linear phase
        time_gradients[i] = 0.1f;                            // Constant time evolution
        freq_gradients[i] = 0.0f;                            // No frequency spreading
    }

    // Output buffers
    std::vector<float> output_magnitudes(fft_bins);
    std::vector<float> output_phases(fft_bins);

    // Setup RTPGHI input/output
    rtpghi::FrameInput input {
        magnitudes.data(), previous_phases.data(), time_gradients.data(), freq_gradients.data(), fft_bins
    };

    rtpghi::FrameOutput output { output_magnitudes.data(), output_phases.data(), fft_bins };

    // Process frame
    auto result = rtpghi::process(input, output);

    if (result != rtpghi::ErrorCode::OK)
    {
        std::cerr << "Error: Processing failed with code " << static_cast<int>(result) << std::endl;
        return 1;
    }

    std::cout << "Processing successful!\n\n";

    // Show some results
    std::cout << "Sample results (first 10 bins):\n";
    std::cout << "Bin | Input Mag | Output Mag | Input Phase | Output Phase\n";
    std::cout << "----+-----------+------------+-------------+-------------\n";

    for (size_t i = 0; i < 10; ++i)
    {
        printf("%3zu | %9.3f | %10.3f | %11.3f | %11.3f\n",
               i,
               magnitudes[i],
               output_magnitudes[i],
               previous_phases[i],
               output_phases[i]);
    }

    std::cout << "\nNote: This is a placeholder implementation.\n";
    std::cout << "The actual RTPGHI algorithm will be implemented later.\n";

    return 0;
}
