/*
  ==============================================================================

  This file contains the basic startup code for an audio console example using dr_wav and dj_fft.

  ==============================================================================
*/

#define DR_WAV_IMPLEMENTATION
#include "dj_fft.h"
#include "dr_wav.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <rtpghi/rtpghi.hpp>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
    // Helper function implementations
    void print_usage()
    {
        std::cout << "Usage: audio_example <input.wav> <output.wav>" << std::endl;
        std::cout << "  Processes input WAV file with RTPGHI and saves the result" << std::endl;
    }

    // Simple Hanning window function
    std::vector<float> create_hanning_window(size_t size)
    {
        std::vector<float> window(size);
        for (size_t i = 0; i < size; ++i)
        {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
        }
        return window;
    }

    std::vector<float> load_audio_file(const char* filepath,
                                       unsigned int& channels,
                                       unsigned int& sampleRate,
                                       drwav_uint64& totalFrameCount)
    {
        float* pSampleData =
            drwav_open_file_and_read_pcm_frames_f32(filepath, &channels, &sampleRate, &totalFrameCount, nullptr);

        if (pSampleData == NULL)
        {
            std::cerr << "Error opening or reading WAV file: " << filepath << std::endl;
            return {};
        }

        std::cout << "Loaded WAV file: " << filepath << std::endl;
        std::cout << "Channels: " << channels << ", Sample Rate: " << sampleRate << ", Frames: " << totalFrameCount
                  << std::endl;

        // Copy to vector and free the original data
        std::vector<float> audio_data(pSampleData, pSampleData + (totalFrameCount * channels));
        drwav_free(pSampleData, nullptr);

        return audio_data;
    }

    std::vector<float> convert_to_mono(const std::vector<float>& audio_data,
                                       unsigned int channels,
                                       drwav_uint64 totalFrameCount)
    {
        std::vector<float> mono_audio(totalFrameCount);
        for (size_t i = 0; i < totalFrameCount; ++i)
        {
            mono_audio[i] = audio_data[i * channels];  // Take first channel
        }
        return mono_audio;
    }

    std::vector<std::vector<std::complex<float>>> process_stft(const std::vector<float>& mono_audio,
                                                               const std::vector<float>& window,
                                                               int window_size,
                                                               int hop_size,
                                                               size_t fft_bins)
    {
        // Calculate number of frames for STFT
        size_t num_frames = (mono_audio.size() - window_size) / hop_size + 1;

        std::cout << "Processing " << num_frames << " frames with RTPGHI..." << std::endl;

        // Note: This function just does STFT, not RTPGHI processing

        // Storage for processed frames
        std::vector<std::vector<std::complex<float>>> processed_stft(num_frames,
                                                                     std::vector<std::complex<float>>(fft_bins));

        // Process each frame
        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            // Extract windowed frame
            dj::fft_arg<float> frame_data(window_size);
            size_t start_idx = frame * hop_size;

            for (int i = 0; i < window_size; ++i)
            {
                float sample = (start_idx + i < mono_audio.size()) ? mono_audio[start_idx + i] : 0.0f;
                frame_data[i] = std::complex<float>(sample * window[i], 0.0f);
            }

            // Forward FFT
            auto fft_result = dj::fft1d(frame_data, dj::fft_dir::DIR_FWD);

            // Extract only positive frequencies (first half + Nyquist)
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                processed_stft[frame][bin] = fft_result[bin];
            }

            if (frame % 100 == 0)
            {
                std::cout << "STFT frame " << frame << "/" << num_frames << std::endl;
            }
        }

        return processed_stft;
    }

    std::vector<float> reconstruct_audio(const std::vector<std::vector<std::complex<float>>>& processed_stft,
                                         const std::vector<float>& window,
                                         int window_size,
                                         int hop_size,
                                         size_t original_length)
    {
        std::vector<float> reconstructed_audio(original_length, 0.0f);
        std::vector<float> overlap_weights(original_length, 0.0f);
        size_t num_frames = processed_stft.size();

        std::cout << "Reconstructing audio..." << std::endl;

        for (size_t frame = 0; frame < num_frames; ++frame)
        {
            // Create full spectrum (mirror negative frequencies)
            dj::fft_arg<float> full_spectrum(window_size);
            size_t fft_bins = processed_stft[frame].size();

            // Positive frequencies
            for (size_t bin = 0; bin < fft_bins; ++bin)
            {
                full_spectrum[bin] = processed_stft[frame][bin];
            }

            // Negative frequencies (complex conjugate of positive, excluding DC and Nyquist)
            for (size_t bin = 1; bin < fft_bins - 1; ++bin)
            {
                full_spectrum[window_size - bin] = std::conj(processed_stft[frame][bin]);
            }

            // Inverse FFT
            auto time_domain = dj::fft1d(full_spectrum, dj::fft_dir::DIR_BWD);

            // Overlap-add with window
            size_t start_idx = frame * hop_size;
            for (int i = 0; i < window_size; ++i)
            {
                if (start_idx + i < reconstructed_audio.size())
                {
                    float windowed_sample = time_domain[i].real() * window[i] / window_size;
                    reconstructed_audio[start_idx + i] += windowed_sample;
                    overlap_weights[start_idx + i] += window[i] * window[i] / window_size;
                }
            }
        }

        // Normalize by overlap weights
        for (size_t i = 0; i < reconstructed_audio.size(); ++i)
        {
            if (overlap_weights[i] > 0.0f)
            {
                reconstructed_audio[i] /= overlap_weights[i];
            }
        }

        return reconstructed_audio;
    }

    bool write_audio_file(const char* output_path,
                          const std::vector<float>& reconstructed_audio,
                          unsigned int channels,
                          unsigned int sampleRate,
                          drwav_uint64 totalFrameCount)
    {
        std::cout << "Writing output WAV file: " << output_path << std::endl;

        // Convert back to multi-channel if needed (duplicate mono to all channels)
        std::vector<float> output_data(totalFrameCount * channels);
        for (size_t i = 0; i < totalFrameCount; ++i)
        {
            for (unsigned int ch = 0; ch < channels; ++ch)
            {
                output_data[i * channels + ch] = reconstructed_audio[i];
            }
        }

        // Set up output format
        drwav_data_format format;
        format.container = drwav_container_riff;
        format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
        format.channels = channels;
        format.sampleRate = sampleRate;
        format.bitsPerSample = 32;

        // Initialize WAV writer
        drwav wav;
        if (!drwav_init_file_write_sequential_pcm_frames(&wav, output_path, &format, totalFrameCount, nullptr))
        {
            std::cerr << "Error initializing output WAV file: " << output_path << std::endl;
            return false;
        }

        // Write audio data
        drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, totalFrameCount, output_data.data());

        // Cleanup
        drwav_uninit(&wav);

        if (frames_written != totalFrameCount)
        {
            std::cerr << "Warning: Expected to write " << totalFrameCount << " frames, but wrote " << frames_written
                      << std::endl;
        }

        std::cout << "Successfully processed and saved: " << output_path << std::endl;
        std::cout << "Output: " << frames_written << " frames, " << channels << " channels, " << sampleRate << " Hz"
                  << std::endl;

        return true;
    }
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        print_usage();
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // Load audio file
    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalFrameCount;

    auto audio_data = load_audio_file(input_path, channels, sampleRate, totalFrameCount);
    if (audio_data.empty())
    {
        return 1;
    }

    // Convert to mono for processing
    auto mono_audio = convert_to_mono(audio_data, channels, totalFrameCount);

    // Set up STFT parameters
    const int window_size = 2048;
    const int hop_size = 512;
    const size_t fft_bins = window_size / 2 + 1;

    // Create window function
    auto window = create_hanning_window(window_size);

    // Create the complex stft
    const auto original_stft = process_stft(mono_audio, window, window_size, hop_size, fft_bins);

    // Calculate gradients for RTPGHI
    float time_step = static_cast<float>(hop_size) / static_cast<float>(sampleRate);
    float freq_step = static_cast<float>(sampleRate) / static_cast<float>(window_size);

    std::cout << "DEBUG: time_step = " << time_step << " seconds" << std::endl;
    std::cout << "DEBUG: freq_step = " << freq_step << " Hz" << std::endl;
    std::cout << "DEBUG: hop_size = " << hop_size << ", sampleRate = " << sampleRate << std::endl;
    std::cout << "DEBUG: window_size = " << window_size << std::endl;

    std::cout << "Calculating gradients offline..." << std::endl;

    // Use GradientResult for offline gradient calculation
    rtpghi::GradientResult gradients;
    gradients.time_frames = original_stft.size();
    gradients.freq_frames = original_stft.size();
    gradients.fft_bins = fft_bins;
    gradients.time_data.resize(original_stft.size() * fft_bins);
    gradients.freq_data.resize(original_stft.size() * fft_bins);

    rtpghi::GradientOutput gradient_output {
        gradients.time_data.data(), gradients.freq_data.data(), original_stft.size(), original_stft.size(), fft_bins
    };

    // Calculate gradients: BACKWARD for time (correct), FORWARD for frequency (as per paper)
    // Time: BACKWARD represents how we got to current frame
    // Frequency: FORWARD represents direction of propagation
    auto error = rtpghi::calculate_spectrum_gradients(original_stft,
                                                      time_step,
                                                      freq_step,
                                                      gradient_output,
                                                      rtpghi::GradientMethod::BACKWARD,
                                                      rtpghi::GradientMethod::FORWARD);

    if (error != rtpghi::ErrorCode::OK)
    {
        std::cerr << "Error calculating gradients for RTPGHI" << std::endl;
        return 1;
    }

    // Get matrix views for easy access during processing
    auto freq_grad_matrix = gradients.freq_gradients();

    std::cout << "Gradients calculated successfully" << std::endl;
    
    // COMPREHENSIVE FREQUENCY GRADIENT VERIFICATION
    std::cout << "\n=== FREQUENCY GRADIENT VERIFICATION ===" << std::endl;
    std::cout << "Checking offline frequency gradients against original STFT data..." << std::endl;
    
    // Check frame 0 around the fundamental frequency (bin 20 = 440Hz)
    size_t test_frame = 0;
    std::cout << "\nFrame " << test_frame << " frequency gradient verification:" << std::endl;
    std::cout << "freq_step = " << freq_step << " Hz" << std::endl;
    
    for (size_t bin = 18; bin <= 23 && bin < fft_bins-1; ++bin) {
        // Get original phases
        float phase_curr = std::arg(original_stft[test_frame][bin]);
        float phase_next = std::arg(original_stft[test_frame][bin+1]);
        
        // Calculate manual frequency gradient using forward difference
        float manual_phase_diff = rtpghi::princarg(phase_next - phase_curr);
        float manual_freq_grad = manual_phase_diff / freq_step;
        
        // Get calculated frequency gradient
        float calculated_freq_grad = freq_grad_matrix[test_frame][bin];
        
        // Check if they match
        float grad_error = std::abs(calculated_freq_grad - manual_freq_grad);
        
        std::cout << "  Bin " << bin << " (" << (bin * freq_step) << " Hz):" << std::endl;
        std::cout << "    phase_curr: " << phase_curr << " rad" << std::endl;
        std::cout << "    phase_next: " << phase_next << " rad" << std::endl;
        std::cout << "    phase_diff: " << manual_phase_diff << " rad" << std::endl;
        std::cout << "    manual_grad: " << manual_freq_grad << " rad/Hz" << std::endl;
        std::cout << "    calculated_grad: " << calculated_freq_grad << " rad/Hz" << std::endl;
        std::cout << "    error: " << grad_error << " rad/Hz" << std::endl;
        std::cout << "    match: " << (grad_error < 1e-6f ? "YES" : "NO") << std::endl;
        
        if (bin == 20) {
            // Special analysis for fundamental frequency
            std::cout << "    *** FUNDAMENTAL FREQUENCY BIN ***" << std::endl;
            std::cout << "    magnitude: " << std::abs(original_stft[test_frame][bin]) << std::endl;
            std::cout << "    Expected small phase diff for 1x, got: " << manual_phase_diff << " rad" << std::endl;
        }
    }
    // DEBUG: Check frequency gradients for first few frames and bins
    std::cout << "DEBUG: Frequency gradients for first frame, first 5 bins: ";
    for (size_t i = 0; i < 5 && i < fft_bins; ++i)
    {
        std::cout << freq_grad_matrix[0][i] << " ";
    }
    std::cout << std::endl;

    // DEBUG: Calculate expected frequency gradients manually for comparison
    std::cout << "DEBUG: Manual frequency gradient calculation:" << std::endl;
    for (size_t bin = 1; bin < 4 && bin < fft_bins - 1; ++bin)
    {
        if (original_stft.size() >= 1)
        {
            float phase_curr = std::arg(original_stft[0][bin]);
            float phase_next = std::arg(original_stft[0][bin + 1]);
            float expected_freq_gradient = rtpghi::phaseDiff(phase_next, phase_curr) / freq_step;
            std::cout << "  Bin " << bin << ": phase_curr=" << phase_curr << ", phase_next=" << phase_next
                      << ", expected_freq_grad=" << expected_freq_gradient
                      << ", calculated_freq_grad=" << freq_grad_matrix[0][bin] << std::endl;
        }
    }

    // FREQUENCY GRADIENT ANALYSIS: Compare original vs calculated frequency gradients\n    std::cout << \"\\n===
    // FREQUENCY GRADIENT ANALYSIS ===\" << std::endl;\n    std::cout << \"Original frequency relationships (frame 0):\"
    // << std::endl;\n    for (size_t bin = 19; bin <= 23 && bin < fft_bins; ++bin) {\n        float phase =
    // std::arg(original_stft[0][bin]);\n        float magnitude = std::abs(original_stft[0][bin]);\n        float
    // freq_gradient = freq_grad_matrix[0][bin];\n        std::cout << \"  Bin \" << bin << \": phase=\" << phase << \",
    // mag=\" << magnitude << \", freq_grad=\" << freq_gradient << std::endl;\n    }\n    \n    std::cout << \"\\nActual
    // vs Expected frequency propagation:\" << std::endl;\n    for (size_t bin = 19; bin <= 22 && bin < fft_bins-1;
    // ++bin) {\n        float curr_phase = std::arg(original_stft[0][bin]);\n        float next_phase =
    // std::arg(original_stft[0][bin+1]);\n        float actual_phase_diff = rtpghi::phaseDiff(next_phase,
    // curr_phase);\n        float freq_grad = freq_grad_matrix[0][bin];\n        float rtpghi_increment = freq_step *
    // freq_grad;\n        \n        // What we should get for 1x reconstruction:\n        float correct_increment =
    // actual_phase_diff;\n        float correct_freq_grad = actual_phase_diff / freq_step;\n        \n        std::cout
    // << \"  Bin \" << bin << \"->\" << (bin+1) << \":\" << std::endl;\n        std::cout << \"    Actual phase_diff:
    // \" << actual_phase_diff << \" rad\" << std::endl;\n        std::cout << \"    Calculated freq_grad: \" <<
    // freq_grad << \" rad/Hz\" << std::endl;\n        std::cout << \"    RTPGHI increment: \" << rtpghi_increment << \"
    // rad (ERROR!)\" << std::endl;\n        std::cout << \"    Correct increment: \" << correct_increment << \" rad\"
    // << std::endl;\n        std::cout << \"    Correct freq_grad: \" << correct_freq_grad << \" rad/Hz\" <<
    // std::endl;\n        std::cout << \"    Scaling factor needed: \" << (correct_increment / rtpghi_increment) <<
    // std::endl;\n    }\n    \n    // DEBUG: Check original phases for first two frames to see if gradients make sense
    std::cout << "DEBUG: Original phase progression for first 3 bins:" << std::endl;
    for (size_t bin = 0; bin < 3 && bin < fft_bins; ++bin)
    {
        if (original_stft.size() >= 2)
        {
            float phase0 = std::arg(original_stft[0][bin]);
            float phase1 = std::arg(original_stft[1][bin]);
            float expected_gradient = rtpghi::phaseDiff(phase1, phase0) / time_step;
            float freq_hz = expected_gradient / (2.0f * M_PI);  // Convert rad/s to Hz
            std::cout << "  Bin " << bin << ": phase0=" << phase0 << ", phase1=" << phase1
                      << ", expected_grad=" << expected_gradient << " rad/s (" << freq_hz << " Hz)" << std::endl;
        }
    }

    // Create output STFT for 1x playback (same size as original)
    std::vector<std::vector<std::complex<float>>> processed_stft(original_stft.size(),
                                                                 std::vector<std::complex<float>>(fft_bins));

    // Process with RTPGHI Processor for phase reconstruction at 1x speed
    // Use moderate tolerance with frequency propagation enabled
    float test_tolerance = 1e-4f;  // Moderate tolerance to limit frequency propagation
    rtpghi::ProcessorConfig config(fft_bins, time_step, freq_step, test_tolerance);

    std::cout << "RTPGHI config - tolerance: " << test_tolerance << ", integration: Forward Euler" << std::endl;
    rtpghi::Processor rtpghi_processor(config);

    std::cout << "Processing " << original_stft.size() << " frames with RTPGHI Processor..." << std::endl;

    // DEBUG: Check magnitude ranges to understand tolerance effects
    if (original_stft.size() >= 1)
    {
        float max_mag = 0.0f, min_mag = 1e10f;
        for (size_t bin = 0; bin < fft_bins; ++bin)
        {
            float mag = std::abs(original_stft[0][bin]);
            max_mag = std::max(max_mag, mag);
            if (mag > 0)
            {
                min_mag = std::min(min_mag, mag);
            }
        }
        float tolerance_threshold = test_tolerance * max_mag;
        std::cout << "DEBUG: Frame 0 - max_mag=" << max_mag << ", min_mag=" << min_mag
                  << ", tolerance_threshold=" << tolerance_threshold << std::endl;
    }

    // Storage for processing
    std::vector<float> previous_magnitudes(fft_bins);
    std::vector<float> previous_phases(fft_bins);
    std::vector<float> current_magnitudes(fft_bins);
    std::vector<float> output_phases(fft_bins);

    // Process first frame differently - use original phases
    for (size_t bin = 0; bin < fft_bins; ++bin)
    {
        processed_stft[0][bin] = original_stft[0][bin];  // Keep original first frame
        previous_phases[bin] = std::arg(original_stft[0][bin]);
        previous_magnitudes[bin] = std::abs(original_stft[0][bin]);
    }

    // Process remaining frames with RTPGHI Processor
    // NOTE: Fundamental inconsistency - we use gradients calculated from original phases
    // but apply them to reconstructed phases that diverge over time. This causes
    // accumulating errors that manifest as "phasey" artifacts.
    for (size_t frame = 1; frame < original_stft.size(); ++frame)
    {
        // Extract magnitudes from current frame
        for (size_t bin = 0; bin < fft_bins; ++bin)
        {
            current_magnitudes[bin] = std::abs(original_stft[frame][bin]);
        }

        // Use BACKWARD gradients directly - they're already mathematically correct!
        // BACKWARD gradient[N] = (phase[N] - phase[N-1]) / dt represents the rate of change
        // that brought us FROM frame N-1 TO frame N, which is exactly what we need for integration.
        auto time_grad_matrix = gradients.time_gradients();

        // Create RTPGHI input frame
        rtpghi::FrameInput input;
        input.current_magnitudes = current_magnitudes.data();
        input.previous_magnitudes = previous_magnitudes.data();
        input.previous_phases = previous_phases.data();
        input.time_gradients = time_grad_matrix[frame];  // Use BACKWARD offline gradients directly!
        input.freq_gradients = freq_grad_matrix[frame];  // Current freq gradients
        input.prev_time_gradients = (frame > 1)
                                        ? time_grad_matrix[frame - 1]
                                        : time_grad_matrix[frame];  // Previous BACKWARD gradients for trapezoidal
        input.synthesis_time_step = time_step;
        input.synthesis_freq_step = freq_step;
        input.fft_bins = fft_bins;

        // Create RTPGHI output frame
        // DEBUG: Initialize output_phases with known values to detect if RTPGHI processes them
        std::fill(output_phases.begin(), output_phases.end(), -999.0f);  // Sentinel value

        rtpghi::FrameOutput output;
        output.magnitudes = current_magnitudes.data();
        output.phases = output_phases.data();
        output.fft_bins = fft_bins;

        if (frame <= 3)
        {  // Show first 3 frames only
            std::cout << "\n=== FRAME " << frame << " DETAILED COMPARISON ===" << std::endl;
            std::cout << "Previous phases: ";
            for (size_t i = 0; i < 3; ++i)
            {
                std::cout << previous_phases[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "BACKWARD offline gradients: ";
            for (size_t i = 0; i < 3; ++i)
            {
                std::cout << time_grad_matrix[frame][i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Time step: " << time_step << std::endl;

            // Manual calculation using SAME BACKWARD offline gradients as RTPGHI
            std::cout << "MANUAL CALCULATION:" << std::endl;
            for (size_t i = 0; i < 3; ++i)
            {
                float manual_unwrapped = previous_phases[i] + time_step * time_grad_matrix[frame][i];
                float manual_wrapped = rtpghi::wrapPi(manual_unwrapped);
                std::cout << "  Bin " << i << ": " << previous_phases[i] << " + (" << time_step << " * "
                          << time_grad_matrix[frame][i] << ") = " << manual_unwrapped
                          << " → wrapped: " << manual_wrapped << std::endl;
            }

            std::cout << "BEFORE RTPGHI - output_phases (first 3): ";
            for (size_t i = 0; i < 3; ++i)
            {
                std::cout << output_phases[i] << " ";
            }
            std::cout << std::endl;
        }

        // Process with RTPGHI
        auto result = rtpghi_processor.process(input, output);
        if (result != rtpghi::ErrorCode::OK)
        {
            std::cerr << "RTPGHI processing failed at frame " << frame << std::endl;
            return 1;
        }

        if (frame <= 3)
        {  // Show first 3 frames only
            std::cout << "AFTER RTPGHI - output_phases (first 3): ";
            for (size_t i = 0; i < 3; ++i)
            {
                std::cout << output_phases[i] << " ";
            }
            std::cout << std::endl;

            // COMPARE MANUAL vs RTPGHI (using same BACKWARD offline gradients)
            std::cout << "MANUAL vs RTPGHI COMPARISON:" << std::endl;
            for (size_t i = 0; i < 3; ++i)
            {
                float manual_unwrapped = previous_phases[i] + time_step * time_grad_matrix[frame][i];
                float manual_wrapped = rtpghi::wrapPi(manual_unwrapped);
                float rtpghi_phase = output_phases[i];
                float difference = rtpghi::wrapPi(rtpghi_phase - manual_wrapped);
                std::cout << "  Bin " << i << ": Manual=" << manual_wrapped << " vs RTPGHI=" << rtpghi_phase
                          << " (diff=" << difference << ")" << std::endl;
            }
        }

        // DEBUG: Check which bins got random phases
        if (frame <= 3)
        {  // Show first 3 frames only
            std::cout << "Bins with random phases (checking first 20): ";
            for (size_t i = 0; i < 20 && i < fft_bins; ++i)
            {
                // A bin likely got a random phase if its magnitude is below tolerance
                // or if the phase is very different from expected
                float expected_phase = previous_phases[i] + time_step * time_grad_matrix[frame][i];
                expected_phase = rtpghi::wrapPi(expected_phase);
                float phase_error = std::abs(rtpghi::wrapPi(output_phases[i] - expected_phase));

                if (phase_error > 0.1f)
                {  // More than 0.1 radians off
                    std::cout << i << "(" << phase_error << ") ";
                }
            }
            std::cout << std::endl;
        }

        // Update processed STFT with RTPGHI results
        for (size_t bin = 0; bin < fft_bins; ++bin)
        {
            processed_stft[frame][bin] = std::polar(output.magnitudes[bin], output.phases[bin]);
        }

        // Save current data for next iteration
        previous_magnitudes = current_magnitudes;
        std::copy(output_phases.begin(), output_phases.end(), previous_phases.begin());

        if (frame % 100 == 0)
        {
            std::cout << "RTPGHI processed frame " << frame << "/" << original_stft.size() << std::endl;
        }

        // DEBUG: Check first few frames for obvious issues
        if (frame <= 3)
        {  // Show first 3 frames only
            std::cout << "\n=== Frame " << frame << " Debug ===" << std::endl;

            // Show magnitudes
            std::cout << "Magnitudes (first 5): ";
            for (size_t i = 0; i < 5 && i < fft_bins; ++i)
            {
                std::cout << current_magnitudes[i] << " ";
            }
            std::cout << std::endl;

            // Show which bins are significant
            std::cout << "Significant bins (first 10): ";
            int sig_count = 0;
            for (size_t i = 0; i < 10 && i < fft_bins; ++i)
            {
                if (current_magnitudes[i] > test_tolerance * 0.0084849f)
                {  // Using the max_mag from frame 0
                    std::cout << i << " ";
                    sig_count++;
                }
            }
            std::cout << " (total: " << sig_count << " in first 10)" << std::endl;

            // Show time gradients being used
            std::cout << "BACKWARD offline gradients used (first 3): ";
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                std::cout << time_grad_matrix[frame][i] << " ";
            }
            std::cout << std::endl;

            // Show previous phases
            std::cout << "Previous phases: ";
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                std::cout << previous_phases[i] << " ";
            }
            std::cout << std::endl;

            // Show phase integration step by step for first 3 bins
            std::cout << "Phase integration for first 3 bins:" << std::endl;
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                float phase_increment = time_step * time_grad_matrix[frame][i];
                float new_phase = previous_phases[i] + phase_increment;
                std::cout << "  Bin " << i << ": prev=" << previous_phases[i] << " + (dt=" << time_step
                          << " * backward_grad=" << time_grad_matrix[frame][i] << ") = " << new_phase
                          << " -> wrapped: " << rtpghi::wrapPi(new_phase) << std::endl;
            }

            // Original comparison
            std::cout << "Frame " << frame << " - Original phases: ";
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                std::cout << std::arg(original_stft[frame][i]) << " ";
            }
            std::cout << std::endl;
            std::cout << "Frame " << frame << " - RTPGHI phases:  ";
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                std::cout << output_phases[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Frame " << frame << " - Phase diffs:   ";
            float total_phase_error = 0.0f;
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                float diff = rtpghi::wrapPi(output_phases[i] - std::arg(original_stft[frame][i]));
                total_phase_error += std::abs(diff);
                std::cout << diff << " ";
            }
            std::cout << " (avg error: " << total_phase_error / 3.0f << " rad)" << std::endl;

            // MANUAL TIME INTEGRATION CHECK: Compare RTPGHI vs simple integration (with BACKWARD offline gradients)
            std::cout << "Manual integration check for first 3 bins:" << std::endl;
            for (size_t i = 0; i < 3 && i < fft_bins; ++i)
            {
                if (frame > 0)
                {
                    float manual_phase = previous_phases[i] + time_step * time_grad_matrix[frame][i];
                    manual_phase = rtpghi::wrapPi(manual_phase);
                    float rtpghi_phase = output_phases[i];
                    float manual_error = rtpghi::wrapPi(rtpghi_phase - manual_phase);
                    std::cout << "  Bin " << i << ": manual=" << manual_phase << " vs RTPGHI=" << rtpghi_phase
                              << " (diff=" << manual_error << ")" << std::endl;
                }
            }

            // Check if phase errors are growing over time
            static float prev_error = 0.0f;
            if (frame > 0)
            {
                float error_growth = total_phase_error / 3.0f - prev_error;
                std::cout << "Phase error growth: " << error_growth << " rad/frame" << std::endl;
            }
            prev_error = total_phase_error / 3.0f;
        }
    }

    // Reconstruct audio at original length (1x playback)
    auto reconstructed_audio = reconstruct_audio(processed_stft, window, window_size, hop_size, mono_audio.size());

    // DEBUG: Check first few samples of reconstructed audio
    std::cout << "\nDEBUG: Audio reconstruction check:" << std::endl;
    std::cout << "Original audio (first 10 samples): ";
    for (size_t i = 0; i < 10 && i < mono_audio.size(); ++i)
    {
        std::cout << mono_audio[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Reconstructed audio (first 10 samples): ";
    for (size_t i = 0; i < 10 && i < reconstructed_audio.size(); ++i)
    {
        std::cout << reconstructed_audio[i] << " ";
    }
    std::cout << std::endl;

    // Check RMS levels
    float orig_rms = 0.0f, recon_rms = 0.0f;
    size_t check_samples = std::min(mono_audio.size(), reconstructed_audio.size());
    for (size_t i = 0; i < check_samples; ++i)
    {
        orig_rms += mono_audio[i] * mono_audio[i];
        recon_rms += reconstructed_audio[i] * reconstructed_audio[i];
    }
    orig_rms = std::sqrt(orig_rms / check_samples);
    recon_rms = std::sqrt(recon_rms / check_samples);
    std::cout << "RMS levels - Original: " << orig_rms << ", Reconstructed: " << recon_rms
              << ", Ratio: " << (recon_rms / orig_rms) << std::endl;

    // Write output file with original frame count
    if (!write_audio_file(output_path, reconstructed_audio, channels, sampleRate, totalFrameCount))
    {
        return 1;
    }

    return 0;
}
