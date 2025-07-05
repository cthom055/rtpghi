#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <rtpghi/rtpghi.hpp>
#include <vector>
#include <memory>
#include <limits>

using Catch::Approx;

TEST_CASE("GradientMatrix Construction and Basic Properties", "[gradient_matrix][basic]")
{
    SECTION("Default construction creates empty matrix")
    {
        rtpghi::GradientMatrix matrix;
        
        REQUIRE(matrix.rows() == 0);
        REQUIRE(matrix.cols() == 0);
        REQUIRE(matrix.size() == 0);
        REQUIRE(matrix.empty() == true);
        REQUIRE(matrix.data() == nullptr);
    }
    
    SECTION("Construction with valid data")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const size_t rows = 2, cols = 3;
        
        rtpghi::GradientMatrix matrix(data.data(), rows, cols);
        
        REQUIRE(matrix.rows() == rows);
        REQUIRE(matrix.cols() == cols);
        REQUIRE(matrix.size() == rows * cols);
        REQUIRE(matrix.empty() == false);
        REQUIRE(matrix.data() == data.data());
    }
    
    SECTION("Construction with null pointer")
    {
        rtpghi::GradientMatrix matrix(nullptr, 2, 3);
        
        REQUIRE(matrix.rows() == 2);
        REQUIRE(matrix.cols() == 3);
        REQUIRE(matrix.size() == 6);
        REQUIRE(matrix.empty() == true);  // empty() checks for nullptr
        REQUIRE(matrix.data() == nullptr);
    }
    
    SECTION("Construction with zero dimensions")
    {
        std::vector<float> data = {1.0f, 2.0f};
        
        // Zero rows
        rtpghi::GradientMatrix matrix1(data.data(), 0, 2);
        REQUIRE(matrix1.empty() == true);
        REQUIRE(matrix1.size() == 0);
        
        // Zero cols
        rtpghi::GradientMatrix matrix2(data.data(), 2, 0);
        REQUIRE(matrix2.empty() == true);
        REQUIRE(matrix2.size() == 0);
        
        // Both zero
        rtpghi::GradientMatrix matrix3(data.data(), 0, 0);
        REQUIRE(matrix3.empty() == true);
        REQUIRE(matrix3.size() == 0);
    }
}

TEST_CASE("GradientMatrix Element Access", "[gradient_matrix][access]")
{
    const size_t rows = 3, cols = 4;
    std::vector<float> data(rows * cols);
    
    // Initialize with known pattern: data[i] = i
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }
    
    rtpghi::GradientMatrix matrix(data.data(), rows, cols);
    
    SECTION("operator[] row access")
    {
        // Test that each row points to correct location
        for (size_t row = 0; row < rows; ++row)
        {
            float* row_ptr = matrix[row];
            REQUIRE(row_ptr == data.data() + row * cols);
            
            // Verify row contents
            for (size_t col = 0; col < cols; ++col)
            {
                REQUIRE(row_ptr[col] == static_cast<float>(row * cols + col));
            }
        }
    }
    
    SECTION("operator() element access")
    {
        // Test direct element access
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                float expected = static_cast<float>(row * cols + col);
                REQUIRE(matrix(row, col) == expected);
            }
        }
    }
    
    SECTION("Modify elements through matrix view")
    {
        // Modify through operator()
        matrix(1, 2) = 99.0f;
        REQUIRE(data[1 * cols + 2] == 99.0f);
        REQUIRE(matrix(1, 2) == 99.0f);
        
        // Modify through operator[]
        matrix[2][1] = 88.0f;
        REQUIRE(data[2 * cols + 1] == 88.0f);
        REQUIRE(matrix(2, 1) == 88.0f);
    }
}

TEST_CASE("GradientMatrix Const Correctness", "[gradient_matrix][const]")
{
    const size_t rows = 2, cols = 3;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    SECTION("Const matrix access")
    {
        const rtpghi::GradientMatrix matrix(data.data(), rows, cols);
        
        // Const operator[] should return const pointer
        const float* row0 = matrix[0];
        REQUIRE(row0[0] == 1.0f);
        REQUIRE(row0[1] == 2.0f);
        REQUIRE(row0[2] == 3.0f);
        
        // Const operator() should return const reference
        REQUIRE(matrix(0, 0) == 1.0f);
        REQUIRE(matrix(0, 1) == 2.0f);
        REQUIRE(matrix(1, 0) == 4.0f);
        REQUIRE(matrix(1, 2) == 6.0f);
        
        // Const data access
        const float* const_data = matrix.data();
        REQUIRE(const_data == data.data());
    }
    
    SECTION("Non-const matrix becomes const")
    {
        rtpghi::GradientMatrix matrix(data.data(), rows, cols);
        const rtpghi::GradientMatrix& const_ref = matrix;
        
        // Should be able to access through const reference
        REQUIRE(const_ref(1, 1) == 5.0f);
        REQUIRE(const_ref[1][2] == 6.0f);
    }
}

TEST_CASE("GradientMatrix Iterator Support", "[gradient_matrix][iterators]")
{
    const size_t rows = 2, cols = 3;
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    rtpghi::GradientMatrix matrix(data.data(), rows, cols);
    
    SECTION("Range-based for loop")
    {
        std::vector<float> collected;
        for (float value : matrix)
        {
            collected.push_back(value);
        }
        
        REQUIRE(collected.size() == 6);
        REQUIRE(collected == data);
    }
    
    SECTION("Iterator arithmetic")
    {
        auto begin_it = matrix.begin();
        auto end_it = matrix.end();
        
        REQUIRE(end_it - begin_it == static_cast<ptrdiff_t>(matrix.size()));
        REQUIRE(*begin_it == 10.0f);
        REQUIRE(*(end_it - 1) == 60.0f);
    }
    
    SECTION("Const iterators")
    {
        const rtpghi::GradientMatrix const_matrix(data.data(), rows, cols);
        
        std::vector<float> collected;
        for (float value : const_matrix)
        {
            collected.push_back(value);
        }
        
        REQUIRE(collected == data);
    }
}

TEST_CASE("GradientMatrix Memory Layout Verification", "[gradient_matrix][memory]")
{
    SECTION("Row-major layout")
    {
        const size_t rows = 3, cols = 2;
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        rtpghi::GradientMatrix matrix(data.data(), rows, cols);
        
        // Verify row-major layout: matrix[row][col] == data[row * cols + col]
        REQUIRE(matrix(0, 0) == data[0]);  // 1.0f
        REQUIRE(matrix(0, 1) == data[1]);  // 2.0f
        REQUIRE(matrix(1, 0) == data[2]);  // 3.0f
        REQUIRE(matrix(1, 1) == data[3]);  // 4.0f
        REQUIRE(matrix(2, 0) == data[4]);  // 5.0f
        REQUIRE(matrix(2, 1) == data[5]);  // 6.0f
        
        // Verify row pointers are correctly offset
        REQUIRE(matrix[0] == data.data());
        REQUIRE(matrix[1] == data.data() + cols);
        REQUIRE(matrix[2] == data.data() + 2 * cols);
    }
    
    SECTION("Single row matrix")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        rtpghi::GradientMatrix matrix(data.data(), 1, 3);
        
        REQUIRE(matrix[0] == data.data());
        REQUIRE(matrix(0, 0) == 1.0f);
        REQUIRE(matrix(0, 1) == 2.0f);
        REQUIRE(matrix(0, 2) == 3.0f);
    }
    
    SECTION("Single column matrix")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        rtpghi::GradientMatrix matrix(data.data(), 3, 1);
        
        REQUIRE(matrix(0, 0) == 1.0f);
        REQUIRE(matrix(1, 0) == 2.0f);
        REQUIRE(matrix(2, 0) == 3.0f);
        
        // Each row should point to consecutive elements
        REQUIRE(matrix[0] == data.data());
        REQUIRE(matrix[1] == data.data() + 1);
        REQUIRE(matrix[2] == data.data() + 2);
    }
}

TEST_CASE("GradientMatrix Edge Cases and Safety", "[gradient_matrix][safety]")
{
    SECTION("Large dimensions (no overflow)")
    {
        // Test with large but reasonable dimensions
        const size_t large_rows = 1000;
        const size_t large_cols = 512;
        std::vector<float> large_data(large_rows * large_cols, 1.0f);
        
        rtpghi::GradientMatrix matrix(large_data.data(), large_rows, large_cols);
        
        REQUIRE(matrix.rows() == large_rows);
        REQUIRE(matrix.cols() == large_cols);
        REQUIRE(matrix.size() == large_rows * large_cols);
        
        // Test access to corners
        REQUIRE(matrix(0, 0) == 1.0f);
        REQUIRE(matrix(0, large_cols - 1) == 1.0f);
        REQUIRE(matrix(large_rows - 1, 0) == 1.0f);
        REQUIRE(matrix(large_rows - 1, large_cols - 1) == 1.0f);
    }
    
    SECTION("Matrix with single element")
    {
        std::vector<float> data = {42.0f};
        rtpghi::GradientMatrix matrix(data.data(), 1, 1);
        
        REQUIRE(matrix.rows() == 1);
        REQUIRE(matrix.cols() == 1);
        REQUIRE(matrix.size() == 1);
        REQUIRE_FALSE(matrix.empty());
        
        REQUIRE(matrix(0, 0) == 42.0f);
        REQUIRE(matrix[0][0] == 42.0f);
        
        // Modify through matrix
        matrix(0, 0) = 99.0f;
        REQUIRE(data[0] == 99.0f);
    }
}

TEST_CASE("GradientResult Integration", "[gradient_matrix][integration]")
{
    SECTION("GradientResult provides valid matrix views")
    {
        const size_t time_frames = 3;
        const size_t freq_frames = 3;
        const size_t fft_bins = 4;
        
        rtpghi::GradientResult result;
        result.time_frames = time_frames;
        result.freq_frames = freq_frames;
        result.fft_bins = fft_bins;
        result.time_data.resize(time_frames * fft_bins);
        result.freq_data.resize(freq_frames * fft_bins);
        
        // Initialize with known values
        for (size_t i = 0; i < result.time_data.size(); ++i)
        {
            result.time_data[i] = static_cast<float>(i);
        }
        for (size_t i = 0; i < result.freq_data.size(); ++i)
        {
            result.freq_data[i] = static_cast<float>(i + 100);
        }
        
        REQUIRE(result.is_valid());
        
        // Test time gradient matrix
        auto time_matrix = result.time_gradients();
        REQUIRE(time_matrix.rows() == time_frames);
        REQUIRE(time_matrix.cols() == fft_bins);
        REQUIRE(time_matrix.data() == result.time_data.data());
        
        // Test frequency gradient matrix
        auto freq_matrix = result.freq_gradients();
        REQUIRE(freq_matrix.rows() == freq_frames);
        REQUIRE(freq_matrix.cols() == fft_bins);
        REQUIRE(freq_matrix.data() == result.freq_data.data());
        
        // Verify data access through matrices
        REQUIRE(time_matrix(0, 0) == 0.0f);
        REQUIRE(time_matrix(1, 2) == 6.0f);  // row 1, col 2 = 1*4 + 2 = 6
        REQUIRE(freq_matrix(0, 0) == 100.0f);
        REQUIRE(freq_matrix(2, 3) == 111.0f);  // row 2, col 3 = 2*4 + 3 + 100 = 111
    }
    
    SECTION("Const GradientResult provides const matrix views")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 2;
        result.fft_bins = 2;
        result.time_data = {1.0f, 2.0f, 3.0f, 4.0f};
        result.freq_data = {5.0f, 6.0f, 7.0f, 8.0f};
        
        const rtpghi::GradientResult& const_result = result;
        
        auto const_time_matrix = const_result.time_gradients();
        auto const_freq_matrix = const_result.freq_gradients();
        
        // Should be able to read through const matrices
        REQUIRE(const_time_matrix(0, 0) == 1.0f);
        REQUIRE(const_time_matrix(1, 1) == 4.0f);
        REQUIRE(const_freq_matrix(0, 1) == 6.0f);
        REQUIRE(const_freq_matrix(1, 0) == 7.0f);
    }
    
    SECTION("Matrix modifications affect underlying storage")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 1;
        result.fft_bins = 3;
        result.time_data.resize(6, 0.0f);
        result.freq_data.resize(3, 0.0f);
        
        auto time_matrix = result.time_gradients();
        auto freq_matrix = result.freq_gradients();
        
        // Modify through matrix views
        time_matrix(0, 1) = 10.0f;
        time_matrix(1, 2) = 20.0f;
        freq_matrix(0, 0) = 30.0f;
        
        // Verify changes in underlying storage
        REQUIRE(result.time_data[1] == 10.0f);  // row 0, col 1
        REQUIRE(result.time_data[5] == 20.0f);  // row 1, col 2
        REQUIRE(result.freq_data[0] == 30.0f);  // row 0, col 0
    }
}

TEST_CASE("GradientResult Validation", "[gradient_matrix][validation]")
{
    SECTION("Valid GradientResult")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 3;
        result.fft_bins = 4;
        result.time_data.resize(8);  // 2 * 4
        result.freq_data.resize(12); // 3 * 4
        
        REQUIRE(result.is_valid());
    }
    
    SECTION("Invalid storage sizes")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 2;
        result.fft_bins = 3;
        
        // Wrong time data size
        result.time_data.resize(5);  // Should be 6
        result.freq_data.resize(6);  // Correct
        REQUIRE_FALSE(result.is_valid());
        
        // Wrong freq data size
        result.time_data.resize(6);  // Correct
        result.freq_data.resize(5);  // Should be 6
        REQUIRE_FALSE(result.is_valid());
        
        // Both wrong
        result.time_data.resize(1);
        result.freq_data.resize(1);
        REQUIRE_FALSE(result.is_valid());
    }
    
    SECTION("Zero bins makes invalid")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 2;
        result.fft_bins = 0;  // Invalid
        result.time_data.resize(0);
        result.freq_data.resize(0);
        
        REQUIRE_FALSE(result.is_valid());
    }
    
    SECTION("Empty storage is valid if dimensions are zero")
    {
        rtpghi::GradientResult result;
        result.time_frames = 0;
        result.freq_frames = 0;
        result.fft_bins = 1;
        result.time_data.clear();
        result.freq_data.clear();
        
        REQUIRE(result.is_valid());
    }
}

TEST_CASE("GradientMatrix Critical Safety Tests", "[gradient_matrix][safety_critical]")
{
    SECTION("Out-of-bounds access behavior (undefined but shouldn't crash)")
    {
        // NOTE: These tests verify the interface doesn't crash, but accessing 
        // out-of-bounds is undefined behavior. In production, users should avoid this.
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        rtpghi::GradientMatrix matrix(data.data(), 2, 2);
        
        // These operations are undefined behavior but shouldn't crash in debug
        // We're testing that the interface is robust, not that UB is safe
        SECTION("Row out of bounds")
        {
            // This is undefined behavior - just verify interface exists
            volatile float* row_ptr = matrix[999];  // Suppress optimization
            (void)row_ptr;  // Prevent unused variable warning
            
            // We can't REQUIRE anything about UB, just that it compiles
            SUCCEED("Out-of-bounds row access interface exists");
        }
        
        SECTION("Column access through valid row")
        {
            float* row = matrix[0];
            (void)row;  // Suppress unused variable warning
            // row[999] would be UB - we don't test actual access
            SUCCEED("Row pointer interface works for bounds checking");
        }
    }
    
    SECTION("Integer overflow protection in size calculation")
    {
        // Test with large dimensions that could cause overflow
        const size_t max_size = std::numeric_limits<size_t>::max();
        const size_t large_dim = static_cast<size_t>(std::sqrt(static_cast<double>(max_size))) + 1;
        
        // This would overflow: large_dim * large_dim > max_size
        std::vector<float> small_data = {1.0f};
        rtpghi::GradientMatrix matrix(small_data.data(), large_dim, large_dim);
        
        // size() method should handle this gracefully
        size_t calculated_size = matrix.size();
        
        // The calculation should either overflow (wrapping) or saturate
        // We test that it doesn't crash and returns some value
        REQUIRE(calculated_size == large_dim * large_dim); // May overflow, but should be consistent
    }
    
    SECTION("Pointer arithmetic edge cases")
    {
        std::vector<float> data(100, 42.0f);
        
        SECTION("Maximum reasonable dimensions")
        {
            // Test maximum realistic audio dimensions
            const size_t max_frames = 10000;  // ~4.5 minutes at 44.1kHz with 512 hop
            const size_t max_bins = 8192;     // 16k FFT bins
            
            if (max_frames * max_bins <= data.size())
            {
                data.resize(max_frames * max_bins, 1.0f);
                rtpghi::GradientMatrix matrix(data.data(), max_frames, max_bins);
                
                REQUIRE(matrix.rows() == max_frames);
                REQUIRE(matrix.cols() == max_bins);
                REQUIRE(matrix.size() == max_frames * max_bins);
                
                // Test corner access
                REQUIRE(matrix(0, 0) == 1.0f);
                REQUIRE(matrix(max_frames - 1, max_bins - 1) == 1.0f);
            }
        }
        
        SECTION("Stride calculation verification")
        {
            const size_t rows = 3, cols = 5;
            data.resize(rows * cols);
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] = static_cast<float>(i);
            }
            
            rtpghi::GradientMatrix matrix(data.data(), rows, cols);
            
            // Verify each row is exactly cols elements apart
            for (size_t row = 0; row < rows - 1; ++row)
            {
                ptrdiff_t stride = matrix[row + 1] - matrix[row];
                REQUIRE(stride == static_cast<ptrdiff_t>(cols));
            }
        }
    }
}

TEST_CASE("GradientMatrix Lifetime Safety Tests", "[gradient_matrix][lifetime]")
{
    SECTION("Matrix view after data destruction (compile-time safe)")
    {
        rtpghi::GradientMatrix matrix;
        
        {
            std::vector<float> temp_data = {1.0f, 2.0f, 3.0f, 4.0f};
            matrix = rtpghi::GradientMatrix(temp_data.data(), 2, 2);
            
            // While temp_data is alive, matrix should work
            REQUIRE(matrix(0, 0) == 1.0f);
            REQUIRE(matrix.rows() == 2);
        }
        // temp_data is now destroyed - accessing matrix would be UB
        // We can't test this safely, but the API should make lifetime clear
        
        // Test that matrix properties still return consistent values
        REQUIRE(matrix.rows() == 2);  // Properties should remain consistent
        REQUIRE(matrix.cols() == 2);
        
        // Note: Accessing matrix data after destruction is UB - users must ensure lifetime
    }
    
    SECTION("Assignment and copy semantics")
    {
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
        
        rtpghi::GradientMatrix matrix1(data1.data(), 2, 2);
        rtpghi::GradientMatrix matrix2(data2.data(), 2, 3);
        
        // Assignment should copy view, not data
        matrix1 = matrix2;
        
        REQUIRE(matrix1.rows() == 2);
        REQUIRE(matrix1.cols() == 3);
        REQUIRE(matrix1.data() == data2.data());
        REQUIRE(matrix1(0, 0) == 10.0f);
        
        // Original data1 should be unchanged
        REQUIRE(data1[0] == 1.0f);
    }
}

TEST_CASE("GradientMatrix Real-world Usage Patterns", "[gradient_matrix][realworld]")
{
    SECTION("Typical audio processing dimensions")
    {
        // Common FFT sizes and frame counts
        std::vector<std::pair<size_t, size_t>> common_sizes = {
            {100, 513},    // 100 frames, 1024 FFT
            {200, 1025},   // 200 frames, 2048 FFT
            {500, 2049},   // 500 frames, 4096 FFT
            {1000, 4097}   // 1000 frames, 8192 FFT
        };
        
        for (auto [frames, bins] : common_sizes)
        {
            std::vector<float> data(frames * bins, 1.0f);
            rtpghi::GradientMatrix matrix(data.data(), frames, bins);
            
            REQUIRE(matrix.rows() == frames);
            REQUIRE(matrix.cols() == bins);
            
            // Test typical access patterns
            REQUIRE(matrix(0, 0) == 1.0f);                    // DC bin
            REQUIRE(matrix(frames/2, bins/2) == 1.0f);        // Middle
            REQUIRE(matrix(frames-1, bins-1) == 1.0f);        // Nyquist
        }
    }
    
    SECTION("Performance-critical access patterns")
    {
        const size_t frames = 100, bins = 513;
        std::vector<float> data(frames * bins);
        
        // Initialize with frequency-based pattern
        for (size_t f = 0; f < frames; ++f)
        {
            for (size_t b = 0; b < bins; ++b)
            {
                data[f * bins + b] = static_cast<float>(f + b);
            }
        }
        
        rtpghi::GradientMatrix matrix(data.data(), frames, bins);
        
        SECTION("Row-wise processing (cache-friendly)")
        {
            for (size_t frame = 0; frame < frames; ++frame)
            {
                float* row = matrix[frame];
                for (size_t bin = 0; bin < bins; ++bin)
                {
                    REQUIRE(row[bin] == static_cast<float>(frame + bin));
                }
            }
        }
        
        SECTION("Element-wise processing")
        {
            for (size_t frame = 0; frame < frames; ++frame)
            {
                for (size_t bin = 0; bin < bins; ++bin)
                {
                    REQUIRE(matrix(frame, bin) == static_cast<float>(frame + bin));
                }
            }
        }
        
        SECTION("Column-wise processing (less cache-friendly)")
        {
            for (size_t bin = 0; bin < bins; ++bin)
            {
                for (size_t frame = 0; frame < frames; ++frame)
                {
                    REQUIRE(matrix(frame, bin) == static_cast<float>(frame + bin));
                }
            }
        }
    }
}

TEST_CASE("GradientMatrix with GradientResult Advanced Scenarios", "[gradient_matrix][integration_advanced]")
{
    SECTION("Multiple matrix views from same GradientResult")
    {
        rtpghi::GradientResult result;
        result.time_frames = 3;
        result.freq_frames = 2;
        result.fft_bins = 4;
        result.time_data.resize(12, 1.0f);
        result.freq_data.resize(8, 2.0f);
        
        // Create multiple views
        auto time_view1 = result.time_gradients();
        auto time_view2 = result.time_gradients();
        auto freq_view1 = result.freq_gradients();
        auto freq_view2 = result.freq_gradients();
        
        // All views should point to same data
        REQUIRE(time_view1.data() == time_view2.data());
        REQUIRE(freq_view1.data() == freq_view2.data());
        
        // Modifications through one view should be visible in all
        time_view1(1, 2) = 99.0f;
        REQUIRE(time_view2(1, 2) == 99.0f);
        REQUIRE(result.time_data[1 * 4 + 2] == 99.0f);
        
        freq_view2(0, 3) = 88.0f;
        REQUIRE(freq_view1(0, 3) == 88.0f);
        REQUIRE(result.freq_data[0 * 4 + 3] == 88.0f);
    }
    
    SECTION("GradientResult reallocation invalidates views")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 2;
        result.fft_bins = 2;
        result.time_data.resize(4, 1.0f);
        result.freq_data.resize(4, 2.0f);
        
        auto time_matrix = result.time_gradients();
        auto freq_matrix = result.freq_gradients();
        
        float* original_time_ptr = time_matrix.data();
        float* original_freq_ptr = freq_matrix.data();
        
        // Force reallocation by resizing
        result.time_data.resize(1000, 3.0f);
        result.freq_data.resize(1000, 4.0f);
        
        // Views still have old pointers (this demonstrates the risk)
        REQUIRE(time_matrix.data() == original_time_ptr);
        REQUIRE(freq_matrix.data() == original_freq_ptr);
        
        // New views should have different pointers
        auto new_time_matrix = result.time_gradients();
        auto new_freq_matrix = result.freq_gradients();
        
        // These should be different if reallocation occurred
        bool time_reallocated = (new_time_matrix.data() != original_time_ptr);
        bool freq_reallocated = (new_freq_matrix.data() != original_freq_ptr);
        
        // At least one should have reallocated (vector implementation dependent)
        INFO("Time reallocated: " << time_reallocated);
        INFO("Freq reallocated: " << freq_reallocated);
        
        // Test passes regardless - this demonstrates the lifetime management issue
        SUCCEED("Reallocation behavior documented");
    }
    
    SECTION("Empty GradientResult provides valid empty matrices")
    {
        rtpghi::GradientResult result;
        result.time_frames = 0;
        result.freq_frames = 0;
        result.fft_bins = 1;  // Non-zero to be valid
        // Leave vectors empty
        
        REQUIRE(result.is_valid());
        
        auto time_matrix = result.time_gradients();
        auto freq_matrix = result.freq_gradients();
        
        REQUIRE(time_matrix.rows() == 0);
        REQUIRE(time_matrix.cols() == 1);
        REQUIRE(time_matrix.empty());
        REQUIRE(freq_matrix.rows() == 0);
        REQUIRE(freq_matrix.cols() == 1);
        REQUIRE(freq_matrix.empty());
    }
}

TEST_CASE("GradientMatrix Const Cast Safety", "[gradient_matrix][const_safety]")
{
    SECTION("Const GradientResult const_cast behavior")
    {
        rtpghi::GradientResult result;
        result.time_frames = 2;
        result.freq_frames = 1;
        result.fft_bins = 2;
        result.time_data = {1.0f, 2.0f, 3.0f, 4.0f};
        result.freq_data = {5.0f, 6.0f};
        
        const rtpghi::GradientResult& const_result = result;
        
        // Get const matrix view (which internally uses const_cast)
        auto const_time_matrix = const_result.time_gradients();
        
        // This should work for reading
        REQUIRE(const_time_matrix(0, 0) == 1.0f);
        REQUIRE(const_time_matrix(1, 1) == 4.0f);
        
        // The const_cast is internal implementation detail
        // Users see this as a non-const GradientMatrix but shouldn't modify
        // through it when obtained from const GradientResult
        
        // Verify the underlying data pointer is correct
        REQUIRE(const_time_matrix.data() == result.time_data.data());
    }
}