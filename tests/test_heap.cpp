#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <rtpghi/rtpghi.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <queue>

using namespace rtpghi;

TEST_CASE("Heap basic functionality", "[heap]")
{
    SECTION("Constructor and basic properties")
    {
        Heap heap(10);
        REQUIRE(heap.empty());
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.capacity() == 10);
    }

    SECTION("Single element insert and extract")
    {
        Heap heap(5);
        heap.insert(42, 3.14f);
        REQUIRE_FALSE(heap.empty());
        REQUIRE(heap.size() == 1);

        HeapEntry entry = heap.extract_max();
        REQUIRE(entry.bin_index == 42);
        REQUIRE(entry.magnitude == Catch::Approx(3.14f));
        REQUIRE(heap.empty());
        REQUIRE(heap.size() == 0);
    }

    SECTION("Multiple element insert and extract (max-heap property)")
    {
        Heap heap(5);
        heap.insert(1, 1.0f);
        heap.insert(2, 5.0f);
        heap.insert(3, 3.0f);
        heap.insert(4, 2.0f);

        REQUIRE(heap.size() == 4);

        // Should extract in descending order of magnitude
        HeapEntry entry1 = heap.extract_max();
        REQUIRE(entry1.bin_index == 2);
        REQUIRE(entry1.magnitude == Catch::Approx(5.0f));

        HeapEntry entry2 = heap.extract_max();
        REQUIRE(entry2.bin_index == 3);
        REQUIRE(entry2.magnitude == Catch::Approx(3.0f));

        HeapEntry entry3 = heap.extract_max();
        REQUIRE(entry3.bin_index == 4);
        REQUIRE(entry3.magnitude == Catch::Approx(2.0f));

        HeapEntry entry4 = heap.extract_max();
        REQUIRE(entry4.bin_index == 1);
        REQUIRE(entry4.magnitude == Catch::Approx(1.0f));

        REQUIRE(heap.empty());
    }

    SECTION("Clear functionality")
    {
        Heap heap(5);
        heap.insert(1, 1.0f);
        heap.insert(2, 2.0f);
        heap.insert(3, 3.0f);

        REQUIRE(heap.size() == 3);
        heap.clear();
        REQUIRE(heap.empty());
        REQUIRE(heap.size() == 0);
    }

    SECTION("Capacity limit")
    {
        Heap heap(2);
        heap.insert(1, 1.0f);
        heap.insert(2, 2.0f);
        REQUIRE(heap.size() == 2);

        // Should ignore insert when at capacity
        heap.insert(3, 3.0f);
        REQUIRE(heap.size() == 2);

        // Should still extract correctly
        HeapEntry entry = heap.extract_max();
        REQUIRE(entry.magnitude == Catch::Approx(2.0f));
    }
}

TEST_CASE("Heap stress test with random data", "[heap]")
{
    SECTION("Random insert and extract maintains heap property")
    {
        constexpr size_t TEST_SIZE = 1000;
        Heap heap(TEST_SIZE);
        std::vector<float> magnitudes;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);

        // Insert random values
        for (size_t i = 0; i < TEST_SIZE; ++i)
        {
            float magnitude = dist(rng);
            magnitudes.push_back(magnitude);
            heap.insert(i, magnitude);
        }

        REQUIRE(heap.size() == TEST_SIZE);

        // Sort magnitudes in descending order
        std::sort(magnitudes.begin(), magnitudes.end(), std::greater<float>());

        // Extract and verify order
        for (size_t i = 0; i < TEST_SIZE; ++i)
        {
            HeapEntry entry = heap.extract_max();
            REQUIRE(entry.magnitude == Catch::Approx(magnitudes[i]));
        }

        REQUIRE(heap.empty());
    }
}

TEST_CASE("Heap with duplicate magnitudes", "[heap]")
{
    SECTION("Handles duplicate values correctly")
    {
        Heap heap(5);
        heap.insert(1, 5.0f);
        heap.insert(2, 5.0f);
        heap.insert(3, 3.0f);
        heap.insert(4, 5.0f);

        std::vector<float> extracted_magnitudes;
        std::vector<size_t> extracted_indices;

        while (!heap.empty())
        {
            HeapEntry entry = heap.extract_max();
            extracted_magnitudes.push_back(entry.magnitude);
            extracted_indices.push_back(entry.bin_index);
        }

        // All 5.0f values should come first
        REQUIRE(extracted_magnitudes[0] == Catch::Approx(5.0f));
        REQUIRE(extracted_magnitudes[1] == Catch::Approx(5.0f));
        REQUIRE(extracted_magnitudes[2] == Catch::Approx(5.0f));
        REQUIRE(extracted_magnitudes[3] == Catch::Approx(3.0f));

        // The indices should be some permutation of 1, 2, 4 for the 5.0f values
        std::vector<size_t> expected_indices = {1, 2, 4};
        std::vector<size_t> actual_indices = {extracted_indices[0], extracted_indices[1], extracted_indices[2]};
        std::sort(actual_indices.begin(), actual_indices.end());
        REQUIRE(actual_indices == expected_indices);
    }
}

TEST_CASE("Heap edge cases", "[heap]")
{
    SECTION("Zero capacity heap")
    {
        Heap heap(0);
        REQUIRE(heap.empty());
        REQUIRE(heap.size() == 0);
        REQUIRE(heap.capacity() == 0);

        // Should not crash when inserting into zero capacity
        heap.insert(1, 1.0f);
        REQUIRE(heap.empty());
        REQUIRE(heap.size() == 0);
    }

    SECTION("Negative and zero magnitudes")
    {
        Heap heap(5);
        heap.insert(1, -5.0f);
        heap.insert(2, 0.0f);
        heap.insert(3, -1.0f);
        heap.insert(4, 2.0f);

        // Should extract in descending order
        HeapEntry entry1 = heap.extract_max();
        REQUIRE(entry1.magnitude == Catch::Approx(2.0f));

        HeapEntry entry2 = heap.extract_max();
        REQUIRE(entry2.magnitude == Catch::Approx(0.0f));

        HeapEntry entry3 = heap.extract_max();
        REQUIRE(entry3.magnitude == Catch::Approx(-1.0f));

        HeapEntry entry4 = heap.extract_max();
        REQUIRE(entry4.magnitude == Catch::Approx(-5.0f));
    }

    SECTION("Large indices")
    {
        Heap heap(3);
        heap.insert(SIZE_MAX - 1, 3.0f);
        heap.insert(SIZE_MAX, 1.0f);
        heap.insert(0, 2.0f);

        HeapEntry entry1 = heap.extract_max();
        REQUIRE(entry1.bin_index == SIZE_MAX - 1);
        REQUIRE(entry1.magnitude == Catch::Approx(3.0f));

        HeapEntry entry2 = heap.extract_max();
        REQUIRE(entry2.bin_index == 0);
        REQUIRE(entry2.magnitude == Catch::Approx(2.0f));

        HeapEntry entry3 = heap.extract_max();
        REQUIRE(entry3.bin_index == SIZE_MAX);
        REQUIRE(entry3.magnitude == Catch::Approx(1.0f));
    }
}

TEST_CASE("Heap comparison with std::priority_queue", "[heap][comparison]")
{
    SECTION("Identical behavior for random data")
    {
        constexpr size_t TEST_SIZE = 100;
        
        // Custom comparator for std::priority_queue (default is max-heap)
        struct HeapEntryComparator
        {
            bool operator()(const HeapEntry& a, const HeapEntry& b) const
            {
                return a.magnitude < b.magnitude; // For max-heap
            }
        };

        Heap custom_heap(TEST_SIZE);
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> std_heap;

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);

        // Insert same data into both heaps
        for (size_t i = 0; i < TEST_SIZE; ++i)
        {
            float magnitude = dist(rng);
            custom_heap.insert(i, magnitude);
            std_heap.push({i, magnitude});
        }

        REQUIRE(custom_heap.size() == std_heap.size());

        // Extract and compare
        while (!custom_heap.empty() && !std_heap.empty())
        {
            HeapEntry custom_entry = custom_heap.extract_max();
            HeapEntry std_entry = std_heap.top();
            std_heap.pop();

            REQUIRE(custom_entry.magnitude == Catch::Approx(std_entry.magnitude));
        }

        REQUIRE(custom_heap.empty());
        REQUIRE(std_heap.empty());
    }
}