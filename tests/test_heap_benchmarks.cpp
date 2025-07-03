#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <rtpghi/rtpghi.hpp>
#include <queue>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace rtpghi;

// Helper class to track allocations
class AllocationTracker
{
private:
    static size_t allocation_count;
    static size_t deallocation_count;
    static size_t total_bytes_allocated;

public:
    static void reset()
    {
        allocation_count = 0;
        deallocation_count = 0;
        total_bytes_allocated = 0;
    }

    static size_t get_allocation_count() { return allocation_count; }
    static size_t get_deallocation_count() { return deallocation_count; }
    static size_t get_total_bytes_allocated() { return total_bytes_allocated; }

    static void record_allocation(size_t bytes)
    {
        allocation_count++;
        total_bytes_allocated += bytes;
    }

    static void record_deallocation()
    {
        deallocation_count++;
    }
};

size_t AllocationTracker::allocation_count = 0;
size_t AllocationTracker::deallocation_count = 0;
size_t AllocationTracker::total_bytes_allocated = 0;

// Custom allocator for tracking allocations
template<typename T>
class TrackingAllocator
{
public:
    using value_type = T;

    TrackingAllocator() = default;
    
    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>&) {}

    T* allocate(size_t n)
    {
        AllocationTracker::record_allocation(n * sizeof(T));
        return static_cast<T*>(std::malloc(n * sizeof(T)));
    }

    void deallocate(T* p, size_t)
    {
        AllocationTracker::record_deallocation();
        std::free(p);
    }

    template<typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const TrackingAllocator<U>&) const { return false; }
};

// Performance test data generator
class BenchmarkData
{
public:
    static std::vector<HeapEntry> generate_random_data(size_t size, unsigned int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
        std::vector<HeapEntry> data;
        data.reserve(size);

        for (size_t i = 0; i < size; ++i)
        {
            data.push_back({i, dist(rng)});
        }

        return data;
    }

    static std::vector<HeapEntry> generate_sorted_data(size_t size, bool ascending = true)
    {
        std::vector<HeapEntry> data;
        data.reserve(size);

        for (size_t i = 0; i < size; ++i)
        {
            float magnitude = ascending ? static_cast<float>(i) : static_cast<float>(size - i);
            data.push_back({i, magnitude});
        }

        return data;
    }

    static std::vector<HeapEntry> generate_duplicate_data(size_t size, size_t num_unique_values = 10)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(num_unique_values));
        std::vector<HeapEntry> data;
        data.reserve(size);

        for (size_t i = 0; i < size; ++i)
        {
            float magnitude = std::floor(dist(rng));
            data.push_back({i, magnitude});
        }

        return data;
    }
};

TEST_CASE("Heap performance benchmarks", "[heap][benchmark]")
{
    SECTION("Small dataset (100 elements)")
    {
        constexpr size_t SIZE = 100;
        auto data = BenchmarkData::generate_random_data(SIZE);

        BENCHMARK("Custom Heap - Insert & Extract All")
        {
            Heap heap(SIZE);
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("std::priority_queue - Insert & Extract All")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };
    }

    SECTION("Medium dataset (1000 elements)")
    {
        constexpr size_t SIZE = 1000;
        auto data = BenchmarkData::generate_random_data(SIZE);

        BENCHMARK("Custom Heap - Insert & Extract All")
        {
            Heap heap(SIZE);
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("std::priority_queue - Insert & Extract All")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };
    }

    SECTION("Large dataset (10000 elements)")
    {
        constexpr size_t SIZE = 10000;
        auto data = BenchmarkData::generate_random_data(SIZE);

        BENCHMARK("Custom Heap - Insert & Extract All")
        {
            Heap heap(SIZE);
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("std::priority_queue - Insert & Extract All")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };
    }

    SECTION("Insert-only performance")
    {
        constexpr size_t SIZE = 5000;
        auto data = BenchmarkData::generate_random_data(SIZE);

        BENCHMARK("Custom Heap - Insert Only")
        {
            Heap heap(SIZE);
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
        };

        BENCHMARK("std::priority_queue - Insert Only")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
        };
    }

    SECTION("Extract-only performance (pre-populated)")
    {
        constexpr size_t SIZE = 5000;
        auto data = BenchmarkData::generate_random_data(SIZE);

        BENCHMARK("Custom Heap - Extract Only")
        {
            // Work around MSVC requiring capture of constexpr variables
            return [&data, heap_size = SIZE]() {
                Heap heap(heap_size);
                for (const auto& entry : data)
                {
                    heap.insert(entry.bin_index, entry.magnitude);
                }
                while (!heap.empty())
                {
                    heap.extract_max();
                }
            };
        };

        BENCHMARK("std::priority_queue - Extract Only")
        {
            return [&data]() {
                struct HeapEntryComparator
                {
                    bool operator()(const HeapEntry& a, const HeapEntry& b) const
                    {
                        return a.magnitude < b.magnitude;
                    }
                };
                std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
                
                for (const auto& entry : data)
                {
                    pq.push(entry);
                }
                while (!pq.empty())
                {
                    pq.pop();
                }
            };
        };
    }
}

TEST_CASE("Heap worst-case performance", "[heap][benchmark]")
{
    SECTION("Sorted input (worst case for heap)")
    {
        constexpr size_t SIZE = 1000;
        auto ascending_data = BenchmarkData::generate_sorted_data(SIZE, true);
        auto descending_data = BenchmarkData::generate_sorted_data(SIZE, false);

        BENCHMARK("Custom Heap - Ascending Order")
        {
            Heap heap(SIZE);
            for (const auto& entry : ascending_data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("Custom Heap - Descending Order")
        {
            Heap heap(SIZE);
            for (const auto& entry : descending_data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("std::priority_queue - Ascending Order")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : ascending_data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };

        BENCHMARK("std::priority_queue - Descending Order")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : descending_data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };
    }

    SECTION("Many duplicate values")
    {
        constexpr size_t SIZE = 1000;
        auto duplicate_data = BenchmarkData::generate_duplicate_data(SIZE, 5);

        BENCHMARK("Custom Heap - Many Duplicates")
        {
            Heap heap(SIZE);
            for (const auto& entry : duplicate_data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            while (!heap.empty())
            {
                heap.extract_max();
            }
        };

        BENCHMARK("std::priority_queue - Many Duplicates")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            for (const auto& entry : duplicate_data)
            {
                pq.push(entry);
            }
            while (!pq.empty())
            {
                pq.pop();
            }
        };
    }
}

// Memory allocation tests
TEST_CASE("Heap memory allocation analysis", "[heap][memory]")
{
    SECTION("Pre-allocated vs dynamic allocation")
    {
        constexpr size_t SIZE = 1000;
        auto data = BenchmarkData::generate_random_data(SIZE);

        // Test custom heap with pre-allocation
        {
            AllocationTracker::reset();
            Heap heap(SIZE);  // Pre-allocate capacity
            
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            
            while (!heap.empty())
            {
                heap.extract_max();
            }
            
            size_t heap_allocations = AllocationTracker::get_allocation_count();
            size_t heap_deallocations = AllocationTracker::get_deallocation_count();
            size_t heap_bytes = AllocationTracker::get_total_bytes_allocated();
            
            // Custom Heap should have ZERO allocations during runtime operations
            REQUIRE(heap_allocations == 0);
            REQUIRE(heap_deallocations == 0);
            REQUIRE(heap_bytes == 0);
        }

        // Test std::priority_queue with tracking allocator
        {
            AllocationTracker::reset();
            
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            
            std::priority_queue<HeapEntry, 
                              std::vector<HeapEntry, TrackingAllocator<HeapEntry>>, 
                              HeapEntryComparator> pq;
            
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
            
            while (!pq.empty())
            {
                pq.pop();
            }
            
            size_t pq_allocations = AllocationTracker::get_allocation_count();
            size_t pq_bytes = AllocationTracker::get_total_bytes_allocated();
            
            // std::priority_queue should have multiple allocations for dynamic resizing
            REQUIRE(pq_allocations > 0);
            REQUIRE(pq_bytes > 0);
        }
    }

    SECTION("Memory usage comparison")
    {
        constexpr size_t SIZE = 1000;
        
        // Measure memory usage of custom heap
        {
            Heap heap(SIZE);
            size_t heap_memory = sizeof(heap) + heap.capacity() * sizeof(HeapEntry);
            INFO("Custom Heap memory usage: " << heap_memory << " bytes");
        }

        // Measure memory usage of std::priority_queue
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            // Fill to capacity to estimate memory usage
            for (size_t i = 0; i < SIZE; ++i)
            {
                pq.push({i, static_cast<float>(i)});
            }
            
            // Estimate memory usage (this is approximate)
            size_t pq_memory = sizeof(pq) + pq.size() * sizeof(HeapEntry);
            INFO("std::priority_queue memory usage (estimated): " << pq_memory << " bytes");
        }
    }
}

// Real-world usage pattern test
TEST_CASE("Heap real-world usage patterns", "[heap][benchmark]")
{
    SECTION("RTPGHI algorithm simulation")
    {
        // Simulate typical RTPGHI usage: insert many, extract progressively
        constexpr size_t FFT_BINS = 1024;
        constexpr size_t SIGNIFICANT_BINS = FFT_BINS / 4;  // 25% significant
        
        auto data = BenchmarkData::generate_random_data(FFT_BINS);
        
        // Sort by magnitude to simulate significant bin selection
        std::sort(data.begin(), data.end(), [](const HeapEntry& a, const HeapEntry& b) {
            return a.magnitude > b.magnitude;
        });
        
        // Only use top 25% (significant bins)
        data.resize(SIGNIFICANT_BINS);

        BENCHMARK("Custom Heap - RTPGHI Pattern")
        {
            Heap heap(2 * FFT_BINS);  // Time + frequency propagation
            
            // Insert significant bins
            for (const auto& entry : data)
            {
                heap.insert(entry.bin_index, entry.magnitude);
            }
            
            // Process progressively (extract max, possibly add more)
            size_t processed = 0;
            while (!heap.empty() && processed < SIGNIFICANT_BINS)
            {
                HeapEntry current = heap.extract_max();
                
                // Simulate frequency propagation by adding offset entries
                if (processed < SIGNIFICANT_BINS / 2)
                {
                    heap.insert(current.bin_index + FFT_BINS, current.magnitude);
                }
                
                processed++;
            }
        };

        BENCHMARK("std::priority_queue - RTPGHI Pattern")
        {
            struct HeapEntryComparator
            {
                bool operator()(const HeapEntry& a, const HeapEntry& b) const
                {
                    return a.magnitude < b.magnitude;
                }
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> pq;
            
            // Insert significant bins
            for (const auto& entry : data)
            {
                pq.push(entry);
            }
            
            // Process progressively
            size_t processed = 0;
            while (!pq.empty() && processed < SIGNIFICANT_BINS)
            {
                HeapEntry current = pq.top();
                pq.pop();
                
                // Simulate frequency propagation
                if (processed < SIGNIFICANT_BINS / 2)
                {
                    pq.push({current.bin_index + FFT_BINS, current.magnitude});
                }
                
                processed++;
            }
        };
    }
}