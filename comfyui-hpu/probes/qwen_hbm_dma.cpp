#include <synapse_api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void check(synStatus status, const char* operation) {
    if (status != synSuccess) {
        std::cerr << operation << " failed with status " << status << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string directory = argv[1];
    const uint64_t size = std::stoull(argv[2]) * 1024 * 1024;
    const int repeats = 20;
    check(synInitialize(), "initialize");
    synDeviceId device;
    check(synDeviceAcquireByDeviceType(&device, synDeviceGaudi2), "acquire");
    synStreamHandle stream;
    check(synStreamCreateGeneric(&stream, device, 0), "stream");
    uint64_t source, destination;
    check(synDeviceMalloc(device, size, 0, 0, &source), "source allocation");
    check(synDeviceMalloc(device, size, 0, 0, &destination), "destination allocation");
    check(synMemsetD8Async(source, 63, size, stream), "source fill");
    check(synStreamSynchronize(stream), "fill synchronize");
    auto copy = [&]() {
        for (int i = 0; i < repeats; ++i) {
            check(synMemCopyAsync(stream, source, size, destination, DRAM_TO_DRAM), "copy");
        }
        check(synStreamSynchronize(stream), "copy synchronize");
    };
    copy();
    std::vector<double> samples;
    for (int i = 0; i < 5; ++i) {
        const auto start = std::chrono::steady_clock::now();
        copy();
        samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    uint32_t required = 0;
    uint64_t trace_address = 0;
    check(synProfilerQueryRequiredMemory(device, &required), "profile size");
    if (required) {
        check(synDeviceMalloc(device, required, 0, 0, &trace_address), "trace allocation");
        check(synProfilerSetUserBuffer(device, reinterpret_cast<void*>(trace_address)), "profile buffer");
    }
    check(synProfilerStart(synTraceDevice, device), "profile start");
    const auto start = std::chrono::steady_clock::now();
    copy();
    const double profiled = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    check(synProfilerStop(synTraceDevice, device), "profile stop");
    size_t trace_size = 0, entries = 0;
    check(synProfilerGetTrace(synTraceAll, device, synTraceFormatTEF, nullptr, &trace_size, &entries), "trace query");
    std::vector<char> buffer(trace_size);
    check(synProfilerGetTrace(synTraceAll, device, synTraceFormatTEF, buffer.data(), &trace_size, &entries), "trace read");
    const auto* events = reinterpret_cast<synTraceEvent*>(buffer.data());
    std::ofstream output(directory + "/native-events.jsonl");
    output << std::setprecision(18);
    for (size_t i = 0; i < entries; ++i) {
        const auto& event = events[i];
        if (!event.category || std::string(event.category) != "BMON BW") continue;
        output << "{\"name\":\"" << event.name << "\",\"category\":\"BMON BW\",\"ts\":" << event.timestamp
               << ",\"counter_values\":[" << event.arguments.minimum << ',' << event.arguments.maximum << ','
               << event.arguments.average << "]}\n";
    }
    std::ofstream run(directory + "/run.json");
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    run << std::setprecision(12) << "{\"operation\":\"synapse_dma\",\"mib\":" << size / 1024 / 1024
        << ",\"iterations\":" << repeats << ",\"known_read_bytes\":" << size * repeats
        << ",\"known_write_bytes\":" << size * repeats << ",\"profiled_seconds\":" << profiled
        << ",\"median_seconds\":" << ordered[2] << ",\"samples_seconds\":[";
    for (size_t i = 0; i < samples.size(); ++i) run << (i ? "," : "") << samples[i];
    run << "]}\n";
    check(synProfilerGetTrace(synTraceAll, device, synTraceFormatTEF, nullptr, nullptr, nullptr), "trace export");
    check(synStreamDestroy(stream), "destroy stream");
    if (trace_address) check(synDeviceFree(device, trace_address, 0), "free trace");
    check(synDeviceFree(device, source, 0), "free source");
    check(synDeviceFree(device, destination, 0), "free destination");
    check(synDeviceRelease(device), "release");
    check(synDestroy(), "destroy");
}
