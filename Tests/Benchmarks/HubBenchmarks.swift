//
//  HubBenchmarks.swift
//
//  Benchmark tests to measure Hub API performance improvements.
//  Run with: RUN_BENCHMARKS=1 swift test --filter HubBenchmarks
//

import Foundation
import Testing

@testable import Hub

@Suite(
    "Hub Benchmarks",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1")
)
struct HubBenchmarks {
    let downloadDestination: URL

    init() {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        downloadDestination = base.appending(component: "huggingface-benchmarks")
    }

    private func cleanDownloadDestination() {
        try? FileManager.default.removeItem(at: downloadDestination)
    }

    // MARK: - Offline mode cache hit (skip re-hash)

    /// Measures time to load cached files in offline mode.
    /// Tests whether hash verification is skipped for already-verified files.
    @Test("Benchmark offline mode with cached files")
    func benchmarkOfflineModeWithCachedFiles() async throws {
        cleanDownloadDestination()
        defer { cleanDownloadDestination() }

        // Qwen tokenizer.json is ~7MB, large enough to show hash computation time
        let repo = Hub.Repo(id: "mlx-community/Qwen3-0.6B-Base-DQ5")
        let hubApi = HubApi(downloadBase: downloadDestination)

        // First, download the tokenizer file (online)
        _ = try await hubApi.snapshot(from: repo, matching: "tokenizer.json")

        // Now measure offline mode access (the optimization target)
        let offlineApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        let iterations = 5
        var times: [Double] = []

        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await offlineApi.snapshot(from: repo, matching: "tokenizer.json")
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("  Iteration \(i): \(String(format: "%.2f", elapsed)) ms")
        }

        let average = times.reduce(0, +) / Double(times.count)
        print("BENCHMARK [Offline mode cache hit]: Average = \(String(format: "%.2f", average)) ms over \(iterations) iterations")
    }

    // MARK: - Cached file retrieval (skip HEAD requests)

    /// Measures time to retrieve already-cached files (6 JSON files from t5-base).
    /// Tests whether HEAD requests are skipped when commit hash matches.
    @Test("Benchmark cached file retrieval")
    func benchmarkCachedFileRetrieval() async throws {
        cleanDownloadDestination()
        defer { cleanDownloadDestination() }

        let repo = Hub.Repo(id: "t5-base")
        let hubApi = HubApi(downloadBase: downloadDestination)

        // First, download the files (online) - this also caches the commit hash
        _ = try await hubApi.snapshot(from: repo, matching: "*.json")

        // Now measure subsequent access (the optimization target)
        let iterations = 3
        var times: [Double] = []

        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await hubApi.snapshot(from: repo, matching: "*.json")
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("  Iteration \(i): \(String(format: "%.2f", elapsed)) ms")
        }

        let average = times.reduce(0, +) / Double(times.count)
        print("BENCHMARK [Cached file retrieval]: Average = \(String(format: "%.2f", average)) ms over \(iterations) iterations")
    }

    /// Measures time to download multiple files.
    /// Tests parallel vs sequential download performance.
    @Test("Benchmark parallel downloads")
    func benchmarkParallelDownloads() async throws {
        cleanDownloadDestination()
        defer { cleanDownloadDestination() }

        let repo = Hub.Repo(id: "t5-base")
        let hubApi = HubApi(downloadBase: downloadDestination)

        let start = CFAbsoluteTimeGetCurrent()
        _ = try await hubApi.snapshot(from: repo, matching: "*.json")
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print("BENCHMARK [Parallel downloads]: \(String(format: "%.2f", elapsed)) ms")
    }
}
