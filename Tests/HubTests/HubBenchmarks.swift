//
//  HubBenchmarks.swift
//
//  Benchmark tests to measure performance improvements.
//  Run these before and after optimization commits to compare.
//

import XCTest

@testable import Hub

class HubBenchmarks: XCTestCase {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-benchmarks")
    }()

    override func setUp() {
        // Clean before each test to ensure consistent starting state
        try? FileManager.default.removeItem(at: downloadDestination)
    }

    override func tearDown() {
        // Clean after each test
        try? FileManager.default.removeItem(at: downloadDestination)
    }

    // MARK: - Offline mode cache hit (skip re-hash)

    /// Measures time to load cached files in offline mode.
    /// Tests whether hash verification is skipped for already-verified files.
    func testBenchmarkOfflineModeWithCachedFiles() async throws {
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
        print("⏱️  BENCHMARK [Offline mode cache hit]: Average = \(String(format: "%.2f", average)) ms over \(iterations) iterations")
    }

    // MARK: - Cached file retrieval (skip HEAD requests)

    /// Measures time to retrieve already-cached files (6 JSON files from t5-base).
    /// Tests whether HEAD requests are skipped when commit hash matches.
    func testBenchmarkCachedFileRetrieval() async throws {
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
        print("⏱️  BENCHMARK [Cached file retrieval]: Average = \(String(format: "%.2f", average)) ms over \(iterations) iterations")
    }

    /// Measures time to download multiple files.
    /// Tests parallel vs sequential download performance.
    func testBenchmarkParallelDownloads() async throws {
        let repo = Hub.Repo(id: "t5-base")
        let hubApi = HubApi(downloadBase: downloadDestination)

        // setUp already cleared cache, so this is a fresh download
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await hubApi.snapshot(from: repo, matching: "*.json")
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        print("⏱️  BENCHMARK [Parallel downloads]: \(String(format: "%.2f", elapsed)) ms")
    }
}
