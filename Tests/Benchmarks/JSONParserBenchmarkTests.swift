//
//  JSONParserBenchmarkTests.swift
//  swift-transformers
//
//  Benchmark tests comparing JSONSerialization vs yyjson performance.
//

import Foundation
import Testing
import Tokenizers
import yyjson

@testable import Hub

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil))
struct JSONParserBenchmarkTests {
    static let modelId = "mlx-community/Qwen3-0.6B-Base-DQ5"

    let modelFolder: URL
    let benchmarkData: Data
    let offlineHubApi: HubApi

    init() async throws {
        // Download model files first (with network)
        let hubApi = HubApi()
        let repo = Hub.Repo(id: Self.modelId)
        let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]
        modelFolder = try await hubApi.snapshot(from: repo, matching: tokenizerFiles)

        let tokenizerURL = modelFolder.appending(path: "tokenizer.json")
        benchmarkData = try Data(contentsOf: tokenizerURL)
        print("Loaded benchmark file: \(ByteCountFormatter.string(fromByteCount: Int64(benchmarkData.count), countStyle: .file))")

        // Create offline HubApi for benchmarking (no network calls)
        offlineHubApi = HubApi(useOfflineMode: true)
    }

    @Test
    func compareParsingSpeed() throws {
        let iterations = 10

        print("Warming up...")
        let _ = try YYJSONParser.parseToConfig(benchmarkData)
        let _ = try JSONSerialization.jsonObject(with: benchmarkData, options: [])

        print("Benchmarking yyjson raw parsing...")
        let yyjsonRawStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            benchmarkData.withUnsafeBytes { buffer in
                let doc = yyjson_read(buffer.baseAddress?.assumingMemoryBound(to: CChar.self), buffer.count, 0)
                yyjson_doc_free(doc)
            }
        }
        let yyjsonRawTime = CFAbsoluteTimeGetCurrent() - yyjsonRawStart

        print("Benchmarking yyjson → Config...")
        let yyjsonDirectStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let _ = try YYJSONParser.parseToConfig(benchmarkData)
        }
        let yyjsonDirectTime = CFAbsoluteTimeGetCurrent() - yyjsonDirectStart

        print("Benchmarking JSONSerialization raw parsing...")
        let jsonSerRawStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let _ = try JSONSerialization.jsonObject(with: benchmarkData, options: [])
        }
        let jsonSerRawTime = CFAbsoluteTimeGetCurrent() - jsonSerRawStart

        print("Benchmarking JSONSerialization → Config...")
        let jsonSerFullStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let parsed = try JSONSerialization.jsonObject(with: benchmarkData, options: [])
            let _ = Config(parsed as! [NSString: Any])
        }
        let jsonSerFullTime = CFAbsoluteTimeGetCurrent() - jsonSerFullStart

        let rawSpeedup = jsonSerRawTime / yyjsonRawTime
        let directSpeedup = jsonSerFullTime / yyjsonDirectTime

        // Calculate average time per operation (in milliseconds)
        let yyjsonRawAvg = (yyjsonRawTime / Double(iterations)) * 1000
        let yyjsonDirectAvg = (yyjsonDirectTime / Double(iterations)) * 1000
        let jsonSerRawAvg = (jsonSerRawTime / Double(iterations)) * 1000
        let jsonSerFullAvg = (jsonSerFullTime / Double(iterations)) * 1000

        // Time saved per operation
        let rawTimeSaved = jsonSerRawAvg - yyjsonRawAvg
        let directTimeSaved = jsonSerFullAvg - yyjsonDirectAvg

        print(
            """

            ============================================
            JSON Parsing Benchmark Results (\(iterations) iterations)
            File size: \(ByteCountFormatter.string(fromByteCount: Int64(benchmarkData.count), countStyle: .file))
            ============================================
            yyjson (raw parse):       \(String(format: "%.3f", yyjsonRawTime))s (\(String(format: "%.1f", yyjsonRawAvg)) ms avg)
            yyjson → Config:          \(String(format: "%.3f", yyjsonDirectTime))s (\(String(format: "%.1f", yyjsonDirectAvg)) ms avg)
            JSONSerialization (raw):  \(String(format: "%.3f", jsonSerRawTime))s (\(String(format: "%.1f", jsonSerRawAvg)) ms avg)
            JSONSerialization+Config: \(String(format: "%.3f", jsonSerFullTime))s (\(String(format: "%.1f", jsonSerFullAvg)) ms avg)
            --------------------------------------------
            Raw parse speedup:        \(String(format: "%.2f", rawSpeedup))x (\(String(format: "%.0f", rawTimeSaved)) ms saved)
            Full path speedup:        \(String(format: "%.2f", directSpeedup))x (\(String(format: "%.0f", directTimeSaved)) ms saved)
            ============================================

            """)
    }

    @Test
    func parsingResultsMatch() throws {
        let yyjsonResult = try YYJSONParser.parseToConfig(benchmarkData)
        let jsonSerParsed = try JSONSerialization.jsonObject(with: benchmarkData, options: []) as! [NSString: Any]
        let jsonSerResult = Config(jsonSerParsed)

        // Compare top-level keys
        let yyjsonKeys = Set(yyjsonResult.dictionary()?.keys.map { $0.string } ?? [])
        let jsonSerKeys = Set(jsonSerResult.dictionary()?.keys.map { $0.string } ?? [])

        #expect(yyjsonKeys == jsonSerKeys, "Top-level keys should match")

        // Compare vocab size if present
        if let yyjsonVocab = yyjsonResult.model.vocab.dictionary(),
            let jsonSerVocab = jsonSerResult.model.vocab.dictionary()
        {
            #expect(yyjsonVocab.count == jsonSerVocab.count, "Vocab sizes should match")
            print("Vocab size: \(yyjsonVocab.count) tokens")
        }
    }

    @Test
    func compareTokenizerLoadingSpeed() async throws {
        let iterations = 5

        print("Warming up...")
        let _ = try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: offlineHubApi)

        print("Benchmarking tokenizer loading with yyjson...")
        let yyjsonStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: offlineHubApi)
        }
        let yyjsonTime = CFAbsoluteTimeGetCurrent() - yyjsonStart

        print("Benchmarking tokenizer loading with JSONSerialization...")
        let jsonSerStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let _ = try await loadTokenizerWithJSONSerialization()
        }
        let jsonSerTime = CFAbsoluteTimeGetCurrent() - jsonSerStart

        let speedup = jsonSerTime / yyjsonTime

        // Calculate average time per load (in milliseconds)
        let yyjsonAvg = (yyjsonTime / Double(iterations)) * 1000
        let jsonSerAvg = (jsonSerTime / Double(iterations)) * 1000
        let timeSaved = jsonSerAvg - yyjsonAvg

        print(
            """

            ============================================
            Tokenizer Loading Benchmark (\(iterations) iterations)
            Model: \(Self.modelId)
            ============================================
            yyjson (current):     \(String(format: "%.3f", yyjsonTime))s (\(String(format: "%.0f", yyjsonAvg)) ms avg)
            JSONSerialization:    \(String(format: "%.3f", jsonSerTime))s (\(String(format: "%.0f", jsonSerAvg)) ms avg)
            --------------------------------------------
            Speedup: \(String(format: "%.2f", speedup))x faster with yyjson (\(String(format: "%.0f", timeSaved)) ms saved)
            ============================================

            """)
    }

    /// Loads a tokenizer using JSONSerialization instead of yyjson for comparison.
    private func loadTokenizerWithJSONSerialization() async throws -> Tokenizer {
        let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        // Load tokenizer data with JSONSerialization
        let tokenizerDataRaw = try Data(contentsOf: tokenizerDataURL)
        let tokenizerDataParsed = try JSONSerialization.jsonObject(with: tokenizerDataRaw, options: []) as! [NSString: Any]
        let tokenizerData = Config(tokenizerDataParsed)

        // Load tokenizer config with JSONSerialization
        let tokenizerConfigRaw = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfigParsed = try JSONSerialization.jsonObject(with: tokenizerConfigRaw, options: []) as! [NSString: Any]
        let tokenizerConfig = Config(tokenizerConfigParsed)

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}
