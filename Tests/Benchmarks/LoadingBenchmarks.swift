//
//  LoadingBenchmarks.swift
//
//  Benchmarks for tokenizer loading performance.
//  Run with: RUN_BENCHMARKS=1 swift test --filter LoadingBenchmarks
//

import Dispatch
import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite(
    "Tokenizer Loading Benchmarks",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil)
)
struct LoadingBenchmarks {

    // MARK: - Benchmark Utilities

    struct BenchmarkStats {
        let mean: Double
        let stdDev: Double
        let min: Double
        let max: Double

        var formatted: String {
            String(format: "%.1f ms (± %.1f)", mean, stdDev)
        }
    }

    /// Measures execution time using monotonic clock, returning individual timings in milliseconds.
    private func measureAsync(
        label: String,
        labelWidth: Int,
        iterations: Int,
        warmup: Int = 2,
        _ block: () async throws -> Void
    ) async rethrows -> [Double] {
        let paddedLabel = label.padding(toLength: labelWidth, withPad: " ", startingAt: 0)
        print("\(paddedLabel) ", terminator: "")
        fflush(stdout)

        // Warmup runs (not measured)
        for _ in 0..<warmup {
            try await block()
        }

        var times: [Double] = []
        times.reserveCapacity(iterations)

        for i in 0..<iterations {
            let start = DispatchTime.now()
            try await block()
            let end = DispatchTime.now()
            let nanoseconds = end.uptimeNanoseconds - start.uptimeNanoseconds
            times.append(Double(nanoseconds) / 1_000_000)

            if (i + 1) % 5 == 0 {
                print(String(format: "%2d", i + 1), terminator: "")
            } else {
                print(".", terminator: "")
            }
            fflush(stdout)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: " %6.1f ms", mean))

        return times
    }

    /// Sync version of measure for non-async operations.
    private func measure(
        label: String,
        labelWidth: Int,
        iterations: Int,
        warmup: Int = 2,
        _ block: () throws -> Void
    ) rethrows -> [Double] {
        let paddedLabel = label.padding(toLength: labelWidth, withPad: " ", startingAt: 0)
        print("\(paddedLabel) ", terminator: "")
        fflush(stdout)

        // Warmup runs (not measured)
        for _ in 0..<warmup {
            try block()
        }

        var times: [Double] = []
        times.reserveCapacity(iterations)

        for i in 0..<iterations {
            let start = DispatchTime.now()
            try block()
            let end = DispatchTime.now()
            let nanoseconds = end.uptimeNanoseconds - start.uptimeNanoseconds
            times.append(Double(nanoseconds) / 1_000_000)

            if (i + 1) % 5 == 0 {
                print(String(format: "%2d", i + 1), terminator: "")
            } else {
                print(".", terminator: "")
            }
            fflush(stdout)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: " %6.1f ms", mean))

        return times
    }

    private func stats(_ times: [Double]) -> BenchmarkStats {
        let mean = times.reduce(0, +) / Double(times.count)
        let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
        let stdDev = sqrt(variance)
        let min = times.min() ?? 0
        let max = times.max() ?? 0
        return BenchmarkStats(mean: mean, stdDev: stdDev, min: min, max: max)
    }

    // MARK: - Tests

    @Test("Benchmark tokenizer loading from local folder")
    func benchmarkLocalLoading() async throws {
        let modelName = "Qwen/Qwen3-0.6B-Base"
        let iterations = 15
        let labelWidth = 20

        // Download model files
        print("Downloading model: \(modelName)...")
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        // Get tokenizer.json size for context
        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerJsonData = try Data(contentsOf: tokenizerJsonURL)

        // Count vocab/merges entries
        var vocabCount = 0
        var mergeCount = 0
        let parsed = try JSONSerialization.jsonObject(with: tokenizerJsonData) as! [String: Any]
        if let model = parsed["model"] as? [String: Any] {
            vocabCount = (model["vocab"] as? [String: Any])?.count ?? 0
            mergeCount = (model["merges"] as? [Any])?.count ?? 0
        }

        print("Benchmarking with \(iterations) iterations...\n")

        let times = try await measureAsync(label: "AutoTokenizer.from", labelWidth: labelWidth, iterations: iterations) {
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder)
        }

        let s = stats(times)

        print(
            """

            ============================================
            Tokenizer Loading Benchmark (\(iterations) iterations)
            Model: \(modelName)
            File size: \(ByteCountFormatter.string(fromByteCount: Int64(tokenizerJsonData.count), countStyle: .file))
            Vocab: \(vocabCount) tokens, Merges: \(mergeCount)
            ============================================
            Loading time: \(s.formatted)
            Min: \(String(format: "%.1f", s.min)) ms, Max: \(String(format: "%.1f", s.max)) ms
            ============================================

            """)
    }

    @Test("Compare optimized vs unoptimized loading")
    func compareOptimizedVsUnoptimized() async throws {
        let modelName = "Qwen/Qwen3-0.6B-Base"
        let iterations = 10
        let labelWidth = 22

        // Download model files
        print("Downloading model: \(modelName)...")
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        print("Benchmarking with \(iterations) iterations...\n")

        // --- UNOPTIMIZED PATH (old way) ---
        // Recreates the old behavior: convert entire JSON to Config including vocab/merges,
        // then let the tokenizer parse vocab/merges from Config (the slow path).
        let unoptimizedTimes = try await measureAsync(label: "Unoptimized (old way)", labelWidth: labelWidth, iterations: iterations) {
            // Read files
            let tokenizerData = try Data(contentsOf: tokenizerJsonURL)
            let configData = try Data(contentsOf: tokenizerConfigURL)

            // Parse JSON (old way - uses JSONSerialization, wraps entire dict in Config)
            let parsedTokenizer = try JSONSerialization.jsonObject(with: tokenizerData) as! [NSString: Any]
            let parsedConfig = try JSONSerialization.jsonObject(with: configData) as! [NSString: Any]

            // Convert ENTIRE dict to Config including vocab/merges (the slow part)
            let tokenizerDataConfig = Config(parsedTokenizer)
            let tokenizerConfig = Config(parsedConfig)

            // Create tokenizer without pre-extracted vocab/merges (falls back to Config parsing)
            let _ = try await AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerDataConfig,
                tokenizerVocab: nil,
                tokenizerMerges: nil
            )
        }

        // --- OPTIMIZED PATH (new way) ---
        let optimizedTimes = try await measureAsync(label: "Optimized (current)", labelWidth: labelWidth, iterations: iterations) {
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder)
        }

        let unoptimizedStats = stats(unoptimizedTimes)
        let optimizedStats = stats(optimizedTimes)
        let speedup = unoptimizedStats.mean / optimizedStats.mean
        let timeSaved = unoptimizedStats.mean - optimizedStats.mean

        print(
            """

            ============================================
            Optimized vs Unoptimized (\(iterations) iterations)
            Model: \(modelName)
            ============================================
            Unoptimized (old way): \(unoptimizedStats.formatted)
              - Converts entire tokenizer.json to Config
              - 300k+ vocab/merges entries wrapped in Config
            Optimized (current):   \(optimizedStats.formatted)
              - Extracts vocab/merges before Config conversion
              - Only small config sections wrapped in Config
            --------------------------------------------
            Speedup: \(String(format: "%.2f", speedup))x (\(String(format: "%.0f", timeSaved)) ms saved)
            ============================================

            """)
    }

    @Test("Detailed breakdown of optimized loading path")
    func detailedOptimizedBreakdown() async throws {
        let modelName = "Qwen/Qwen3-0.6B-Base"
        let iterations = 10

        // Download model files
        print("Downloading model: \(modelName)...")
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        print("Running detailed breakdown (\(iterations) iterations)...\n")

        // Warmup
        for _ in 0..<2 {
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder)
        }

        // Collect times for each stage across iterations
        var stage1Times: [Double] = []
        var stage2Times: [Double] = []
        var stage3Times: [Double] = []
        var stage4Times: [Double] = []
        var stage5Times: [Double] = []
        var stage6aTimes: [Double] = []
        var stage6bTimes: [Double] = []

        for i in 0..<iterations {
            // --- Stage 1: Read files from disk ---
            let stage1Start = DispatchTime.now()
            let tokenizerJsonData = try Data(contentsOf: tokenizerJsonURL)
            let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
            stage1Times.append(Double(DispatchTime.now().uptimeNanoseconds - stage1Start.uptimeNanoseconds) / 1_000_000)

            // --- Stage 2: Parse JSON ---
            let stage2Start = DispatchTime.now()
            let parsedAny = try YYJSONParser.parseToNSDictionary(tokenizerJsonData)
            let parsed = NSMutableDictionary(dictionary: parsedAny)
            let parsedConfig = try JSONSerialization.jsonObject(with: tokenizerConfigData) as! [NSString: Any]
            stage2Times.append(Double(DispatchTime.now().uptimeNanoseconds - stage2Start.uptimeNanoseconds) / 1_000_000)

            // --- Stage 3: Extract vocab/merges ---
            let stage3Start = DispatchTime.now()
            var tokenizerVocab: NSDictionary?
            var tokenizerMerges: [Any]?
            if let modelDict = parsed["model"] as? NSDictionary {
                let model = NSMutableDictionary(dictionary: modelDict)
                tokenizerVocab = model["vocab"] as? NSDictionary
                tokenizerMerges = model["merges"] as? [Any]
                model.removeObject(forKey: "vocab")
                model.removeObject(forKey: "merges")
                parsed["model"] = model
            }
            stage3Times.append(Double(DispatchTime.now().uptimeNanoseconds - stage3Start.uptimeNanoseconds) / 1_000_000)

            // --- Stage 4: Config conversion (small sections only) ---
            let stage4Start = DispatchTime.now()
            let tokenizerData = Config(parsed as! [NSString: Any])
            _ = Config(parsedConfig)
            stage4Times.append(Double(DispatchTime.now().uptimeNanoseconds - stage4Start.uptimeNanoseconds) / 1_000_000)

            // --- Stage 5: PreTrainedTokenizer init (excluding model init) ---
            let stage5Start = DispatchTime.now()
            var addedTokens: [String: Int] = [:]
            for addedToken in tokenizerData["addedTokens"].array(or: []) {
                guard let id = addedToken["id"].integer() else { continue }
                guard let content = addedToken.content.string() else { continue }
                addedTokens[content] = id
            }
            let _ = PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
            let _ = NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
            let _ = PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
            let _ = DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: Set(addedTokens.keys))
            let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
                guard let content = addedToken.content.string() else { return nil }
                let prefix = addedToken["lstrip"].boolean(or: false)
                let suffix = addedToken["rstrip"].boolean(or: false)
                return (content: content, prefix: prefix, suffix: suffix)
            }.sorted { $0.content.count > $1.content.count }
            let addedTokensRegexString = unwrappedAddedTokens.map {
                let token = NSRegularExpression.escapedPattern(for: $0.content)
                let prefix = $0.prefix ? #"\s*"# : ""
                let suffix = $0.suffix ? #"\s*"# : ""
                return "\(prefix)(\(token))\(suffix)"
            }.joined(separator: "|")
            let _ = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])
            stage5Times.append(Double(DispatchTime.now().uptimeNanoseconds - stage5Start.uptimeNanoseconds) / 1_000_000)

            // --- Stage 6: BPETokenizer model init ---
            let vocabForAsync = tokenizerVocab!
            let mergesForAsync = tokenizerMerges ?? []
            let addedTokensForAsync = addedTokens

            // 6a: Phase 1 - Build tokensToIds and parse merges in parallel
            let stage6aStart = DispatchTime.now()
            async let tokensToIdsTask = BPETokenizer.buildTokensToIds(rawVocab: vocabForAsync, addedTokens: addedTokensForAsync)
            async let mergesTask = BPETokenizer.mergesFromRawJSON(mergesForAsync)
            let tokensToIds = await tokensToIdsTask
            let merges = await mergesTask
            stage6aTimes.append(Double(DispatchTime.now().uptimeNanoseconds - stage6aStart.uptimeNanoseconds) / 1_000_000)

            // 6b: Phase 2 - Build remaining dictionaries in parallel
            let stage6bStart = DispatchTime.now()
            async let stringToIdTask = BPETokenizer.buildStringToIdIfNeeded(from: tokensToIds)
            async let bpeRanksTask = BPETokenizer.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
            async let idsToTokensTask = BPETokenizer.buildIdsToTokens(from: tokensToIds)
            _ = await (stringToIdTask, bpeRanksTask, idsToTokensTask)
            stage6bTimes.append(Double(DispatchTime.now().uptimeNanoseconds - stage6bStart.uptimeNanoseconds) / 1_000_000)

            // Progress indicator
            if (i + 1) % 5 == 0 {
                print(String(format: "%2d", i + 1), terminator: "")
            } else {
                print(".", terminator: "")
            }
            fflush(stdout)
        }
        print(" done\n")

        // Calculate stats for each stage
        let s1 = stats(stage1Times)
        let s2 = stats(stage2Times)
        let s3 = stats(stage3Times)
        let s4 = stats(stage4Times)
        let s5 = stats(stage5Times)
        let s6a = stats(stage6aTimes)
        let s6b = stats(stage6bTimes)

        let totalMean = s1.mean + s2.mean + s3.mean + s4.mean + s5.mean + s6a.mean + s6b.mean

        func pct(_ v: Double) -> String { String(format: "%4.1f", v / totalMean * 100) }

        print(
            """
            ============================================
            Detailed Breakdown (\(iterations) iterations)
            Model: \(modelName)
            ============================================
            Stage 1 - Read files:           \(String(format: "%5.1f", s1.mean)) ms (± \(String(format: "%.1f", s1.stdDev)))  \(pct(s1.mean))%
            Stage 2 - Parse JSON:           \(String(format: "%5.1f", s2.mean)) ms (± \(String(format: "%.1f", s2.stdDev)))  \(pct(s2.mean))%
            Stage 3 - Extract vocab/merges: \(String(format: "%5.1f", s3.mean)) ms (± \(String(format: "%.1f", s3.stdDev)))  \(pct(s3.mean))%
            Stage 4 - Config conversion:    \(String(format: "%5.1f", s4.mean)) ms (± \(String(format: "%.1f", s4.stdDev)))  \(pct(s4.mean))%
            Stage 5 - Tokenizer setup:      \(String(format: "%5.1f", s5.mean)) ms (± \(String(format: "%.1f", s5.stdDev)))  \(pct(s5.mean))%
            Stage 6a - tokensToIds+merges:  \(String(format: "%5.1f", s6a.mean)) ms (± \(String(format: "%.1f", s6a.stdDev)))  \(pct(s6a.mean))%
            Stage 6b - bpeRanks+idsToTokens:\(String(format: "%5.1f", s6b.mean)) ms (± \(String(format: "%.1f", s6b.stdDev)))  \(pct(s6b.mean))%
            --------------------------------------------
            TOTAL:                          \(String(format: "%5.1f", totalMean)) ms
            ============================================

            """)

        // Identify bottlenecks
        let stages: [(String, Double)] = [
            ("Read files", s1.mean),
            ("Parse JSON", s2.mean),
            ("Extract vocab/merges", s3.mean),
            ("Config conversion", s4.mean),
            ("Tokenizer setup", s5.mean),
            ("tokensToIds + merges", s6a.mean),
            ("bpeRanks + idsToTokens", s6b.mean),
        ]
        let sorted = stages.sorted { $0.1 > $1.1 }

        print("Top bottlenecks:")
        for (i, (name, time)) in sorted.prefix(3).enumerated() {
            print("  \(i + 1). \(name): \(String(format: "%.1f", time)) ms (\(pct(time))%)")
        }
        print("")
    }

    @Test("Sequential vs parallel dictionary building")
    func sequentialVsParallelDictBuilding() async throws {
        let modelName = "Qwen/Qwen3-0.6B-Base"
        let iterations = 10
        let labelWidth = 12

        // Download model files
        print("Downloading model: \(modelName)...")
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")

        // Parse JSON and extract vocab/merges
        let tokenizerJsonData = try Data(contentsOf: tokenizerJsonURL)
        let parsedAny = try YYJSONParser.parseToNSDictionary(tokenizerJsonData)
        guard let modelDict = parsedAny["model"] as? NSDictionary,
            let rawVocab = modelDict["vocab"] as? NSDictionary,
            let rawMerges = modelDict["merges"] as? [Any]
        else {
            print("Failed to parse tokenizer.json")
            return
        }

        let addedTokens: [String: Int] = [:]

        print("Benchmarking with \(iterations) iterations...\n")

        // --- SEQUENTIAL (both phases) ---
        let sequentialTimes = measure(label: "Sequential", labelWidth: labelWidth, iterations: iterations) {
            // Phase 1 sequential
            let tokensToIds = BPETokenizer.buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
            let merges = BPETokenizer.mergesFromRawJSON(rawMerges)
            // Phase 2 sequential
            let _ = BPETokenizer.buildStringToIdIfNeeded(from: tokensToIds)
            let _ = BPETokenizer.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
            let _ = BPETokenizer.buildIdsToTokens(from: tokensToIds)
        }

        // --- PARALLEL (both phases) ---
        let parallelTimes = await measureAsync(label: "Parallel", labelWidth: labelWidth, iterations: iterations) {
            // Phase 1 parallel
            async let tokensToIdsTask = BPETokenizer.buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
            async let mergesTask = BPETokenizer.mergesFromRawJSON(rawMerges)
            let tokensToIds = await tokensToIdsTask
            let merges = await mergesTask
            // Phase 2 parallel
            async let stringToId = BPETokenizer.buildStringToIdIfNeeded(from: tokensToIds)
            async let bpeRanks = BPETokenizer.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
            async let idsToTokens = BPETokenizer.buildIdsToTokens(from: tokensToIds)
            _ = await (stringToId, bpeRanks, idsToTokens)
        }

        let seqStats = stats(sequentialTimes)
        let parStats = stats(parallelTimes)
        let speedup = seqStats.mean / parStats.mean
        let timeSaved = seqStats.mean - parStats.mean

        print(
            """

            ============================================
            Sequential vs Parallel (\(iterations) iterations)
            Vocab: \(rawVocab.count) tokens, Merges: \(rawMerges.count)
            ============================================
            Sequential: \(seqStats.formatted)
            Parallel:   \(parStats.formatted)
            --------------------------------------------
            Speedup: \(String(format: "%.2f", speedup))x (\(String(format: "%.0f", timeSaved)) ms saved)
            ============================================

            """)
    }
}
