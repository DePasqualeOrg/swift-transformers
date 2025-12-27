//
//  LoadingBenchmarks.swift
//
//  Benchmarks for tokenizer loading performance.
//  Run with: swift test --filter LoadingBenchmarks
//

import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Tokenizer Loading Benchmarks", .serialized)
struct LoadingBenchmarks {

    @Test("Benchmark tokenizer loading from local folder")
    func benchmarkLocalLoading() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("TOKENIZER LOADING BENCHMARK")
        print(String(repeating: "=", count: 60))
        print("\nThis benchmark measures end-to-end tokenizer loading time")
        print("from a local model folder (simulating cached models).\n")

        let modelName = "Qwen/Qwen3-0.6B-Base"

        // Download model files
        print("Downloading model: \(modelName)")
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)
        print("Model folder: \(modelFolder.path)\n")

        // Get tokenizer.json size for context
        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerJsonSize = try Data(contentsOf: tokenizerJsonURL).count
        print("tokenizer.json size: \(tokenizerJsonSize / 1024) KB")

        // Count vocab/merges entries
        let parsed =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: tokenizerJsonURL)
            ) as! [String: Any]
        if let model = parsed["model"] as? [String: Any] {
            if let vocab = model["vocab"] as? [String: Any] {
                print("Vocab entries: \(vocab.count)")
            }
            if let merges = model["merges"] as? [Any] {
                print("Merge entries: \(merges.count)")
            }
        }
        print("")

        // Warm-up run
        print("Warm-up run...")
        let _ = try await AutoTokenizer.from(modelFolder: modelFolder)

        // Timed runs
        let iterations = 5
        print("Running \(iterations) iterations...\n")

        var times: [Double] = []
        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("  Run \(i): \(String(format: "%.1f", elapsed)) ms")
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let minTime = times.min()!
        let maxTime = times.max()!

        print("\n" + String(repeating: "-", count: 40))
        print("RESULTS")
        print(String(repeating: "-", count: 40))
        print("  Average: \(String(format: "%.1f", avgTime)) ms")
        print("  Min:     \(String(format: "%.1f", minTime)) ms")
        print("  Max:     \(String(format: "%.1f", maxTime)) ms")
        print(String(repeating: "=", count: 60))
        print("")
    }

    @Test("Compare optimized vs unoptimized loading")
    func compareOptimizedVsUnoptimized() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("OPTIMIZED vs UNOPTIMIZED COMPARISON")
        print(String(repeating: "=", count: 60))
        print("\nCompares the optimized path (vocab/merges extracted before")
        print("Config conversion) vs the old unoptimized path.\n")

        let modelName = "Qwen/Qwen3-0.6B-Base"

        // Download model files
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        let iterations = 3

        // --- UNOPTIMIZED PATH (old way) ---
        // Recreates the old behavior: convert entire JSON to Config including vocab/merges
        print("UNOPTIMIZED PATH (old way):")
        print("  - Converts entire tokenizer.json to Config")
        print("  - 300k+ vocab/merges entries wrapped in Config objects")
        print("")

        var unoptimizedTimes: [Double] = []
        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()

            // Read files
            let tokenizerData = try Data(contentsOf: tokenizerJsonURL)
            let configData = try Data(contentsOf: tokenizerConfigURL)

            // Parse JSON (old way - no BOM preservation, but close enough for timing)
            let parsedTokenizer = try JSONSerialization.jsonObject(with: tokenizerData) as! [NSString: Any]
            let parsedConfig = try JSONSerialization.jsonObject(with: configData) as! [NSString: Any]

            // Convert ENTIRE dict to Config (THIS IS THE SLOW PART)
            let tokenizerDataConfig = Config(parsedTokenizer)
            let tokenizerConfig = Config(parsedConfig)

            // Create tokenizer (will re-extract vocab from Config - wasteful)
            let _ = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerDataConfig)

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            unoptimizedTimes.append(elapsed)
            print("  Run \(i): \(String(format: "%.1f", elapsed)) ms")
        }

        let unoptimizedAvg = unoptimizedTimes.reduce(0, +) / Double(unoptimizedTimes.count)

        // --- OPTIMIZED PATH (new way) ---
        print("\nOPTIMIZED PATH (new way):")
        print("  - Extracts vocab/merges before Config conversion")
        print("  - Only small config sections wrapped in Config")
        print("")

        var optimizedTimes: [Double] = []
        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            optimizedTimes.append(elapsed)
            print("  Run \(i): \(String(format: "%.1f", elapsed)) ms")
        }

        let optimizedAvg = optimizedTimes.reduce(0, +) / Double(optimizedTimes.count)
        let speedup = unoptimizedAvg / optimizedAvg

        print("\n" + String(repeating: "-", count: 50))
        print("COMPARISON RESULTS")
        print(String(repeating: "-", count: 50))
        print("  Unoptimized avg: \(String(format: "%7.1f", unoptimizedAvg)) ms")
        print("  Optimized avg:   \(String(format: "%7.1f", optimizedAvg)) ms")
        print("  Speedup:         \(String(format: "%7.2f", speedup))x")
        print(String(repeating: "=", count: 60))
        print("")
    }

    @Test("Detailed breakdown of optimized loading path")
    func detailedOptimizedBreakdown() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("DETAILED BREAKDOWN - OPTIMIZED PATH")
        print(String(repeating: "=", count: 60))
        print("\nShows where time is spent in the current optimized loading.\n")

        let modelName = "Qwen/Qwen3-0.6B-Base"

        // Download model files
        let hubApi = HubApi()
        let repo = Hub.Repo(id: modelName)
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        // Warm up
        let _ = try await AutoTokenizer.from(modelFolder: modelFolder)

        // --- Stage 1: Read files from disk ---
        let stage1Start = CFAbsoluteTimeGetCurrent()
        let tokenizerJsonData = try Data(contentsOf: tokenizerJsonURL)
        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let stage1Time = (CFAbsoluteTimeGetCurrent() - stage1Start) * 1000

        // --- Stage 2: Parse JSON ---
        let stage2Start = CFAbsoluteTimeGetCurrent()
        let parsedAny = try YYJSONParser.parseToNSDictionary(tokenizerJsonData)
        let parsed = NSMutableDictionary(dictionary: parsedAny)
        let parsedConfig = try JSONSerialization.jsonObject(with: tokenizerConfigData) as! [NSString: Any]
        let stage2Time = (CFAbsoluteTimeGetCurrent() - stage2Start) * 1000

        // --- Stage 3: Extract vocab/merges ---
        let stage3Start = CFAbsoluteTimeGetCurrent()
        var tokenizerVocab: NSDictionary? = nil
        var tokenizerMerges: [Any]? = nil

        if let modelDict = parsed["model"] as? NSDictionary {
            let model = NSMutableDictionary(dictionary: modelDict)
            tokenizerVocab = model["vocab"] as? NSDictionary
            tokenizerMerges = model["merges"] as? [Any]
            model.removeObject(forKey: "vocab")
            model.removeObject(forKey: "merges")
            parsed["model"] = model
        }
        let stage3Time = (CFAbsoluteTimeGetCurrent() - stage3Start) * 1000

        // --- Stage 4: Config conversion (small sections only) ---
        let stage4Start = CFAbsoluteTimeGetCurrent()
        let tokenizerData = Config(parsed as! [NSString: Any])
        _ = Config(parsedConfig) // tokenizerConfig - unused but timed
        let stage4Time = (CFAbsoluteTimeGetCurrent() - stage4Start) * 1000

        // --- Stage 5: PreTrainedTokenizer init (excluding model init) ---
        // This includes: addedTokens parsing, regex building, pre/post processor creation
        let stage5Start = CFAbsoluteTimeGetCurrent()

        var addedTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue }
            guard let content = addedToken.content.string() else { continue }
            addedTokens[content] = id
        }

        // Pre-tokenizer, normalizer, post-processor, decoder creation
        let _ = PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        let _ = NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        let _ = PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        let _ = DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: Set(addedTokens.keys))

        // Build addedTokens regex (sorted by length)
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

        let stage5Time = (CFAbsoluteTimeGetCurrent() - stage5Start) * 1000

        // --- Stage 6: BPETokenizer model init ---
        // Capture values to avoid Swift 6 warnings about captured vars in concurrent code
        let vocabForAsync = tokenizerVocab!
        let mergesForAsync = tokenizerMerges ?? []
        let addedTokensForAsync = addedTokens

        // 6a: Phase 1 - Build tokensToIds and parse merges IN PARALLEL
        let stage6aStart = CFAbsoluteTimeGetCurrent()
        async let tokensToIdsTask = BPETokenizer.buildTokensToIds(rawVocab: vocabForAsync, addedTokens: addedTokensForAsync)
        async let mergesTask = BPETokenizer.mergesFromRawJSON(mergesForAsync)
        let tokensToIds = await tokensToIdsTask
        let merges = await mergesTask
        let stage6aTime = (CFAbsoluteTimeGetCurrent() - stage6aStart) * 1000

        // 6b: Phase 2 - Build remaining dictionaries IN PARALLEL
        let stage6bStart = CFAbsoluteTimeGetCurrent()
        async let stringToIdTask = BPETokenizer.buildStringToIdIfNeeded(from: tokensToIds)
        async let bpeRanksTask = BPETokenizer.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
        async let idsToTokensTask = BPETokenizer.buildIdsToTokens(from: tokensToIds)
        let stringToId = await stringToIdTask
        let bpeRanks = await bpeRanksTask
        let idsToTokens = await idsToTokensTask
        let stage6bTime = (CFAbsoluteTimeGetCurrent() - stage6bStart) * 1000

        // Verify dictionaries were built (silence unused variable warnings)
        _ = stringToId?.count
        _ = bpeRanks.count
        _ = idsToTokens.count

        let stage6Time = stage6aTime + stage6bTime

        let totalTime = stage1Time + stage2Time + stage3Time + stage4Time + stage5Time + stage6Time

        // Print results
        print("Stage 1 - Read files:           \(String(format: "%6.1f", stage1Time)) ms  (\(String(format: "%4.1f", stage1Time / totalTime * 100))%)")
        print("Stage 2 - Parse JSON:           \(String(format: "%6.1f", stage2Time)) ms  (\(String(format: "%4.1f", stage2Time / totalTime * 100))%)")
        print("Stage 3 - Extract vocab/merges: \(String(format: "%6.1f", stage3Time)) ms  (\(String(format: "%4.1f", stage3Time / totalTime * 100))%)")
        print("Stage 4 - Config conversion:    \(String(format: "%6.1f", stage4Time)) ms  (\(String(format: "%4.1f", stage4Time / totalTime * 100))%)")
        print("Stage 5 - Tokenizer setup:      \(String(format: "%6.1f", stage5Time)) ms  (\(String(format: "%4.1f", stage5Time / totalTime * 100))%)")
        print("Stage 6 - BPE model init:       \(String(format: "%6.1f", stage6Time)) ms  (\(String(format: "%4.1f", stage6Time / totalTime * 100))%)")
        print("  6a - Phase 1 (tokensToIds + merges):")
        print("        Wall time:              \(String(format: "%6.1f", stage6aTime)) ms  (\(String(format: "%4.1f", stage6aTime / totalTime * 100))%)")
        print("  6b - Phase 2 (stringToId + bpeRanks + idsToTokens):")
        print("        Wall time:              \(String(format: "%6.1f", stage6bTime)) ms  (\(String(format: "%4.1f", stage6bTime / totalTime * 100))%)")
        print(String(repeating: "-", count: 50))
        print("TOTAL:                          \(String(format: "%6.1f", totalTime)) ms")
        print("")

        // Identify bottlenecks
        let stages = [
            ("Read files", stage1Time),
            ("Parse JSON", stage2Time),
            ("Extract vocab/merges", stage3Time),
            ("Config conversion", stage4Time),
            ("Tokenizer setup", stage5Time),
            ("Phase 1 (tokensToIds + merges)", stage6aTime),
            ("Phase 2 (bpeRanks + idsToTokens)", stage6bTime),
        ]
        let sorted = stages.sorted { $0.1 > $1.1 }

        print("TOP BOTTLENECKS:")
        for (i, (name, time)) in sorted.prefix(3).enumerated() {
            let pct = time / totalTime * 100
            print("  \(i + 1). \(name): \(String(format: "%.1f", time)) ms (\(String(format: "%.1f", pct))%)")
        }

        print("")
        print("Current Swift time: \(String(format: "%.0f", totalTime)) ms")
        print(String(repeating: "=", count: 60))
        print("")
    }

    @Test("Sequential vs parallel dictionary building")
    func sequentialVsParallelDictBuilding() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("SEQUENTIAL vs PARALLEL DICTIONARY BUILDING")
        print(String(repeating: "=", count: 60))
        print("\nCompares sequential vs async let parallel building.")
        print("Phase 1: tokensToIds + merges (independent)")
        print("Phase 2: stringToId + bpeRanks + idsToTokens (depend on Phase 1)\n")

        let modelName = "Qwen/Qwen3-0.6B-Base"

        // Download model files
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

        print("Vocab size: \(rawVocab.count)")
        print("Merges count: \(rawMerges.count)\n")

        let iterations = 5

        // --- SEQUENTIAL (both phases) ---
        print("SEQUENTIAL (all steps one after another):")
        var sequentialTimes: [Double] = []
        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
            // Phase 1 sequential
            let tokensToIds = BPETokenizer.buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
            let merges = BPETokenizer.mergesFromRawJSON(rawMerges)
            // Phase 2 sequential
            let _ = BPETokenizer.buildStringToIdIfNeeded(from: tokensToIds)
            let _ = BPETokenizer.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
            let _ = BPETokenizer.buildIdsToTokens(from: tokensToIds)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            sequentialTimes.append(elapsed)
            print("  Run \(i): \(String(format: "%.1f", elapsed)) ms")
        }
        let seqAvg = sequentialTimes.reduce(0, +) / Double(sequentialTimes.count)

        // --- PARALLEL (both phases) ---
        print("\nPARALLEL (async let for both phases):")
        var parallelTimes: [Double] = []
        for i in 1...iterations {
            let start = CFAbsoluteTimeGetCurrent()
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
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            parallelTimes.append(elapsed)
            print("  Run \(i): \(String(format: "%.1f", elapsed)) ms")
        }
        let parAvg = parallelTimes.reduce(0, +) / Double(parallelTimes.count)

        let speedup = seqAvg / parAvg
        let savings = seqAvg - parAvg

        print("\n" + String(repeating: "-", count: 50))
        print("RESULTS")
        print(String(repeating: "-", count: 50))
        print("  Sequential avg:  \(String(format: "%6.1f", seqAvg)) ms")
        print("  Parallel avg:    \(String(format: "%6.1f", parAvg)) ms")
        print("  Speedup:         \(String(format: "%6.2f", speedup))x")
        print("  Time saved:      \(String(format: "%6.1f", savings)) ms")
        print(String(repeating: "=", count: 60))
        print("")
    }
}
