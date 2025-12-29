//
//  LoadingInstrumentationTests.swift
//
//  Instrumentation tests for measuring tokenizer loading performance.
//  Run with: swift test --filter LoadingInstrumentationTests
//

import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

/// Timing utility for instrumentation
private func measure<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
    let start = CFAbsoluteTimeGetCurrent()
    let result = try block()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
    print("[\(label)] \(String(format: "%.2f", elapsed))ms")
    return result
}

private func measure<T>(_ label: String, _ block: () async throws -> T) async rethrows -> T {
    let start = CFAbsoluteTimeGetCurrent()
    let result = try await block()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
    print("[\(label)] \(String(format: "%.2f", elapsed))ms")
    return result
}

private func makeHubApi() -> (api: HubApi, downloadDestination: URL) {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    let destination = base.appending(component: "huggingface-tests-instrumentation")
    return (HubApi(downloadBase: destination), destination)
}

@Suite("Loading Instrumentation")
struct LoadingInstrumentationTests {

    /// Instruments the full loading pipeline, breaking down each phase
    @Test
    func instrumentFullLoadingPipeline() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let modelName = "coreml-projects/Llama-2-7b-chat-coreml"
        print("\n=== Instrumenting tokenizer loading for: \(modelName) ===\n")

        // Phase 1: Download files from Hub
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        let modelFolder = try await measure("1. Download snapshot") {
            try await hubApi.snapshot(from: repo, matching: filesToDownload)
        }

        // Phase 2: Read and parse JSON files
        let tokenizerJsonURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        let tokenizerJsonData = try measure("2a. Read tokenizer.json") {
            try Data(contentsOf: tokenizerJsonURL)
        }
        print("    File size: \(ByteCountFormatter.string(fromByteCount: Int64(tokenizerJsonData.count), countStyle: .file))")

        let tokenizerConfigData = try measure("2b. Read tokenizer_config.json") {
            try Data(contentsOf: tokenizerConfigURL)
        }
        print("    File size: \(ByteCountFormatter.string(fromByteCount: Int64(tokenizerConfigData.count), countStyle: .file))")

        // Phase 3: Parse JSON
        let tokenizerJsonParsed = try measure("3a. Parse tokenizer.json") {
            try JSONSerialization.jsonObject(with: tokenizerJsonData, options: [])
        }

        let tokenizerConfigParsed = try measure("3b. Parse tokenizer_config.json") {
            try JSONSerialization.jsonObject(with: tokenizerConfigData, options: [])
        }

        // Phase 4: Create Config objects
        let tokenizerData = measure("4a. Create Config from tokenizer.json") {
            Config(tokenizerJsonParsed as! [NSString: Any])
        }

        let tokenizerConfig = measure("4b. Create Config from tokenizer_config.json") {
            Config(tokenizerConfigParsed as! [NSString: Any])
        }

        // Phase 5: Create tokenizer (this includes all component initialization)
        let tokenizer = try measure("5. Create PreTrainedTokenizer (total)") {
            try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        }

        // Verify it works
        let inputIds = tokenizer("Hello, world!")
        print("\n=== Verification ===")
        print("Input: \"Hello, world!\"")
        print("Token IDs: \(inputIds)")

        #expect(inputIds.count > 0)
    }

    /// Instruments component creation by accessing factories directly
    @Test
    func instrumentComponentCreation() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let modelName = "coreml-projects/Llama-2-7b-chat-coreml"
        print("\n=== Instrumenting component creation for: \(modelName) ===\n")

        // Load configs first (not timed - covered by other test)
        let config = LanguageModelConfigurationFromHub(modelName: modelName, hubApi: hubApi)
        let tokenizerConfig = try await config.tokenizerConfig!
        let tokenizerData = try await config.tokenizerData

        // Instrument added tokens parsing
        var addedTokens: [String: Int] = [:]
        measure("1. Parse added tokens") {
            for addedToken in tokenizerData["addedTokens"].array(or: []) {
                guard let id = addedToken["id"].integer() else { continue }
                guard let content = addedToken.content.string() else { continue }
                addedTokens[content] = id
            }
        }
        print("   Found \(addedTokens.count) added tokens")

        // Instrument regex compilation
        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = tokenizerData["addedTokens"].array(or: []).compactMap { addedToken in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
            return (content, prefix, suffix)
        }.sorted { $0.content.count > $1.content.count }

        _ = measure("2. Compile added tokens regex") {
            let pattern = unwrappedAddedTokens.map {
                let token = NSRegularExpression.escapedPattern(for: $0.content)
                let prefix = $0.prefix ? #"\s*"# : ""
                let suffix = $0.suffix ? #"\s*"# : ""
                return "\(prefix)(\(token))\(suffix)"
            }.joined(separator: "|")
            return try? NSRegularExpression(pattern: pattern, options: [])
        }

        // Instrument factory creation
        _ = measure("3. Create PreTokenizer") {
            PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        }

        _ = measure("4. Create Normalizer") {
            NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        }

        _ = measure("5. Create PostProcessor") {
            PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        }

        _ = measure("6. Create Decoder") {
            DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: Set(addedTokens.keys))
        }

        // Instrument model creation
        _ = try measure("7. Create TokenizerModel") {
            try TokenizerModel.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                addedTokens: addedTokens
            )
        }

        print("\n=== Component creation complete ===")
    }

    /// Compare loading times across different tokenizer types
    @Test
    func compareTokenizerLoadingTimes() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let models = [
            "coreml-projects/Llama-2-7b-chat-coreml",  // BPE
            "openai/whisper-large-v2",                  // BPE (Whisper)
            "bert-base-uncased",                        // WordPiece
        ]

        print("\n=== Comparing tokenizer loading times ===\n")

        for modelName in models {
            print("--- \(modelName) ---")

            // Pre-download to isolate tokenizer creation time
            let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
            let repo = Hub.Repo(id: modelName)
            let modelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

            // Time the loading from local folder (no network)
            _ = try await measure("Load from disk") {
                try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: hubApi)
            }

            print("")
        }
    }
}
