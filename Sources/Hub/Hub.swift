//
//  Hub.swift
//
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

/// A namespace struct providing access to Hugging Face Hub functionality.
///
/// The Hub struct serves as the entry point for interacting with the Hugging Face model repository,
/// providing static methods for downloading models, retrieving file metadata, and managing repository snapshots.
/// All operations are performed through the shared HubApi instance unless specified otherwise.
public enum Hub {}

/// Vocabulary data extracted from tokenizer.json for fast initialization.
///
/// This enum provides type-safe access to vocabulary data, avoiding the need for
/// runtime type casting at each usage site.
///
/// - Note: `@unchecked Sendable` is safe because the underlying data is immutable after extraction from JSON.
public enum TokenizerVocab: @unchecked Sendable {
    /// BPE vocabulary: dictionary mapping token strings to token IDs.
    case bpe(NSDictionary)
    /// Unigram vocabulary: array of [token, score] pairs.
    case unigram(NSArray)
}

/// Merge rules extracted from tokenizer.json for fast BPE initialization.
///
/// - Note: `@unchecked Sendable` is safe because the underlying data is immutable after extraction from JSON.
public struct TokenizerMerges: @unchecked Sendable {
    /// The raw merge rules as extracted from JSON.
    public let rules: [Any]

    public init(_ rules: [Any]) {
        self.rules = rules
    }
}

public extension Hub {
    /// Errors that can occur during Hub client operations.
    ///
    /// This enumeration covers all possible error conditions that may arise when
    /// interacting with the Hugging Face Hub, including network issues, authentication
    /// problems, file system errors, and parsing failures.
    enum HubClientError: LocalizedError {
        /// Authentication is required but no valid token was provided.
        case authorizationRequired
        /// An HTTP error occurred with the specified status code.
        case httpStatusCode(Int)
        /// Failed to parse server response or configuration data.
        case parse
        /// Expected json response could not be parsed as json.
        case jsonSerialization(fileURL: URL, message: String)
        /// An unexpected error occurred during operation.
        case unexpectedError
        /// A download operation failed with the specified error message.
        case downloadError(String)
        /// The requested file was not found on the server or locally.
        case fileNotFound(String)
        /// A network error occurred during communication.
        case networkError(URLError)
        /// The requested resource could not be found.
        case resourceNotFound(String)
        /// A required configuration file is missing.
        case configurationMissing(String)
        /// A file system operation failed.
        case fileSystemError(Error)
        /// Failed to parse data with the specified error message.
        case parseError(String)

        public var errorDescription: String? {
            switch self {
            case .authorizationRequired:
                "Authentication required. Please provide a valid Hugging Face token."
            case let .httpStatusCode(code):
                "HTTP error with status code: \(code)"
            case .parse:
                "Failed to parse server response."
            case .jsonSerialization(_, let message):
                message
            case .unexpectedError:
                "An unexpected error occurred."
            case let .downloadError(message):
                "Download failed: \(message)"
            case let .fileNotFound(filename):
                "File not found: \(filename)"
            case let .networkError(error):
                "Network error: \(error.localizedDescription)"
            case let .resourceNotFound(resource):
                "Resource not found: \(resource)"
            case let .configurationMissing(file):
                "Required configuration file missing: \(file)"
            case let .fileSystemError(error):
                "File system error: \(error.localizedDescription)"
            case let .parseError(message):
                "Parse error: \(message)"
            }
        }
    }

    /// The type of repository on the Hugging Face Hub.
    enum RepoType: String, Codable {
        /// Model repositories containing machine learning models.
        case models
        /// Dataset repositories containing training and evaluation data.
        case datasets
        /// Spaces repositories containing applications and demos.
        case spaces
    }

    /// Represents a repository on the Hugging Face Hub.
    ///
    /// A repository is identified by its unique ID and type, allowing access to
    /// different kinds of resources hosted on the Hub platform.
    struct Repo: Codable {
        /// The unique identifier for the repository (e.g., "microsoft/DialoGPT-medium").
        public let id: String
        /// The type of repository (models, datasets, or spaces).
        public let type: RepoType

        /// Creates a new repository reference.
        ///
        /// - Parameters:
        ///   - id: The unique identifier for the repository
        ///   - type: The type of repository (defaults to .models)
        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

/// Manages language model configuration loading from the Hugging Face Hub.
///
/// This actor handles the asynchronous loading and processing of model configurations,
/// tokenizer configurations, and tokenizer data from either remote Hub repositories
/// or local model directories. It provides fallback mechanisms for missing configurations
/// and manages the complexities of different model types and their specific requirements.
///
/// Vocabulary and merge rules are extracted directly from the raw JSON before Config
/// conversion and made available through `tokenizerVocab` and `tokenizerMerges`.
/// This avoids expensive recursive wrapping of hundreds of thousands of entries
/// into nested Config objects.
public actor LanguageModelConfigurationFromHub {
    private enum Source {
        case remote(modelName: String, revision: String, hubApi: HubApi)
        case local(modelFolder: URL, hubApi: HubApi)
    }

    private let source: Source
    private var isLoaded = false

    // Cached values (populated once during load)
    private var _modelConfig: Config?
    private var _tokenizerConfig: Config?
    private var _tokenizerData: Config?
    private var _tokenizerVocab: TokenizerVocab?
    private var _tokenizerMerges: TokenizerMerges?

    /// Initializes configuration loading from a remote Hub repository.
    ///
    /// - Parameters:
    ///   - modelName: The name/ID of the model repository (e.g., "microsoft/DialoGPT-medium")
    ///   - revision: The git revision to use (defaults to "main")
    ///   - hubApi: The Hub API client to use (defaults to shared instance)
    public init(
        modelName: String,
        revision: String = "main",
        hubApi: HubApi = .shared
    ) {
        self.source = .remote(modelName: modelName, revision: revision, hubApi: hubApi)
    }

    /// Initializes configuration loading from a local model directory.
    ///
    /// - Parameters:
    ///   - modelFolder: The local directory containing model configuration files
    ///   - hubApi: The Hub API client to use for parsing configurations (defaults to shared instance)
    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) {
        self.source = .local(modelFolder: modelFolder, hubApi: hubApi)
    }

    /// Loads configurations on first access and caches the results.
    private func ensureLoaded() async throws {
        guard !isLoaded else { return }

        let configs: LoadedConfigurations
        switch source {
        case .remote(let modelName, let revision, let hubApi):
            configs = try await Self.loadConfigurations(
                modelName: modelName, revision: revision, hubApi: hubApi)
        case .local(let modelFolder, let hubApi):
            configs = try await Self.loadConfigurations(
                modelFolder: modelFolder, hubApi: hubApi)
        }

        _modelConfig = configs.modelConfig
        _tokenizerData = configs.tokenizerData
        _tokenizerVocab = configs.tokenizerVocab
        _tokenizerMerges = configs.tokenizerMerges

        // Resolve tokenizerConfig with fallbacks
        _tokenizerConfig = Self.resolveTokenizerConfig(
            hubConfig: configs.tokenizerConfig,
            modelType: configs.modelConfig?.modelType.string()
        )

        isLoaded = true
    }

    /// The main model configuration containing architecture and parameter settings.
    ///
    /// - Returns: The loaded model configuration
    /// - Throws: Hub errors if configuration loading fails
    public var modelConfig: Config? {
        get async throws {
            try await ensureLoaded()
            return _modelConfig
        }
    }

    /// The tokenizer configuration with automatic fallback handling.
    ///
    /// This property attempts to load the tokenizer configuration from the Hub,
    /// applying fallback configurations when needed and inferring tokenizer classes
    /// based on the model type when not explicitly specified.
    ///
    /// - Returns: The tokenizer configuration, or nil if not available
    /// - Throws: Hub errors if configuration loading fails
    public var tokenizerConfig: Config? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerConfig
        }
    }

    /// The tokenizer data containing vocabulary and merge rules.
    ///
    /// - Returns: The loaded tokenizer data configuration
    /// - Throws: Hub errors if configuration loading fails
    public var tokenizerData: Config {
        get async throws {
            try await ensureLoaded()
            guard let data = _tokenizerData else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }
            return data
        }
    }

    /// Vocabulary data extracted directly from JSON for fast tokenizer initialization.
    ///
    /// Returns a type-safe enum that distinguishes between:
    /// - `.bpe(NSDictionary)`: Token string to ID mapping
    /// - `.unigram(NSArray)`: Array of [token, score] pairs
    public var tokenizerVocab: TokenizerVocab? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerVocab
        }
    }

    /// Merge rules extracted directly from JSON for fast BPE tokenizer initialization.
    public var tokenizerMerges: TokenizerMerges? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerMerges
        }
    }

    /// The model architecture type extracted from the configuration.
    ///
    /// - Returns: The model type string, or nil if not specified
    /// - Throws: Hub errors if configuration loading fails
    public var modelType: String? {
        get async throws {
            try await ensureLoaded()
            return _modelConfig?.modelType.string()
        }
    }

    // MARK: - Private Loading

    /// Raw configurations before fallback resolution.
    private struct LoadedConfigurations {
        var modelConfig: Config?
        var tokenizerConfig: Config?
        var tokenizerData: Config
        var tokenizerVocab: TokenizerVocab?
        var tokenizerMerges: TokenizerMerges?
    }

    /// Resolves tokenizerConfig with fallback logic.
    ///
    /// Applies a series of fallback strategies when the tokenizer configuration
    /// is missing or incomplete:
    /// 1. If tokenizerClass is already present, use the config as-is
    /// 2. Try a bundled fallback config for the model type
    /// 3. Guess the tokenizer class by capitalizing the model type
    /// 4. If no hub config exists at all, use a fallback config if available
    private static func resolveTokenizerConfig(hubConfig: Config?, modelType: String?) -> Config? {
        if let hubConfig {
            // If tokenizerClass is present, use as-is
            if hubConfig.tokenizerClass?.string() != nil {
                return hubConfig
            }

            guard let modelType else { return hubConfig }

            // Try fallback config for this model type
            if let fallbackConfig = fallbackTokenizerConfig(for: modelType) {
                let merged =
                    fallbackConfig.dictionary()?.merging(
                        hubConfig.dictionary(or: [:]),
                        strategy: { current, _ in current }
                    ) ?? [:]
                return Config(merged)
            }

            // Guess tokenizer class by capitalizing model type
            var configuration = hubConfig.dictionary(or: [:])
            configuration["tokenizer_class"] = .init("\(modelType.capitalized)Tokenizer")
            return Config(configuration)
        }

        // No hub config - use fallback if available
        guard let modelType else { return nil }
        return fallbackTokenizerConfig(for: modelType)
    }

    private static func loadConfigurations(
        modelName: String,
        revision: String,
        hubApi: HubApi = .shared
    ) async throws -> LoadedConfigurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, revision: revision, matching: filesToDownload)
            return try await loadConfigurations(modelFolder: downloadedModelFolder, hubApi: hubApi)
        } catch {
            // Convert generic errors to more specific ones
            if let urlError = error as? URLError {
                switch urlError.code {
                case .notConnectedToInternet, .networkConnectionLost:
                    throw Hub.HubClientError.networkError(urlError)
                case .resourceUnavailable:
                    throw Hub.HubClientError.resourceNotFound(modelName)
                default:
                    throw Hub.HubClientError.networkError(urlError)
                }
            } else {
                throw error
            }
        }
    }

    private static func loadConfigurations(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> LoadedConfigurations {
        do {
            // Load model config (optional)
            let modelConfigURL = modelFolder.appending(path: "config.json")

            var modelConfig: Config? = nil
            if FileManager.default.fileExists(atPath: modelConfigURL.path) {
                modelConfig = try hubApi.configuration(fileURL: modelConfigURL)
            }

            let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }

            // Parse tokenizer.json with yyjson and extract vocab/merges before Config conversion.
            // This avoids the expensive recursive wrapping of 300k+ entries in Config objects.
            let tokenizerJsonData = try Data(contentsOf: tokenizerDataURL)
            let parsedAny = try YYJSONParser.parseToNSDictionary(tokenizerJsonData)
            let parsed = NSMutableDictionary(dictionary: parsedAny)

            // Extract vocab and merges for fast tokenizer initialization
            var tokenizerVocab: TokenizerVocab? = nil
            var tokenizerMerges: TokenizerMerges? = nil

            if let modelDict = parsed["model"] as? NSDictionary {
                let model = NSMutableDictionary(dictionary: modelDict)
                let modelType = model["type"] as? String

                if modelType == "BPE", let vocab = model["vocab"] as? NSDictionary {
                    tokenizerVocab = .bpe(vocab)
                    if let merges = model["merges"] as? [Any] {
                        tokenizerMerges = TokenizerMerges(merges)
                    }

                    // Strip vocab and merges from the dictionary before Config conversion
                    model.removeObject(forKey: "vocab")
                    model.removeObject(forKey: "merges")
                    parsed["model"] = model
                } else if modelType == "Unigram", let vocab = model["vocab"] as? NSArray {
                    tokenizerVocab = .unigram(vocab)

                    // Strip vocab from the dictionary before Config conversion
                    model.removeObject(forKey: "vocab")
                    parsed["model"] = model
                }
            }

            // Convert the (now stripped) dictionary to Config
            guard let parsedDict = parsed as? [NSString: Any] else {
                throw Hub.HubClientError.parseError("Expected JSON object at root of tokenizer.json")
            }
            let tokenizerData = Config(parsedDict)

            // Load tokenizer config (optional)
            var tokenizerConfig: Config? = nil
            let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")
            if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
            }

            // Check for chat template and merge if available.
            // Prefer .jinja template over .json template.
            var chatTemplate: String? = nil
            let chatTemplateJinjaURL = modelFolder.appending(path: "chat_template.jinja")
            let chatTemplateJsonURL = modelFolder.appending(path: "chat_template.json")

            if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path) {
                chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
            } else if FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
                let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateJsonURL)
            {
                chatTemplate = chatTemplateConfig.chatTemplate.string()
            }

            if let chatTemplate {
                if var configDict = tokenizerConfig?.dictionary() {
                    configDict["chat_template"] = .init(chatTemplate)
                    tokenizerConfig = Config(configDict)
                } else {
                    tokenizerConfig = Config(["chat_template": chatTemplate])
                }
            }

            return LoadedConfigurations(
                modelConfig: modelConfig,
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                tokenizerVocab: tokenizerVocab,
                tokenizerMerges: tokenizerMerges
            )
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            if let nsError = error as NSError? {
                if nsError.domain == NSCocoaErrorDomain, nsError.code == NSFileReadNoSuchFileError {
                    throw Hub.HubClientError.fileSystemError(error)
                } else if nsError.domain == "NSJSONSerialization" {
                    throw Hub.HubClientError.parseError("Invalid JSON format: \(nsError.localizedDescription)")
                }
            }
            throw Hub.HubClientError.fileSystemError(error)
        }
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        let allowedModelTypeScalars = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        guard !modelType.isEmpty, modelType.unicodeScalars.allSatisfy(allowedModelTypeScalars.contains) else {
            return nil
        }
        let fallbackTokenizerConfigBaseName = "\(modelType)_tokenizer_config"

        // Fallback tokenizer configuration files are located in the `Sources/Hub/Resources` directory
        // On Linux, Bundle.module may not be available if resources aren't properly bundled
        #if canImport(Darwin)
        guard let url = Bundle.module.url(forResource: fallbackTokenizerConfigBaseName, withExtension: "json") else {
            return nil
        }
        #else
        let fileName = "\(fallbackTokenizerConfigBaseName).json"
        // On non-Darwin platforms, also try to locate resources relative to the executable
        var possiblePaths: [URL] = []
        if let executableDirectoryURL = Bundle.main.executableURL?.deletingLastPathComponent() {
            possiblePaths = [
                executableDirectoryURL
                    .appendingPathComponent("swift-transformers_Hub.resources", isDirectory: true)
                    .appendingPathComponent(fileName, isDirectory: false),

                executableDirectoryURL
                    .appendingPathComponent("swift-transformers_Hub.resources", isDirectory: true)
                    .appendingPathComponent("Contents", isDirectory: true)
                    .appendingPathComponent("Resources", isDirectory: true)
                    .appendingPathComponent("Resources", isDirectory: true)
                    .appendingPathComponent(fileName, isDirectory: false)
                    .standardizedFileURL,
            ]
        }
        guard let url = possiblePaths.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            return nil
        }
        #endif

        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [NSString: Any] else {
                throw Hub.HubClientError.parseError("Failed to parse fallback tokenizer config")
            }
            return Config(dictionary)
        } catch let error as Hub.HubClientError {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        } catch {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        }
    }
}
