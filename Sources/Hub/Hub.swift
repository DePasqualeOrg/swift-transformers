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
public actor LanguageModelConfigurationFromHub {
    private enum Source {
        case remote(modelName: String, revision: String, hubApi: HubApi)
        case local(modelFolder: URL, hubApi: HubApi)
    }

    private let source: Source
    private let stripVocabForPerformance: Bool
    private var isLoaded = false

    // Cached values (populated once during load)
    private var _modelConfig: Config?
    private var _tokenizerConfig: Config?
    private var _tokenizerData: Config?
    private var _tokenizerVocab: Any?
    private var _tokenizerMerges: [Any]?

    /// Initializes configuration loading from a remote Hub repository.
    ///
    /// - Parameters:
    ///   - modelName: The name/ID of the model repository (e.g., "microsoft/DialoGPT-medium")
    ///   - revision: The git revision to use (defaults to "main")
    ///   - hubApi: The Hub API client to use (defaults to shared instance)
    ///   - stripVocabForPerformance: When `true`, removes `vocab` and `merges` from
    ///     `tokenizerData.model` after extraction, speeding up Config conversion by avoiding
    ///     300k+ nested Config object creations.
    ///     - `true`: ~3x faster loading, but `tokenizerData.model.vocab` will be empty.
    ///       Use `tokenizerVocab` and `tokenizerMerges` properties instead.
    ///     - `false` (default): Full compatibility with existing code that accesses
    ///       `tokenizerData.model.vocab` directly, but slower loading.
    public init(
        modelName: String,
        revision: String = "main",
        hubApi: HubApi = .shared,
        stripVocabForPerformance: Bool = false
    ) {
        self.source = .remote(modelName: modelName, revision: revision, hubApi: hubApi)
        self.stripVocabForPerformance = stripVocabForPerformance
    }

    /// Initializes configuration loading from a local model directory.
    ///
    /// - Parameters:
    ///   - modelFolder: The local directory containing model configuration files
    ///   - hubApi: The Hub API client to use for parsing configurations (defaults to shared instance)
    ///   - stripVocabForPerformance: When `true`, removes `vocab` and `merges` from
    ///     `tokenizerData.model` after extraction, speeding up Config conversion by avoiding
    ///     300k+ nested Config object creations.
    ///     - `true`: ~3x faster loading, but `tokenizerData.model.vocab` will be empty.
    ///       Use `tokenizerVocab` and `tokenizerMerges` properties instead.
    ///     - `false` (default): Full compatibility with existing code that accesses
    ///       `tokenizerData.model.vocab` directly, but slower loading.
    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared,
        stripVocabForPerformance: Bool = false
    ) {
        self.source = .local(modelFolder: modelFolder, hubApi: hubApi)
        self.stripVocabForPerformance = stripVocabForPerformance
    }

    /// Loads configurations on first access and caches the results.
    private func ensureLoaded() async throws {
        guard !isLoaded else { return }

        let configs: LoadedConfigurations
        switch source {
        case .remote(let modelName, let revision, let hubApi):
            configs = try await Self.loadConfigurations(
                modelName: modelName, revision: revision, hubApi: hubApi,
                stripVocabForPerformance: stripVocabForPerformance)
        case .local(let modelFolder, let hubApi):
            configs = try await Self.loadConfigurations(
                modelFolder: modelFolder, hubApi: hubApi,
                stripVocabForPerformance: stripVocabForPerformance)
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
    public var modelConfig: Config? {
        get async throws {
            try await ensureLoaded()
            return _modelConfig
        }
    }

    /// The tokenizer configuration with automatic fallback handling.
    public var tokenizerConfig: Config? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerConfig
        }
    }

    /// The tokenizer data containing vocabulary and merge rules.
    public var tokenizerData: Config {
        get async throws {
            try await ensureLoaded()
            guard let data = _tokenizerData else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }
            return data
        }
    }

    /// Raw vocabulary data extracted directly from JSON for fast tokenizer initialization.
    /// For BPE: `NSDictionary`. For Unigram: `[[Any]]` array of [token, score] pairs.
    public var tokenizerVocab: Any? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerVocab
        }
    }

    /// Raw merges array extracted directly from JSON for fast BPE tokenizer initialization.
    public var tokenizerMerges: [Any]? {
        get async throws {
            try await ensureLoaded()
            return _tokenizerMerges
        }
    }

    /// The model architecture type extracted from the configuration.
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
        var tokenizerVocab: Any?
        var tokenizerMerges: [Any]?
    }

    /// Resolves tokenizerConfig with fallback logic.
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
        hubApi: HubApi = .shared,
        stripVocabForPerformance: Bool
    ) async throws -> LoadedConfigurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, revision: revision, matching: filesToDownload)
            return try await loadConfigurations(modelFolder: downloadedModelFolder, hubApi: hubApi, stripVocabForPerformance: stripVocabForPerformance)
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
        hubApi: HubApi = .shared,
        stripVocabForPerformance: Bool
    ) async throws -> LoadedConfigurations {
        do {
            // Load required configurations
            let modelConfigURL = modelFolder.appending(path: "config.json")

            var modelConfig: Config? = nil
            if FileManager.default.fileExists(atPath: modelConfigURL.path) {
                modelConfig = try hubApi.configuration(fileURL: modelConfigURL)
            }

            let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }

            // Parse tokenizer.json and extract vocab/merges BEFORE Config conversion
            // This avoids the expensive recursive wrapping of 300k+ entries in Config
            let tokenizerJsonData = try Data(contentsOf: tokenizerDataURL)
            // Use yyjson for fast parsing with BOM handling
            let parsedAny = try YYJSONParser.parseToNSDictionary(tokenizerJsonData)
            // Keep as NSMutableDictionary to preserve NSString keys (avoids Unicode normalization)
            let parsed = NSMutableDictionary(dictionary: parsedAny)

            // Extract vocab/merges for fast tokenizer initialization (BPE and Unigram)
            var tokenizerVocab: Any? = nil
            var tokenizerMerges: [Any]? = nil

            if let modelDict = parsed["model"] as? NSDictionary {
                let model = NSMutableDictionary(dictionary: modelDict)
                let modelType = model["type"] as? String

                // Only extract and strip for BPE and Unigram models
                if modelType == "BPE" || modelType == "Unigram" {
                    // Extract vocab preserving NSString keys to avoid Unicode normalization
                    tokenizerVocab = model["vocab"]
                    tokenizerMerges = model["merges"] as? [Any]

                    // Only strip if opted in (for backward compatibility)
                    if stripVocabForPerformance {
                        model.removeObject(forKey: "vocab")
                        model.removeObject(forKey: "merges")
                        parsed["model"] = model
                    }
                }
            }

            // Convert dict to Config (fast if stripped, slower if not)
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

            // Check for chat template and merge if available
            // Prefer .jinja template over .json template
            var chatTemplate: String? = nil
            let chatTemplateJinjaURL = modelFolder.appending(path: "chat_template.jinja")
            let chatTemplateJsonURL = modelFolder.appending(path: "chat_template.json")

            if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path) {
                // Try to load .jinja template as plain text
                chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
            } else if FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
                let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateJsonURL)
            {
                // Fall back to .json template
                chatTemplate = chatTemplateConfig.chatTemplate.string()
            }

            if let chatTemplate {
                // Create or update tokenizer config with chat template
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
