//
//  HubTests.swift
//
//  Created by Pedro Cuenca on 18/05/2023.
//

import XCTest

@testable import Hub

class HubTests: XCTestCase {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    override func setUp() {}

    override func tearDown() {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    func testConfigDownload() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            guard let config = try await configLoader.modelConfig else {
                XCTFail("Test repo is expected to have a config.json file")
                return
            }

            // Test leaf value (Int)
            guard let eos = config["eos_token_id"].integer() else {
                XCTFail("nil leaf value (Int)")
                return
            }
            XCTAssertEqual(eos, 1)

            // Test leaf value (String)
            guard let modelType = config["model_type"].string() else {
                XCTFail("nil leaf value (String)")
                return
            }
            XCTAssertEqual(modelType, "t5")

            // Test leaf value (Array)
            guard let architectures: [String] = config["architectures"].get() else {
                XCTFail("nil array")
                return
            }
            XCTAssertEqual(architectures, ["T5ForConditionalGeneration"])

            // Test nested wrapper
            guard !config["task_specific_params"].isNull() else {
                XCTFail("nil nested wrapper")
                return
            }

            guard let summarizationMaxLength = config["task_specific_params"]["summarization"]["max_length"].integer() else {
                XCTFail("cannot traverse nested containers")
                return
            }
            XCTAssertEqual(summarizationMaxLength, 200)
        } catch {
            XCTFail("Cannot download test configuration from the Hub: \(error)")
        }
    }

    func testConfigCamelCase() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            guard let config = try await configLoader.modelConfig else {
                XCTFail("Test repo is expected to have a config.json file")
                return
            }

            // Test leaf value (Int)
            guard let eos = config["eosTokenId"].integer() else {
                XCTFail("nil leaf value (Int)")
                return
            }
            XCTAssertEqual(eos, 1)

            // Test leaf value (String)
            guard let modelType = config["modelType"].string() else {
                XCTFail("nil leaf value (String)")
                return
            }
            XCTAssertEqual(modelType, "t5")

            guard let summarizationMaxLength = config["taskSpecificParams"]["summarization"]["maxLength"].integer() else {
                XCTFail("cannot traverse nested containers")
                return
            }
            XCTAssertEqual(summarizationMaxLength, 200)
        } catch {
            XCTFail("Cannot download test configuration from the Hub: \(error)")
        }
    }

    // MARK: - Repo Info Tests

    /// Tests that getRepoInfo returns both filenames and a valid commit hash
    func testGetRepoInfoReturnsFilenamesAndSha() async throws {
        let repo = Hub.Repo(id: "t5-base")
        let (files, sha) = try await hubApi.getRepoInfo(from: repo, matching: ["*.json"])

        // Should return filenames
        let filenames = files.map { $0.filename }
        XCTAssertFalse(filenames.isEmpty, "Should return at least one JSON file")
        XCTAssertTrue(filenames.contains("config.json"), "Should contain config.json")

        // Should return a valid 40-character commit hash
        XCTAssertEqual(sha.count, 40, "SHA should be 40 characters")
        XCTAssertTrue(sha.allSatisfy { $0.isHexDigit }, "SHA should be hexadecimal")
    }

    // MARK: - Snapshot Download Tests

    /// Tests basic file download and verifies file content (like huggingface_hub's test_download_model)
    func testSnapshotDownloadsFilesWithCorrectContent() async throws {
        let repo = Hub.Repo(id: "t5-base")

        let destination = try await hubApi.snapshot(from: repo, matching: ["config.json"])

        // Verify file exists
        let configPath = destination.appending(path: "config.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path), "config.json should exist")

        // Verify file content is valid JSON with expected fields
        let data = try Data(contentsOf: configPath)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json, "config.json should be valid JSON")
        XCTAssertEqual(json?["model_type"] as? String, "t5", "Should be t5 model")
    }

    /// Tests downloading multiple files with glob pattern (like huggingface_hub's test_download_model_with_allow_pattern)
    func testSnapshotWithGlobPattern() async throws {
        let repo = Hub.Repo(id: "t5-base")

        let destination = try await hubApi.snapshot(from: repo, matching: ["*.json"])

        // Verify multiple JSON files were downloaded
        let files = try FileManager.default.contentsOfDirectory(atPath: destination.path)
        let jsonFiles = files.filter { $0.hasSuffix(".json") }
        XCTAssertGreaterThan(jsonFiles.count, 1, "Should download multiple JSON files")

        // Verify no non-JSON files were downloaded (glob filtering works)
        let nonJsonFiles = files.filter { !$0.hasSuffix(".json") && !$0.hasPrefix(".") }
        XCTAssertEqual(nonJsonFiles.count, 0, "Should only download JSON files")
    }

    /// Tests that cached files are returned without re-downloading
    /// This validates the knownCommitHash optimization that skips HEAD requests
    func testSnapshotReturnsCachedFiles() async throws {
        let repo = Hub.Repo(id: "t5-base")

        // First download
        let destination1 = try await hubApi.snapshot(from: repo, matching: ["config.json"])

        // Get file modification date
        let configPath = destination1.appending(path: "config.json")
        let attrs1 = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let modDate1 = attrs1[.modificationDate] as? Date

        // Small delay
        try await Task.sleep(for: .milliseconds(100))

        // Second download (should use cache)
        let destination2 = try await hubApi.snapshot(from: repo, matching: ["config.json"])

        // Verify same destination
        XCTAssertEqual(destination1.path, destination2.path, "Should return same destination")

        // Verify file was not re-downloaded (modification date unchanged)
        let attrs2 = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let modDate2 = attrs2[.modificationDate] as? Date
        XCTAssertEqual(modDate1, modDate2, "File should not be re-downloaded")
    }

    // MARK: - Offline Mode Tests

    /// Tests that offline mode returns cached files
    func testOfflineModeReturnsCachedFiles() async throws {
        let repo = Hub.Repo(id: "t5-base")

        // First, download with online mode
        let onlineDestination = try await hubApi.snapshot(from: repo, matching: ["config.json"])
        let configPath = onlineDestination.appending(path: "config.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path))

        // Now use offline mode - should return cached files
        let offlineApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)
        let offlineDestination = try await offlineApi.snapshot(from: repo, matching: ["config.json"])

        XCTAssertEqual(onlineDestination.path, offlineDestination.path, "Offline should return same path")
        XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path), "File should exist from cache")
    }

    /// Tests that offline mode fails when files are not cached
    func testOfflineModeFailsWithoutCache() async throws {
        let repo = Hub.Repo(id: "t5-base")

        // Use a fresh cache directory that has no cached files
        let freshCacheDir = downloadDestination.appending(path: "fresh-cache-\(UUID().uuidString)")
        let offlineApi = HubApi(downloadBase: freshCacheDir, useOfflineMode: true)

        do {
            _ = try await offlineApi.snapshot(from: repo, matching: ["config.json"])
            XCTFail("Should throw error when files not cached in offline mode")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case .offlineModeError:
                break // Expected
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

}
