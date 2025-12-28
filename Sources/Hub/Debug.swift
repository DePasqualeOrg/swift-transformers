//
//  Debug.swift
//
//  Instrumentation utilities for swift-transformers.
//  Enable by setting SWIFT_TRANSFORMERS_DEBUG=1 environment variable.
//

import Foundation

/// Debug instrumentation for tokenizer and hub operations.
///
/// Enable debug output by setting the `SWIFT_TRANSFORMERS_DEBUG` environment variable:
/// ```bash
/// SWIFT_TRANSFORMERS_DEBUG=1 swift run MyApp
/// ```
///
/// All logging and timing operations are no-ops when debugging is disabled,
/// with negligible performance overhead (single boolean check per call site).
public enum Debug {
    /// Whether debug output is enabled. Evaluated once at startup.
    public static let isEnabled: Bool = {
        ProcessInfo.processInfo.environment["SWIFT_TRANSFORMERS_DEBUG"] != nil
    }()

    /// Log a debug message. No-op when debugging is disabled.
    ///
    /// Uses `@autoclosure` to avoid string interpolation costs when disabled.
    ///
    /// - Parameter message: The message to log (lazily evaluated)
    @inlinable
    public static func log(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print("[swift-transformers] \(message())")
    }

    /// Time a synchronous block and log the duration.
    ///
    /// - Parameters:
    ///   - label: Description of the operation being timed
    ///   - block: The code block to execute and time
    /// - Returns: The result of the block
    /// - Throws: Rethrows any error from the block
    @inlinable
    public static func time<T>(_ label: @autoclosure () -> String, _ block: () throws -> T) rethrows -> T {
        guard isEnabled else { return try block() }
        let labelValue = label()
        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("[swift-transformers] \(labelValue): \(String(format: "%.2f", elapsed))ms")
        return result
    }

    /// Time an async block and log the duration.
    ///
    /// - Parameters:
    ///   - label: Description of the operation being timed
    ///   - block: The async code block to execute and time
    /// - Returns: The result of the block
    /// - Throws: Rethrows any error from the block
    @inlinable
    public static func time<T>(_ label: @autoclosure () -> String, _ block: () async throws -> T) async rethrows -> T {
        guard isEnabled else { return try await block() }
        let labelValue = label()
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("[swift-transformers] \(labelValue): \(String(format: "%.2f", elapsed))ms")
        return result
    }

    /// Log file size information.
    ///
    /// - Parameters:
    ///   - data: The data whose size to report
    ///   - filename: The filename for context
    @inlinable
    public static func logDataSize(_ data: Data, filename: String) {
        guard isEnabled else { return }
        let kb = Double(data.count) / 1024.0
        if kb > 1024 {
            print("[swift-transformers] \(filename): \(String(format: "%.2f", kb / 1024.0)) MB")
        } else {
            print("[swift-transformers] \(filename): \(String(format: "%.2f", kb)) KB")
        }
    }
}
