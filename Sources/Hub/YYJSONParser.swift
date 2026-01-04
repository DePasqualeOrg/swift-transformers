//
//  YYJSONParser.swift
//  swift-transformers
//
//  High-performance JSON parsing using yyjson.
//

import Foundation
import yyjson

/// A high-performance JSON parser using yyjson.
///
/// This parser provides significantly faster JSON parsing compared to Foundation's
/// JSONSerialization, especially for large files like tokenizer.json (10+ MB).
enum YYJSONParser {
    /// Error types for yyjson parsing failures.
    enum ParseError: Error, LocalizedError {
        case readFailed(code: UInt32, message: String, position: Int)
        case nullDocument

        var errorDescription: String? {
            switch self {
            case .readFailed(let code, let message, let position):
                return "yyjson read failed (code \(code)) at position \(position): \(message)"
            case .nullDocument:
                return "yyjson returned null document"
            }
        }
    }

    /// Parses JSON data directly into a Config object.
    ///
    /// This is the most efficient path as it goes directly from yyjson to Config
    /// without intermediate Foundation object creation.
    ///
    /// - Parameter data: The JSON data to parse
    /// - Returns: A Config object
    /// - Throws: ParseError if parsing fails
    static func parseToConfig(_ data: Data) throws -> Config {
        try data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> Config in
            guard let baseAddress = buffer.baseAddress else {
                throw ParseError.nullDocument
            }

            var err = yyjson_read_err()
            let doc = yyjson_read_opts(
                UnsafeMutableRawPointer(mutating: baseAddress).assumingMemoryBound(to: CChar.self),
                buffer.count,
                0,
                nil,
                &err
            )

            guard let doc = doc else {
                let message = err.msg.map { String(cString: $0) } ?? "unknown error"
                throw ParseError.readFailed(code: err.code, message: message, position: err.pos)
            }

            defer { yyjson_doc_free(doc) }

            guard let root = yyjson_doc_get_root(doc) else {
                throw ParseError.nullDocument
            }

            return convertToConfig(root)
        }
    }

    /// Parses JSON data into a Config object, preserving BOM characters in strings.
    ///
    /// Unlike Foundation's `JSONSerialization`, yyjson correctly preserves BOM
    /// characters (`\u{feff}`) within string values. This matters for tokenizers
    /// like Gemma that use BOM as a token prefix (e.g., `"\u{feff}#"`).
    ///
    /// See: https://github.com/huggingface/swift-transformers/issues/116
    static func bomPreservingParseToConfig(_ data: Data) throws -> Config {
        try parseToConfig(data)
    }

    // MARK: - Direct Config conversion

    private static func convertToConfig(_ val: UnsafeMutablePointer<yyjson_val>) -> Config {
        let tag = yyjson_get_tag(val)
        let type = tag & 0x07
        let subtype = tag & 0x18

        switch type {
        case 0x02: // YYJSON_TYPE_NULL
            return Config()
        case 0x03: // YYJSON_TYPE_BOOL
            return Config(subtype == 0x08)
        case 0x04: // YYJSON_TYPE_NUM
            if subtype == 0x00 { // YYJSON_SUBTYPE_UINT
                return Config(Int(yyjson_get_uint(val)))
            } else if subtype == 0x08 { // YYJSON_SUBTYPE_SINT
                return Config(Int(yyjson_get_sint(val)))
            } else { // YYJSON_SUBTYPE_REAL
                return Config(Float(yyjson_get_real(val)))
            }
        case 0x05: // YYJSON_TYPE_STR
            guard let str = yyjson_get_str(val) else { return Config("") }
            return Config(String(cString: str))
        case 0x06: // YYJSON_TYPE_ARR
            return convertArrayToConfig(val)
        case 0x07: // YYJSON_TYPE_OBJ
            return convertObjectToConfig(val)
        default:
            return Config()
        }
    }

    private static func convertObjectToConfig(_ obj: UnsafeMutablePointer<yyjson_val>) -> Config {
        let size = yyjson_obj_size(obj)
        var result: [BinaryDistinctString: Config] = Dictionary(minimumCapacity: Int(size))

        var iter = yyjson_obj_iter()
        yyjson_obj_iter_init(obj, &iter)

        while let key = yyjson_obj_iter_next(&iter) {
            guard let keyPtr = yyjson_get_str(key),
                let val = yyjson_obj_iter_get_val(key)
            else {
                continue
            }

            let keyString = String(cString: keyPtr)
            result[BinaryDistinctString(keyString)] = convertToConfig(val)
        }

        return Config(result)
    }

    private static func convertArrayToConfig(_ arr: UnsafeMutablePointer<yyjson_val>) -> Config {
        let size = yyjson_arr_size(arr)
        var result: [Config] = []
        result.reserveCapacity(Int(size))

        var iter = yyjson_arr_iter()
        yyjson_arr_iter_init(arr, &iter)

        while let val = yyjson_arr_iter_next(&iter) {
            result.append(convertToConfig(val))
        }

        return Config(result)
    }
}
