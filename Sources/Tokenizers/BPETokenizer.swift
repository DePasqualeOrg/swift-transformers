//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    /// Merge ranks stored as packed token ID pairs for fast integer hashing.
    /// Key is `(idA << 32) | idB`, value is the merge rank.
    private let bpeRanks: [UInt64: Int]
    /// Token string to ID mapping. Uses NSString keys to preserve exact byte sequences.
    private let tokensToIds: [NSString: Int]
    /// ID to token mapping. Uses NSString values to preserve exact byte sequences (e.g., BOM chars).
    private let idsToTokens: [Int: NSString]
    /// Normalized String to ID mapping. Only built for tokenizers with Unicode edge cases
    /// (when NSString→String conversion causes collisions due to normalization).
    private let stringToId: [String: Int]?

    /// Packs two token IDs into a single UInt64 for fast merge lookup.
    @inline(__always)
    private static func packIds(_ a: Int, _ b: Int) -> UInt64 {
        UInt64(UInt32(truncatingIfNeeded: a)) << 32 | UInt64(UInt32(truncatingIfNeeded: b))
    }

    /// Looks up the merge rank for a pair of token strings.
    @inline(__always)
    private func mergeRank(_ a: String, _ b: String) -> Int? {
        guard let idA = tokensToIds[a as NSString] ?? stringToId?[a],
            let idB = tokensToIds[b as NSString] ?? stringToId?[b]
        else { return nil }
        return bpeRanks[Self.packIds(idA, idB)]
    }

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Parse merges from raw JSON array, supporting both formats:
    /// - Modern: `[["a", "b"], ["c", "d"]]` - array of string pairs
    /// - Legacy: `["a b", "c d"]` - space-separated strings
    /// Returns NSString pairs to preserve Unicode (avoids normalization that loses BOM chars).
    /// Uses ContiguousArray for better iteration performance.
    static func mergesFromRawJSON(_ rawMerges: [Any]) -> ContiguousArray<(NSString, NSString)> {
        var result = ContiguousArray<(NSString, NSString)>()
        result.reserveCapacity(rawMerges.count)
        for element in rawMerges {
            // Modern format: array of two strings
            if let pair = element as? [Any], pair.count == 2,
                let a = pair[0] as? NSString,
                let b = pair[1] as? NSString
            {
                result.append((a, b))
                continue
            }
            // Legacy format: space-separated string
            if let str = element as? NSString {
                let range = str.range(of: " ")
                if range.location != NSNotFound {
                    let a = str.substring(to: range.location) as NSString
                    let b = str.substring(from: range.location + 1) as NSString
                    result.append((a, b))
                }
            }
        }
        return result
    }

    // MARK: - Static Dictionary Builders (for parallel construction)

    /// Builds tokensToIds dictionary from raw vocabulary.
    /// Uses NSString keys to preserve exact byte sequences (e.g., BOM characters).
    static func buildTokensToIds(
        rawVocab: NSDictionary,
        addedTokens: [String: Int]
    ) -> [NSString: Int] {
        var tokensToIds: [NSString: Int] = [:]
        tokensToIds.reserveCapacity(rawVocab.count + addedTokens.count)
        for (key, idValue) in rawVocab {
            guard let token = key as? NSString else { continue }
            if let id = idValue as? Int {
                tokensToIds[token] = id
            } else if let id = (idValue as? NSNumber)?.intValue {
                tokensToIds[token] = id
            }
        }
        for (token, id) in addedTokens {
            tokensToIds[token as NSString] = id
        }
        return tokensToIds
    }

    /// Builds bpeRanks dictionary using packed token IDs.
    static func buildBpeRanks(
        merges: ContiguousArray<(NSString, NSString)>,
        tokensToIds: [NSString: Int]
    ) -> [UInt64: Int] {
        var bpeRanks: [UInt64: Int] = [:]
        bpeRanks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            guard let idA = tokensToIds[merge.0],
                let idB = tokensToIds[merge.1]
            else { continue }
            bpeRanks[packIds(idA, idB)] = rank
        }
        return bpeRanks
    }

    /// Builds idsToTokens dictionary (inverse of tokensToIds).
    static func buildIdsToTokens(from tokensToIds: [NSString: Int]) -> [Int: NSString] {
        Utils.invert(tokensToIds)
    }

    /// Builds stringToId only if Unicode normalization causes collisions.
    /// Returns nil if no collisions detected (no stringToId needed).
    static func buildStringToIdIfNeeded(from tokensToIds: [NSString: Int]) -> [String: Int]? {
        var stringToId: [String: Int] = [:]
        stringToId.reserveCapacity(tokensToIds.count)
        for (nsKey, id) in tokensToIds {
            stringToId[nsKey as String] = id
        }
        // If counts differ, Unicode normalization caused collisions - we need this dict
        if stringToId.count < tokensToIds.count {
            return stringToId
        }
        // No collisions - stringToId not needed, save 50ms
        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }

        // Build tokensToIds with NSString keys to preserve exact byte sequences
        let addedTokensDict = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        let tokensToIds = vocab.merging(addedTokensDict) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }
        self.tokensToIds = tokensToIds

        // Build stringToId only if Unicode normalization causes collisions
        self.stringToId = Self.buildStringToIdIfNeeded(from: tokensToIds)

        self.idsToTokens = Self.buildIdsToTokens(from: tokensToIds)

        // Build bpeRanks using packed token IDs for fast integer hashing
        var bpeRanks: [UInt64: Int] = [:]
        bpeRanks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            guard let idA = tokensToIds[merge[0] as NSString],
                let idB = tokensToIds[merge[1] as NSString]
            else { continue }
            bpeRanks[Self.packIds(idA, idB)] = rank
        }
        self.bpeRanks = bpeRanks

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Fast-path initializer using pre-extracted vocab and merges.
    ///
    /// This initializer bypasses Config parsing for the large vocab/merges data,
    /// significantly improving tokenizer loading performance.
    init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        rawVocab: NSDictionary,
        rawMerges: [Any]
    ) throws {
        // Use static builders for consistency with async path
        let tokensToIds = Self.buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
        let merges = Self.mergesFromRawJSON(rawMerges)

        self.tokensToIds = tokensToIds
        self.stringToId = Self.buildStringToIdIfNeeded(from: tokensToIds)
        self.bpeRanks = Self.buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
        self.idsToTokens = Self.buildIdsToTokens(from: tokensToIds)

        // Populate special tokens from config
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Initializer accepting pre-built dictionaries (used by async factory for parallel construction).
    init(
        tokenizerConfig: Config,
        tokensToIds: [NSString: Int],
        stringToId: [String: Int]?,
        bpeRanks: [UInt64: Int],
        idsToTokens: [Int: NSString]
    ) {
        self.tokensToIds = tokensToIds
        self.stringToId = stringToId
        self.bpeRanks = bpeRanks
        self.idsToTokens = idsToTokens

        // Populate special tokens from config
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Async factory that builds dictionaries in parallel for faster loading.
    ///
    /// Uses Swift concurrency (`async let`) to maximize parallelism:
    /// 1. tokensToIds and merges parsing run in parallel (independent)
    /// 2. stringToId, bpeRanks, idsToTokens run in parallel (depend on tokensToIds)
    static func createAsync(
        tokenizerConfig: Config,
        rawVocab: NSDictionary,
        rawMerges: [Any],
        addedTokens: [String: Int]
    ) async -> BPETokenizer {
        // Phase 1: Build tokensToIds and parse merges in parallel (independent)
        async let tokensToIdsTask = buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
        async let mergesTask = mergesFromRawJSON(rawMerges)

        let tokensToIds = await tokensToIdsTask
        let merges = await mergesTask

        // Phase 2: Build remaining dicts in parallel (all depend on tokensToIds)
        async let stringToIdTask = buildStringToIdIfNeeded(from: tokensToIds)
        async let bpeRanksTask = buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
        async let idsToTokensTask = buildIdsToTokens(from: tokensToIds)

        return await BPETokenizer(
            tokenizerConfig: tokenizerConfig,
            tokensToIds: tokensToIds,
            stringToId: stringToIdTask,
            bpeRanks: bpeRanksTask,
            idsToTokens: idsToTokensTask
        )
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? stringToId?[token] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { token -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }

    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { token -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }

    func bpe(token: String) -> String {
        if token.count <= 1 {
            return token
        }

        var word = Array(token).map { String($0) }

        while word.count > 1 {
            // Find the pair with the lowest merge rank
            var minRank = Int.max
            var minPair: (first: String, second: String)?

            for i in 0..<(word.count - 1) {
                if let rank = mergeRank(word[i], word[i + 1]), rank < minRank {
                    minRank = rank
                    minPair = (word[i], word[i + 1])
                }
            }

            guard let pair = minPair else { break }

            // Merge all occurrences of the selected pair
            let first = pair.first
            let second = pair.second
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i..<word.count].firstIndex(of: first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i..<word.count])
                    break
                }
                if word[i] == first, i < word.count - 1, word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }
        return word.joined(separator: " ")
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text).split(separator: " ").map { String($0) }
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}
