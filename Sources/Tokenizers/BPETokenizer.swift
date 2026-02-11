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
///
/// Merge ranks are stored as packed UInt64 keys (two token IDs packed into one integer)
/// for fast integer hashing instead of string-based `BytePair` hashing.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    /// Merge ranks stored as packed token ID pairs for fast integer hashing.
    /// Key is `(idA << 32) | idB`, value is the merge rank.
    private let bpeRanks: [UInt64: Int]

    /// Token string to ID mapping. Uses NSString keys to preserve exact byte sequences.
    private let tokensToIds: [NSString: Int]

    /// ID to token mapping. Uses NSString values to preserve exact byte sequences.
    private let idsToTokens: [Int: NSString]

    // TODO: Benchmark whether building stringToId in parallel adds to loading time
    /// Canonical String-keyed fallback for tokenizers whose vocabulary contains tokens
    /// in different Unicode normalization forms (e.g., Gemma uses NFD for some entries).
    /// NSString comparison is byte-level, so NFC vs NFD forms won't match. This dict
    /// uses Swift's canonical String comparison as a fallback. Only non-nil when the
    /// vocabulary actually contains normalization collisions.
    private let stringToId: [String: Int]?

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

    // MARK: - UInt64-Packed Merge Rank Helpers

    /// Packs two token IDs into a single UInt64 for fast merge lookup.
    /// Assumes token IDs fit in UInt32 (max ~4.3B tokens).
    @inline(__always)
    private static func packIds(_ a: Int, _ b: Int) -> UInt64 {
        UInt64(UInt32(a)) << 32 | UInt64(UInt32(b))
    }

    /// Looks up the merge rank for a pair of token strings.
    @inline(__always)
    private func mergeRank(_ a: String, _ b: String) -> Int? {
        guard let idA = tokensToIds[a as NSString] ?? stringToId?[a],
            let idB = tokensToIds[b as NSString] ?? stringToId?[b]
        else { return nil }
        return bpeRanks[Self.packIds(idA, idB)]
    }

    // MARK: - Static Dictionary Builders

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

    /// Builds a String-keyed fallback dict only when Unicode normalization causes
    /// collisions between NSString keys (i.e., distinct NSStrings map to the same String).
    /// Returns nil when no collisions exist, avoiding memory overhead for most tokenizers.
    static func buildStringToIdIfNeeded(from tokensToIds: [NSString: Int]) -> [String: Int]? {
        var stringToId: [String: Int] = [:]
        stringToId.reserveCapacity(tokensToIds.count)
        for (nsKey, id) in tokensToIds {
            stringToId[nsKey as String] = id
        }
        if stringToId.count < tokensToIds.count {
            return stringToId
        }
        return nil
    }

    /// Parse merges from raw JSON array, supporting both formats:
    /// - Modern: `[["a", "b"], ["c", "d"]]` - array of string pairs
    /// - Legacy: `["a b", "c d"]` - space-separated strings
    /// Returns NSString pairs to preserve Unicode (avoids normalization that loses BOM chars).
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

    /// Parse merges from Config, supporting both formats:
    /// - Modern: each merge is a list of 2 items (tokenizers >= 0.20.0)
    /// - Legacy: each merge is a space-separated string
    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() {
                    result.append(val)
                }
                if let val: String = element.get() {
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    // MARK: - Private Designated Initializer

    /// Initializer accepting pre-built dictionaries.
    /// All other initializers delegate to this one.
    private init(
        tokensToIds: [NSString: Int],
        bpeRanks: [UInt64: Int],
        idsToTokens: [Int: NSString],
        stringToId: [String: Int]?,
        tokenizerConfig: Config
    ) {
        self.tokensToIds = tokensToIds
        self.bpeRanks = bpeRanks
        self.idsToTokens = idsToTokens
        self.stringToId = stringToId

        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken.flatMap { tokensToIds[$0 as NSString] }

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken.flatMap { tokensToIds[$0 as NSString] }

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    // MARK: - Protocol Conformance Init

    /// Initializes a BPE tokenizer from configuration data with optional pre-extracted vocab/merges.
    ///
    /// When vocab and merges are provided (fast path), builds dictionaries from raw data.
    /// Otherwise falls back to parsing from Config.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    ///   - vocab: Pre-extracted vocabulary (nil to parse from Config)
    ///   - merges: Pre-extracted merge rules (nil to parse from Config)
    /// - Throws: `TokenizerError` if required configuration is missing
    required convenience init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        vocab: TokenizerVocab? = nil,
        merges: TokenizerMerges? = nil
    ) throws {
        let tokensToIds: [NSString: Int]
        let bpeRanks: [UInt64: Int]

        if case .bpe(let rawVocab) = vocab, let merges {
            // Fast path: build from pre-extracted raw data
            tokensToIds = Self.buildTokensToIds(rawVocab: rawVocab, addedTokens: addedTokens)
            let parsedMerges = Self.mergesFromRawJSON(merges.rules)
            bpeRanks = Self.buildBpeRanks(merges: parsedMerges, tokensToIds: tokensToIds)
        } else {
            // Fallback: parse vocab and merges from Config
            guard let configMerges = Self.mergesFromConfig(tokenizerData.model.merges) else {
                fatalError("BPETokenizer requires merges")
            }
            guard let configVocab = tokenizerData.model.vocab.dictionary() else {
                throw TokenizerError.missingVocab
            }

            let addedTokensDict = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
                result[BinaryDistinctString(element.key)] = .init(element.value)
            }
            tokensToIds = configVocab.merging(addedTokensDict) { $1 }.reduce(into: [NSString: Int]()) { result, element in
                result[element.key.nsString] = element.value.integer()
            }

            var ranks: [UInt64: Int] = [:]
            ranks.reserveCapacity(configMerges.count)
            for (rank, merge) in configMerges.enumerated() {
                guard let idA = tokensToIds[merge[0] as NSString],
                    let idB = tokensToIds[merge[1] as NSString]
                else { continue }
                ranks[Self.packIds(idA, idB)] = rank
            }
            bpeRanks = ranks
        }

        self.init(
            tokensToIds: tokensToIds,
            bpeRanks: bpeRanks,
            idsToTokens: Self.buildIdsToTokens(from: tokensToIds),
            stringToId: Self.buildStringToIdIfNeeded(from: tokensToIds),
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Async Factory

    /// Async factory that builds dictionaries in parallel for faster loading.
    ///
    /// Uses Swift concurrency (`async let`) to maximize parallelism:
    /// 1. tokensToIds and merges parsing run in parallel (independent)
    /// 2. bpeRanks and idsToTokens run in parallel (depend on tokensToIds)
    static func create(
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
        async let bpeRanksTask = buildBpeRanks(merges: merges, tokensToIds: tokensToIds)
        async let idsToTokensTask = buildIdsToTokens(from: tokensToIds)
        async let stringToIdTask = buildStringToIdIfNeeded(from: tokensToIds)

        let bpeRanks = await bpeRanksTask
        let idsToTokens = await idsToTokensTask
        let stringToId = await stringToIdTask

        return BPETokenizer(
            tokensToIds: tokensToIds,
            bpeRanks: bpeRanks,
            idsToTokens: idsToTokens,
            stringToId: stringToId,
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Token Conversion

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

    // MARK: - Encoding Helpers

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

    // MARK: - BPE Algorithm

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
