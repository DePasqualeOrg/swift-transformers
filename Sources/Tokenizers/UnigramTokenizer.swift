//
//  UnigramTokenizer.swift
//
//
//  Created by Pedro Cuenca on 20240131.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Foundation
import Hub

/// A Unigram tokenizer implementation based on the SentencePiece algorithm.
///
/// Unigram tokenizers use a probabilistic approach where each token has a score,
/// and the tokenization process finds the most probable segmentation of the input text.
/// This is commonly used in models like T5 and XLM-RoBERTa.
class UnigramTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    /// A token with its associated score in the Unigram model.
    struct SentencePieceToken {
        var token: String
        var score: Float
    }

    /// The complete vocabulary of tokens with their scores.
    let vocab: [SentencePieceToken]

    /// The special token used for unknown/out-of-vocabulary text.
    let unknownPiece: SentencePieceToken

    /// The score associated with the unknown token.
    var unknownTokenScore: Float { unknownPiece.score }

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// The unknown token string.
    var unknownToken: String? { unknownPiece.token }

    /// The minimum score found in the vocabulary (used for score calculations).
    let minScore: Float

    /// Mapping from token strings to their numeric IDs.
    let tokensToIds: [NSString: Int]

    /// The beginning-of-sequence token (hardcoded as space for Unigram).
    let bosToken: String? = " "

    /// The numeric ID of the beginning-of-sequence token.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// Whether consecutive unknown tokens should be fused (always true for Unigram).
    let fuseUnknownTokens: Bool = true

    private let trie: Trie<Character>

    // MARK: - Static Builder Methods

    /// Parses raw JSON vocab data into SentencePieceToken array.
    ///
    /// - Parameter rawVocab: Array of [token, score] pairs from JSON
    /// - Returns: Parsed vocabulary array
    static func buildVocab(from rawVocab: [[Any]]) -> [SentencePieceToken] {
        rawVocab.compactMap { pair -> SentencePieceToken? in
            guard pair.count == 2,
                let token = pair[0] as? String
            else { return nil }

            let score: Float
            if let floatScore = pair[1] as? Double {
                score = Float(floatScore)
            } else if let intScore = pair[1] as? Int {
                score = Float(intScore)
            } else {
                return nil
            }

            return SentencePieceToken(token: token, score: score)
        }
    }

    /// Builds the token-to-ID mapping dictionary from a vocab array.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: Dictionary mapping token strings (as NSString) to their integer IDs
    static func buildTokensToIds(from vocab: [SentencePieceToken]) -> [NSString: Int] {
        Dictionary(
            uniqueKeysWithValues: vocab.map { $0.token as NSString }.enumerated().map { ($1, $0) }
        )
    }

    /// Builds a character trie from the vocabulary for prefix matching.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: A populated Trie for efficient common-prefix search
    static func buildTrie(from vocab: [SentencePieceToken]) -> Trie<Character> {
        let trie = Trie<Character>()
        trie.append(contentsOf: vocab.map { $0.token })
        return trie
    }

    /// Computes the minimum score across all tokens in the vocabulary.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: The minimum score value
    static func computeMinScore(from vocab: [SentencePieceToken]) -> Float {
        vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }
    }

    // MARK: - Private Designated Initializer

    /// Initializer accepting pre-built data structures.
    /// All other initializers and the async factory delegate to this one.
    private init(
        vocab: [SentencePieceToken],
        tokensToIds: [NSString: Int],
        trie: Trie<Character>,
        minScore: Float,
        unknownTokenId: Int,
        tokenizerConfig: Config
    ) {
        self.vocab = vocab
        self.tokensToIds = tokensToIds
        self.trie = trie
        self.minScore = minScore
        self.unknownTokenId = unknownTokenId
        self.unknownPiece = SentencePieceToken(
            token: vocab[unknownTokenId].token,
            score: minScore - 10
        )

        // bosToken is hardcoded as " " for Unigram tokenizers
        self.bosTokenId = tokensToIds[" " as NSString]

        let eos = tokenizerConfig.eosToken.string()
        self.eosToken = eos
        if let eos {
            self.eosTokenId = tokensToIds[eos as NSString]
        } else {
            self.eosTokenId = nil
        }
    }

    // MARK: - Protocol Conformance Init

    /// Initializes a Unigram tokenizer from configuration data with optional pre-extracted vocab.
    ///
    /// When vocab is provided (fast path), builds data structures from raw arrays.
    /// Otherwise falls back to parsing from Config.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and scores
    ///   - addedTokens: Additional tokens to include in the vocabulary
    ///   - vocab: Pre-extracted vocabulary (nil to parse from Config)
    ///   - merges: Pre-extracted merge rules (unused by Unigram)
    /// - Throws: `TokenizerError` if the vocabulary is missing or malformed
    required convenience init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        vocab: TokenizerVocab? = nil,
        merges: TokenizerMerges? = nil
    ) throws {
        let vocabArray: [SentencePieceToken]

        if case .unigram(let rawVocabArray) = vocab, let rawVocab = rawVocabArray as? [[Any]] {
            // Fast path: build from pre-extracted raw data
            vocabArray = Self.buildVocab(from: rawVocab)
        } else {
            // Fallback: parse from Config
            guard let configVocab = tokenizerData.model.vocab.array() else {
                throw TokenizerError.missingVocab
            }

            vocabArray = try configVocab.map { piece -> SentencePieceToken in
                let tuple = piece.array(or: [])

                guard let token = tuple.first?.string(),
                    let scoreValue = tuple.last
                else {
                    throw TokenizerError.malformedVocab
                }

                let score: Float
                if let floatScore = scoreValue.floating() {
                    score = floatScore
                } else if let numberScore = scoreValue.integer() {
                    score = Float(numberScore)
                } else {
                    throw TokenizerError.malformedVocab
                }

                return SentencePieceToken(token: token, score: score)
            }
        }

        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else {
            throw TokenizerError.malformedVocab
        }

        self.init(
            vocab: vocabArray,
            tokensToIds: Self.buildTokensToIds(from: vocabArray),
            trie: Self.buildTrie(from: vocabArray),
            minScore: Self.computeMinScore(from: vocabArray),
            unknownTokenId: unknownTokenId,
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Async Factory

    /// Async factory that builds data structures in parallel for faster loading.
    ///
    /// Phase 1: Parse raw vocab array into [SentencePieceToken] (sequential, since later
    ///          phases all depend on the parsed vocab).
    /// Phase 2: Build tokensToIds, trie, and computeMinScore in parallel (independent).
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data (used for unkId)
    ///   - rawVocab: Raw vocab array of [token, score] pairs
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Returns: A fully initialized UnigramTokenizer
    /// - Throws: `TokenizerError` if the vocabulary is malformed
    static func create(
        tokenizerConfig: Config,
        tokenizerData: Config,
        rawVocab: [[Any]],
        addedTokens: [String: Int]
    ) async throws -> UnigramTokenizer {
        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else {
            throw TokenizerError.malformedVocab
        }

        // Phase 1: Parse raw vocab (sequential, all other phases depend on this)
        let vocabArray = buildVocab(from: rawVocab)

        // Phase 2: Build independent data structures in parallel
        async let tokensToIdsTask = buildTokensToIds(from: vocabArray)
        async let trieTask = buildTrie(from: vocabArray)
        async let minScoreTask = computeMinScore(from: vocabArray)

        let tokensToIds = await tokensToIdsTask
        let trie = await trieTask
        let minScore = await minScoreTask

        return UnigramTokenizer(
            vocab: vocabArray,
            tokensToIds: tokensToIds,
            trie: trie,
            minScore: minScore,
            unknownTokenId: unknownTokenId,
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Token Conversion

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string
    func convertIdToToken(_ id: Int) -> String? {
        vocab[id].token
    }

    // MARK: - Tokenization

    /// Tokenizes input text using the Unigram algorithm with dynamic programming.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings representing the most probable segmentation
    func tokenize(text: String) -> [String] {
        var lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId ?? 0, eosTokenId: eosTokenId ?? 0)

        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false

            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                guard let tokenId = tokensToIds[token as NSString] else { fatalError("Token not in vocab: \(token)") }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode, token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        return lattice.tokens
    }
}
