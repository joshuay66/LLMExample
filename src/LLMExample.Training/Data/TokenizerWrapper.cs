// =============================================================================
// TokenizerWrapper.cs - GPT-2 BPE Tokenizer
// =============================================================================
// Before a language model can process text, the text must be converted into
// numbers (tokens). This is called "tokenization."
//
// GPT-2 uses BPE (Byte Pair Encoding), which works like this:
//   1. Start with individual bytes/characters as the base vocabulary.
//   2. Find the most frequently occurring pair of adjacent tokens in the
//      training corpus (e.g., "t" + "h" appearing together often).
//   3. Merge that pair into a single new token ("th").
//   4. Repeat steps 2-3 until you have the desired vocabulary size (50,257).
//
// The result: common words like "the" become a single token, while rare words
// get split into subword pieces. For example:
//   "Hello world"  → [15496, 995]          (2 tokens - common words)
//   "tokenization"  → [30001, 1634]         (2 tokens - split into subwords)
//   "Anthropomorphic" → [31635, 6361, ...]  (multiple tokens - rare word)
//
// This approach means the model can handle ANY text (even made-up words)
// because it can always fall back to individual byte-level tokens.
//
// We use SharpToken, a C# port of OpenAI's tiktoken library, which provides
// the exact same tokenization as the original GPT-2.
// =============================================================================

using SharpToken;

namespace LLMExample.Training.Data;

public class TokenizerWrapper
{
    private readonly GptEncoding _encoding;

    /// <summary>
    /// The special "end of text" token ID. GPT-2 uses token 50256 for this.
    /// We insert this between documents during training so the model learns
    /// where one piece of text ends and another begins.
    /// </summary>
    public int EosTokenId { get; } = 50256;

    /// <summary>
    /// Total vocabulary size (50,257 tokens: 50,256 BPE tokens + 1 special token).
    /// This must match the model's VocabSize configuration.
    /// </summary>
    public int VocabSize { get; } = 50257;

    public TokenizerWrapper()
    {
        // Load the GPT-2 encoding. SharpToken has this built-in.
        // "r50k_base" is the encoding name for GPT-2's BPE tokenizer with 50,257 tokens.
        // (SharpToken uses encoding names like "r50k_base" rather than model names like "gpt-2")
        _encoding = GptEncoding.GetEncoding("r50k_base");
    }

    /// <summary>
    /// Convert text into a sequence of token IDs.
    /// Example: "Hello world" → [15496, 995]
    /// </summary>
    public List<int> Encode(string text)
    {
        return _encoding.Encode(text);
    }

    /// <summary>
    /// Convert a sequence of token IDs back into text.
    /// Example: [15496, 995] → "Hello world"
    /// </summary>
    public string Decode(IEnumerable<int> tokens)
    {
        return _encoding.Decode(tokens.ToList());
    }
}
