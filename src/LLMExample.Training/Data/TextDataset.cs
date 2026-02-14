// =============================================================================
// TextDataset.cs - Training Data Pipeline
// =============================================================================
// This class handles the entire data pipeline for training:
//   1. LOAD: Read all .txt files from a directory
//   2. TOKENIZE: Convert text into sequences of token IDs using the BPE tokenizer
//   3. CONCATENATE: Join all tokenized documents into one long array,
//      separated by end-of-text (EOS) tokens
//   4. BATCH: Serve random chunks of tokens as training examples
//
// HOW LANGUAGE MODEL TRAINING DATA WORKS:
// ========================================
// A language model learns to predict the next token given previous tokens.
// So for a sequence like [A, B, C, D, E], we create:
//   Input:  [A, B, C, D]   (everything except the last token)
//   Target: [B, C, D, E]   (everything shifted right by one)
//
// The model sees [A] and tries to predict [B].
// Then sees [A, B] and tries to predict [C].
// And so on. The loss function measures how wrong the predictions are.
//
// We randomly sample chunks of length BlockSize from our giant token array
// to create diverse training batches.
// =============================================================================

using TorchSharp;
using static TorchSharp.torch;
using LLMExample.Training.Config;

namespace LLMExample.Training.Data;

public class TextDataset
{
    // All tokens from all files, concatenated into one long array.
    // This is the standard approach for language model training --
    // we treat all our text as one continuous stream of tokens.
    private long[] _tokens = Array.Empty<long>();

    private int _blockSize;
    private readonly Random _rng = new();

    /// <summary>
    /// Total number of tokens in the dataset.
    /// </summary>
    public long TokenCount => _tokens.Length;

    /// <summary>
    /// Load all .txt files from the specified directory, tokenize them,
    /// and prepare them for training.
    /// </summary>
    /// <param name="directory">Path to directory containing .txt files</param>
    /// <param name="tokenizer">The BPE tokenizer to convert text → token IDs</param>
    /// <param name="blockSize">The context window size (how many tokens per sample)</param>
    public void Load(string directory, TokenizerWrapper tokenizer, int blockSize)
    {
        _blockSize = blockSize;

        // Find all .txt files in the directory
        var files = Directory.GetFiles(directory, "*.txt");
        if (files.Length == 0)
        {
            throw new InvalidOperationException(
                $"No .txt files found in '{directory}'. " +
                "Please add at least one .txt file to train on.");
        }

        Console.WriteLine($"Found {files.Length} text file(s) in '{directory}'");

        // Tokenize each file and collect all tokens
        var allTokens = new List<long>();

        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            Console.WriteLine($"  Loading: {Path.GetFileName(file)} ({text.Length:N0} characters)");

            // Tokenize the text. Each word/subword becomes a number.
            var tokens = tokenizer.Encode(text);
            allTokens.AddRange(tokens.Select(t => (long)t));

            // Add an EOS (end-of-sequence) token between documents.
            // This teaches the model where one document ends and another begins.
            // Without this, the model would think all documents are one continuous text.
            allTokens.Add(tokenizer.EosTokenId);
        }

        _tokens = allTokens.ToArray();

        Console.WriteLine($"  Total tokens: {_tokens.Length:N0}");
        Console.WriteLine($"  Vocabulary utilization: {_tokens.Distinct().Count():N0} unique tokens used");

        // Sanity check: we need at least blockSize+1 tokens to create even one example
        if (_tokens.Length < blockSize + 1)
        {
            throw new InvalidOperationException(
                $"Not enough training data. Need at least {blockSize + 1} tokens, " +
                $"but only got {_tokens.Length}. Add more text to your training files.");
        }
    }

    /// <summary>
    /// Get a random batch of training examples.
    ///
    /// Returns (input, target) where:
    ///   input  = tensor of shape [batchSize, blockSize] with token IDs
    ///   target = tensor of shape [batchSize, blockSize] with next-token IDs
    ///
    /// Each row is a random chunk from the dataset, and target is shifted by 1.
    /// </summary>
    /// <param name="batchSize">Number of examples in the batch</param>
    /// <param name="device">The device (CPU/MPS/CUDA) to place tensors on</param>
    public (Tensor input, Tensor target) GetBatch(int batchSize, Device device)
    {
        // We need blockSize tokens for input + 1 more for the last target token,
        // so valid starting positions are [0, _tokens.Length - blockSize - 1).
        int maxStart = _tokens.Length - _blockSize - 1;

        // Allocate arrays for the batch
        var inputData = new long[batchSize * _blockSize];
        var targetData = new long[batchSize * _blockSize];

        for (int b = 0; b < batchSize; b++)
        {
            // Pick a random starting position in the token array
            int start = _rng.Next(0, maxStart);

            // Copy blockSize tokens as input, and the next blockSize tokens
            // (shifted by 1) as the target
            for (int i = 0; i < _blockSize; i++)
            {
                inputData[b * _blockSize + i] = _tokens[start + i];
                targetData[b * _blockSize + i] = _tokens[start + i + 1]; // shifted by 1!
            }
        }

        // Convert to TorchSharp tensors and move to the target device
        //
        // WHY Int64 (long)?
        // Token IDs are indices into the embedding table. PyTorch/TorchSharp
        // requires these to be 64-bit integers for indexing operations.
        var input = torch.tensor(inputData, dtype: ScalarType.Int64)
            .reshape(batchSize, _blockSize)
            .to(device);

        var target = torch.tensor(targetData, dtype: ScalarType.Int64)
            .reshape(batchSize, _blockSize)
            .to(device);

        return (input, target);
    }
}
