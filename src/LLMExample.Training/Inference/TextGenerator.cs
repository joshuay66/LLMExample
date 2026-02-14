// =============================================================================
// TextGenerator.cs - Autoregressive Text Generation
// =============================================================================
// This is where the trained model generates new text! The process is called
// "autoregressive generation" because each new token becomes part of the
// input for generating the next token.
//
// THE GENERATION LOOP:
// ====================
// 1. Start with a prompt: "Once upon a"
// 2. Feed it to the model → get probabilities for the next token
// 3. Sample from those probabilities → maybe "time"
// 4. Append "time" to the sequence: "Once upon a time"
// 5. Feed the updated sequence back to the model
// 6. Repeat until we hit the max length or an end-of-text token
//
// TEMPERATURE:
// ============
// Temperature controls the "randomness" of generation:
//   - temperature = 0.1: Very deterministic, always picks the most likely token.
//     Good for factual/predictable output. Can be repetitive.
//   - temperature = 1.0: Samples proportionally to the model's confidence.
//     Balanced between quality and creativity.
//   - temperature = 2.0: Very random, often picks unlikely tokens.
//     Creative but often incoherent.
//
// Mathematically: divide the logits by temperature before softmax.
// Lower temperature → sharper probability distribution → less randomness.
// Higher temperature → flatter distribution → more randomness.
//
// NOTE ON OUTPUT QUALITY:
// =======================
// Our tiny 7.3M parameter model trained on limited data will NOT produce
// coherent, human-like text. That requires billions of parameters and
// terabytes of training data. What you WILL see:
//   - Learned patterns from your training data (word frequencies, common phrases)
//   - Some grammatical structure (if trained long enough)
//   - Repetition and occasional nonsense
// This is expected and educational! It shows the MECHANISM of text generation.
// =============================================================================

using TorchSharp;
using static TorchSharp.torch;
using LLMExample.Training.Config;
using LLMExample.Training.Data;
using LLMExample.Training.Model;

namespace LLMExample.Training.Inference;

public class TextGenerator
{
    private readonly Gpt2Model _model;
    private readonly TokenizerWrapper _tokenizer;
    private readonly ModelConfig _config;
    private readonly Device _device;

    public TextGenerator(Gpt2Model model, TokenizerWrapper tokenizer, ModelConfig config, Device device)
    {
        _model = model;
        _tokenizer = tokenizer;
        _config = config;
        _device = device;
    }

    /// <summary>
    /// Generate text from a prompt.
    /// </summary>
    /// <param name="prompt">The starting text (e.g., "Once upon a time")</param>
    /// <param name="maxNewTokens">Maximum number of tokens to generate</param>
    /// <param name="temperature">Controls randomness: lower=more deterministic, higher=more random</param>
    /// <returns>The generated text (including the prompt)</returns>
    public string Generate(string prompt, int maxNewTokens = 200, float temperature = 0.8f)
    {
        // Switch to evaluation mode: disables dropout so generation is deterministic
        // (given the same random seed). During training, dropout randomly zeros neurons;
        // during inference, we want the full model capacity.
        _model.eval();

        // Tokenize the prompt
        var tokens = _tokenizer.Encode(prompt);
        if (tokens.Count == 0)
        {
            // If the prompt is empty, start with the EOS token
            tokens.Add(_tokenizer.EosTokenId);
        }

        // Disable gradient computation. During generation, we don't need gradients
        // (we're not training), so this saves memory and computation.
        using var _ = torch.no_grad();

        for (int i = 0; i < maxNewTokens; i++)
        {
            // Truncate to the context window if the sequence is too long.
            // The model can only "see" BlockSize tokens at once due to the
            // positional embedding limit.
            var contextTokens = tokens.Count > _config.BlockSize
                ? tokens.Skip(tokens.Count - _config.BlockSize).ToList()
                : tokens;

            // Convert token list to a tensor: [1, seq_len] (batch size of 1)
            var inputArray = contextTokens.Select(t => (long)t).ToArray();
            var input = torch.tensor(inputArray, dtype: ScalarType.Int64)
                .unsqueeze(0) // Add batch dimension: [seq_len] → [1, seq_len]
                .to(_device);

            // Forward pass: get logits for all positions
            var logits = _model.forward(input); // [1, seq_len, vocab_size]

            // We only care about the LAST position's predictions
            // (the model's prediction for what comes next)
            var lastLogits = logits[0, -1, TensorIndex.Ellipsis]; // [vocab_size]

            // Apply temperature scaling.
            // Dividing by temperature < 1 makes the distribution sharper (more confident).
            // Dividing by temperature > 1 makes it flatter (more random).
            if (temperature != 1.0f)
            {
                lastLogits = lastLogits / temperature;
            }

            // Convert logits to probabilities using softmax.
            // Softmax: prob_i = exp(logit_i) / sum(exp(logit_j) for all j)
            // This normalizes the scores to sum to 1.0.
            var probs = nn.functional.softmax(lastLogits, dim: -1); // [vocab_size]

            // Sample the next token from the probability distribution.
            // torch.multinomial draws from a categorical distribution:
            // tokens with higher probability are more likely to be picked.
            var nextToken = torch.multinomial(probs, num_samples: 1); // [1]

            // Get the token ID and append to our sequence
            int nextTokenId = (int)nextToken.item<long>();
            tokens.Add(nextTokenId);

            // Clean up tensors
            input.Dispose();
            logits.Dispose();
            lastLogits.Dispose();
            probs.Dispose();
            nextToken.Dispose();

            // Stop if we generated an end-of-text token
            if (nextTokenId == _tokenizer.EosTokenId)
            {
                break;
            }
        }

        // Switch back to training mode (in case we continue training after generation)
        _model.train();

        // Decode the full sequence back to text
        return _tokenizer.Decode(tokens);
    }
}
